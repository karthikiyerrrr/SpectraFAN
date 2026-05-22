"""U-Net backbone with FAM injected on every skip connection.

Channel widths are `channels=(64, 128, 256, 512)` per Liu et al. 2026 §3.2.
The bottleneck width is a constructor argument so calibration can settle
the standard 1024 vs the literal-paper-reading 512 question.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from spectrafan.fam import ConvKind, FAMComplex

OutputNorm = Literal["bn", "none", "groupnorm"]


class DoubleConv(nn.Module):
    """Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderModule(nn.Module):
    """MaxPool 2x2 -> DoubleConv. Halves spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DecoderModule(nn.Module):
    """ConvTranspose 2x2 (stride 2) -> concat skip -> DoubleConv.

    The DoubleConv input width is `out_channels * 2` because the skip
    (already FAM-filtered, same channel count as the upsampled input)
    is concatenated along the channel axis.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OutputConv(nn.Module):
    """Conv 1x1 -> BN. Returns logits; callers apply sigmoid externally for probabilities.

    Note: deviates from the paper's Fig. 2(e), which draws this block as
    Conv -> BN -> ReLU -> Sigmoid. The ReLU before Sigmoid would clamp the
    output to [0.5, 1.0] (sigmoid of any non-negative number is >= 0.5),
    making confident background predictions impossible. The Sigmoid is also
    dropped from the module so that the training loop can use
    BCEWithLogitsLoss for numerical stability; downstream code applies
    ``torch.sigmoid`` explicitly when probabilities are needed (e.g. metrics,
    visualization). See docs/superpowers/notes/03_architecture_deviations.md.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        norm: OutputNorm = "bn",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm: nn.Module
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "groupnorm":
            # GroupNorm with one group computes per-sample statistics at both
            # train and eval time, so there is no train/eval running-stats skew.
            self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unknown OutputConv norm: {norm!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class FANet(nn.Module):
    """U-Net backbone with FAMComplex on every skip connection.

    Args:
        in_channels: input image channels (3 for RGB-formatted inputs).
        channels: encoder channel widths at each scale.
        bottleneck: bottleneck channel width (next power of 2 above
            channels[-1] by U-Net convention, or channels[-1] if the
            paper's "four scales" wording is meant literally).
        conv_kind: passed through to every FAM instance.
        out_channels: segmentation mask channels (1 for binary).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 512),
        bottleneck: int = 1024,
        conv_kind: ConvKind = "depthwise",
        out_channels: int = 1,
        output_norm: OutputNorm = "bn",
    ) -> None:
        super().__init__()
        self.channels = tuple(channels)
        self.bottleneck = bottleneck

        # Encoder stem + down-stages.
        self.stem = DoubleConv(in_channels, self.channels[0])
        self.encoders = nn.ModuleList()
        prev = self.channels[0]
        for c in self.channels[1:]:
            self.encoders.append(EncoderModule(prev, c))
            prev = c
        # Bottleneck is one more EncoderModule that does not produce a skip.
        self.bottleneck_module = EncoderModule(self.channels[-1], bottleneck)

        # One FAM per skip (one per encoder scale, including the stem).
        self.fams = nn.ModuleList([FAMComplex(c, conv_kind=conv_kind) for c in self.channels])

        # Decoder up-stages, mirrored against encoder widths.
        # In channels of each decoder = output of previous stage (starts at bottleneck).
        # Out channels of each decoder = the skip's channel width at this scale.
        self.decoders = nn.ModuleList()
        prev = bottleneck
        for c in reversed(self.channels):
            self.decoders.append(DecoderModule(prev, c))
            prev = c

        self.out = OutputConv(self.channels[0], out_channels, norm=output_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = self.stem(x)
        skips.append(h)
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
        h = self.bottleneck_module(h)

        # Apply FAM to each skip, then walk the decoders top-down.
        filtered = [fam(s) for fam, s in zip(self.fams, skips, strict=True)]
        for dec, skip in zip(self.decoders, reversed(filtered), strict=True):
            h = dec(h, skip)
        return self.out(h)
