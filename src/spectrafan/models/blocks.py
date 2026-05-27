"""Reusable U-Net building blocks."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

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
