"""U-Net backbone with FAM injected on every skip connection.

Channel widths are `channels=(64, 128, 256, 512)` per Liu et al. 2026 §3.2.
The bottleneck width is a constructor argument so calibration can settle
the standard 1024 vs the literal-paper-reading 512 question.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from spectrafan.models.blocks import (
    DecoderModule,
    DoubleConv,
    EncoderModule,
    OutputConv,
    OutputNorm,
)
from spectrafan.models.fam import ConvKind, FAMComplex


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


class FANetMini(FANet):
    """FANet sized for TEMImageNet (256x256 binary atom masks).

    Encoder widths (32, 64, 128) with a 256-channel bottleneck, mirroring
    AtomSegNet's depth/width on the same dataset. Three FAMs on the skips;
    FAM block, encoder/decoder blocks, and output head are inherited from
    FANet without modification.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_kind: ConvKind = "depthwise",
        out_channels: int = 1,
        output_norm: OutputNorm = "bn",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            channels=(32, 64, 128),
            bottleneck=256,
            conv_kind=conv_kind,
            out_channels=out_channels,
            output_norm=output_norm,
        )
