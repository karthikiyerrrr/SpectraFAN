"""U-Net backbone with FAM injected on every skip connection.

Channel widths are `channels=(64, 128, 256, 512)` per Liu et al. 2026 §3.2.
The bottleneck width is a constructor argument so calibration can settle
the standard 1024 vs the literal-paper-reading 512 question.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from spectrafan.models._registries import SKIP_TRANSFORM_REGISTRY
from spectrafan.models.blocks import (
    DecoderModule,
    DoubleConv,
    EncoderModule,
    OutputConv,
    OutputNorm,
)
from spectrafan.models.fam import ConvKind


class FANet(nn.Module):
    """U-Net backbone with a configurable skip-connection transform on every skip.

    Args:
        in_channels: input image channels (3 for RGB-formatted inputs).
        channels: encoder channel widths at each scale.
        bottleneck: bottleneck channel width (next power of 2 above
            channels[-1] by U-Net convention, or channels[-1] if the
            paper's "four scales" wording is meant literally).
        conv_kind: passed through to every FAM instance.
        skip_transform: registry key naming the skip-transform variant built
            on each skip (default "fam_complex").
        out_channels: segmentation mask channels (1 for binary).
        output_norm: normalization applied in the OutputConv head
            ("bn" | "none" | "groupnorm").
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 512),
        bottleneck: int = 1024,
        conv_kind: ConvKind = "depthwise",
        skip_transform: str = "fam_complex",
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
        self.fams = nn.ModuleList(
            [
                SKIP_TRANSFORM_REGISTRY.build(skip_transform, channels=c, conv_kind=conv_kind)
                for c in self.channels
            ]
        )

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

    Encoder widths (32, 64, 128) with a 256-channel bottleneck: a three-scale
    encoder that reaches the same 32x32 bottleneck resolution as FANet does at
    512x512, one downsample shallower to suit the 256x256 input. Three FAMs on
    the skips; FAM block, encoder/decoder blocks, and output head are inherited
    from FANet without modification.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_kind: ConvKind = "depthwise",
        skip_transform: str = "fam_complex",
        out_channels: int = 1,
        output_norm: OutputNorm = "bn",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            channels=(32, 64, 128),
            bottleneck=256,
            conv_kind=conv_kind,
            skip_transform=skip_transform,
            out_channels=out_channels,
            output_norm=output_norm,
        )


# Architecture registration (kept at end to avoid an import cycle via the package __init__).
from spectrafan.config import DataConfig, ModelConfig  # noqa: E402
from spectrafan.models._registries import MODEL_REGISTRY  # noqa: E402


@MODEL_REGISTRY.register("fanet")
def _build_fanet(model_cfg: ModelConfig, data_cfg: DataConfig) -> nn.Module:
    return FANet(
        in_channels=data_cfg.in_channels,
        channels=tuple(model_cfg.channels),
        bottleneck=model_cfg.bottleneck,
        conv_kind=model_cfg.fam_conv_kind,
        skip_transform=model_cfg.skip_transform,
        output_norm=model_cfg.output_norm,
    )


@MODEL_REGISTRY.register("fanetmini")
def _build_fanetmini(model_cfg: ModelConfig, data_cfg: DataConfig) -> nn.Module:
    return FANetMini(
        in_channels=data_cfg.in_channels,
        conv_kind=model_cfg.fam_conv_kind,
        skip_transform=model_cfg.skip_transform,
        output_norm=model_cfg.output_norm,
    )
