"""AtomSegNet segmentation U-Net (Lin et al., Sci. Rep. 2021), adapted to TEMImageNet.

Faithful to the paper's Fig. 4 topology: a two-pool encoder (64 -> 128) with a
256-channel bottleneck and a bilinear-upsample + 1x1-conv decoder. Two deviations,
both required by this project's pipeline and both improving comparability with
FANetMini: 3-channel input (TEMImageNet is loaded 3-channel) and a logit head
(no in-model sigmoid; training uses BCEWithLogitsLoss).

A configurable skip transform on each skip gives the FAM ablation: "identity"
(no FAM) vs "fam_complex" (FAM on each skip). The fam_complex arm is FAMSegNet.
"""

from __future__ import annotations

import torch
from torch import nn

from spectrafan.models._registries import SKIP_TRANSFORM_REGISTRY
from spectrafan.models.blocks import (
    BilinearUp,
    DoubleConv,
    EncoderModule,
    OutputConv,
    OutputNorm,
)
from spectrafan.models.fam import ConvKind


class AtomSegNet(nn.Module):
    """Faithful AtomSegNet U-Net with a configurable skip transform.

    The per-skip module list is named ``skip_transforms`` (rather than FANet's ``fams``)
    because it may hold either FAMs or the identity pass-through.

    Args:
        in_channels: input image channels (3 for TEMImageNet's RGB-formatted inputs).
        conv_kind: passed through to FAM instances on the skips.
        skip_transform: registry key for the per-skip transform ("identity" = no FAM,
            "fam_complex" = FAMSegNet).
        out_channels: segmentation mask channels (1 for binary).
        output_norm: normalization in the OutputConv head ("bn" | "none" | "groupnorm").
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_kind: ConvKind = "depthwise",
        skip_transform: str = "identity",
        out_channels: int = 1,
        output_norm: OutputNorm = "bn",
    ) -> None:
        super().__init__()
        self.stem = DoubleConv(in_channels, 64)  # skip A (64 ch, full res)
        self.down = EncoderModule(64, 128)  # MaxPool -> DoubleConv -> skip B (128 ch, /2)
        self.bottleneck_module = EncoderModule(128, 256)  # MaxPool -> DoubleConv (256 ch, /4)
        self.up1 = BilinearUp(256, 128)  # -> concat skip B -> 128 (/2)
        self.up2 = BilinearUp(128, 64)  # -> concat skip A -> 64 (full res)
        self.skip_transforms = nn.ModuleList(
            [
                SKIP_TRANSFORM_REGISTRY.build(skip_transform, channels=64, conv_kind=conv_kind),
                SKIP_TRANSFORM_REGISTRY.build(skip_transform, channels=128, conv_kind=conv_kind),
            ]
        )
        self.out = OutputConv(64, out_channels, norm=output_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.stem(x)  # 64, full
        b = self.down(a)  # 128, /2
        h = self.bottleneck_module(b)  # 256, /4
        a_f = self.skip_transforms[0](a)
        b_f = self.skip_transforms[1](b)
        h = self.up1(h, b_f)  # 128, /2
        h = self.up2(h, a_f)  # 64, full
        return self.out(h)


# Architecture registration (kept at end to avoid an import cycle via the package __init__).
from spectrafan.config import DataConfig, ModelConfig  # noqa: E402
from spectrafan.models._registries import MODEL_REGISTRY  # noqa: E402


@MODEL_REGISTRY.register("atomsegnet")
def _build_atomsegnet(model_cfg: ModelConfig, data_cfg: DataConfig) -> nn.Module:
    # No-FAM arm: skip transform fixed to identity regardless of config default.
    return AtomSegNet(
        in_channels=data_cfg.in_channels,
        skip_transform="identity",
        output_norm=model_cfg.output_norm,
    )


@MODEL_REGISTRY.register("famsegnet")
def _build_famsegnet(model_cfg: ModelConfig, data_cfg: DataConfig) -> nn.Module:
    # FAM arm: FAMComplex on each skip.
    return AtomSegNet(
        in_channels=data_cfg.in_channels,
        conv_kind=model_cfg.fam_conv_kind,
        skip_transform="fam_complex",
        output_norm=model_cfg.output_norm,
    )
