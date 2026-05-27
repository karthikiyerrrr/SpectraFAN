"""Frequency Attention Module — complex-FFT reproduction of Liu et al. 2026.

FAMComplex implements the FAM as drawn in Fig. 2(a):

    x ──► fft2 ──► split real/imag ──► per-part branch ──► combine ──► ifft2 ──►
       └────────────────── residual add ─────────────────────────────► + ──► conv1x1 ──►

Two `conv_kind` variants are exposed behind a constructor flag so the
depthwise-vs-depthwise-separable ambiguity in the paper can be settled by
parameter/GFLOP calibration against Table 1.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

ConvKind = Literal["depthwise", "depthwise_separable"]


def _make_branch_conv(channels: int, conv_kind: ConvKind) -> nn.Module:
    """One 3x3 conv stage inside a FAM branch.

    `depthwise`           = a single depthwise 3x3.
    `depthwise_separable` = depthwise 3x3 followed by a pointwise 1x1.
    """
    depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
    if conv_kind == "depthwise":
        return depthwise
    if conv_kind == "depthwise_separable":
        pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        return nn.Sequential(depthwise, pointwise)
    raise ValueError(f"unknown conv_kind: {conv_kind!r}")


class _Branch(nn.Module):
    """conv -> BN -> ReLU -> conv. Operates on a single real tensor."""

    def __init__(self, channels: int, conv_kind: ConvKind) -> None:
        super().__init__()
        self.conv1 = _make_branch_conv(channels, conv_kind)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = _make_branch_conv(channels, conv_kind)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.bn(self.conv1(z))))


class FAMComplex(nn.Module):
    """Frequency Attention Module — complex-FFT variant.

    Args:
        channels: input/output channel count (preserved).
        conv_kind: per-branch conv flavor; see _make_branch_conv.
    """

    def __init__(self, channels: int, conv_kind: ConvKind = "depthwise") -> None:
        super().__init__()
        self.channels = channels
        self.conv_kind = conv_kind
        self.branch_real = _Branch(channels, conv_kind)
        self.branch_imag = _Branch(channels, conv_kind)
        self.final = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.fft.fft2 doesn't support bf16/fp16, so escape any outer autocast
        # and run the frequency-domain block in fp32. The rest of the network keeps
        # its autocast benefit; only this module pays the precision tax.
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            freq = torch.fft.fft2(x_fp32)
            r_prime = self.branch_real(freq.real)
            i_prime = self.branch_imag(freq.imag)
            freq_hat = torch.complex(r_prime, i_prime)
            spatial_hat = torch.fft.ifft2(freq_hat).real
            out = self.final(x_fp32 + spatial_hat)
        return out.to(orig_dtype)


from spectrafan.models._registries import SKIP_TRANSFORM_REGISTRY  # noqa: E402


@SKIP_TRANSFORM_REGISTRY.register("fam_complex")
def _build_fam_complex(channels: int, conv_kind: ConvKind = "depthwise") -> nn.Module:
    return FAMComplex(channels, conv_kind=conv_kind)
