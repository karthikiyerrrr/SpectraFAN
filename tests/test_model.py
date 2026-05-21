"""Tests for FAMComplex and FANet — architecture and calibration."""

from __future__ import annotations

import pytest
import torch

from spectrafan.fam import ConvKind, FAMComplex
from spectrafan.unet import FANet

CONV_KINDS: list[ConvKind] = ["depthwise", "depthwise_separable"]


@pytest.mark.parametrize("conv_kind", CONV_KINDS)
def test_fam_shape_roundtrip(conv_kind: ConvKind) -> None:
    """FAM preserves shape and dtype, produces no NaNs."""
    torch.manual_seed(0)
    fam = FAMComplex(channels=64, conv_kind=conv_kind)
    x = torch.randn(2, 64, 32, 32)
    y = fam(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_fanet_shape() -> None:
    """FANet maps (B, 3, 512, 512) to (B, 1, 512, 512) with values in [0, 1]."""
    torch.manual_seed(0)
    model = FANet()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 512, 512)
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    assert (y >= 0).all() and (y <= 1).all()
