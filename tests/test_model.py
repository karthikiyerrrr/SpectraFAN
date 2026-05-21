"""Tests for FAMComplex and FANet — architecture and calibration."""

from __future__ import annotations

import pytest
import torch

from spectrafan.fam import ConvKind, FAMComplex

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
