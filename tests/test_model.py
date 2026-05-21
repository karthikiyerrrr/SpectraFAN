"""Tests for FAMComplex and FANet — architecture and calibration."""

from __future__ import annotations

import pytest
import torch

from spectrafan.fam import FAMComplex

CONV_KINDS = ["depthwise", "depthwise_separable"]


@pytest.mark.parametrize("conv_kind", CONV_KINDS)
def test_fam_shape_roundtrip(conv_kind: str) -> None:
    """FAM preserves shape and dtype, produces no NaNs."""
    torch.manual_seed(0)
    fam = FAMComplex(channels=64, conv_kind=conv_kind)
    x = torch.randn(2, 64, 32, 32)
    y = fam(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("conv_kind", CONV_KINDS)
def test_fam_imag_residue_small(conv_kind: str) -> None:
    """Pre-`.real` IFFT output has near-zero imaginary part.

    The forward pass discards `.imag` after IFFT. Without this guard, a bug
    that leaves a meaningful imaginary component would be silently swallowed
    and only surface as a downstream accuracy regression.
    """
    torch.manual_seed(0)
    fam = FAMComplex(channels=16, conv_kind=conv_kind)
    x = torch.randn(1, 16, 64, 64)
    with torch.no_grad():
        complex_out = fam._filter_in_frequency(x)
    assert complex_out.is_complex()
    assert complex_out.imag.abs().mean().item() < 1e-5
