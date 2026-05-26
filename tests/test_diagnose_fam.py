"""Tests for spectrafan.diagnose_fam."""

from __future__ import annotations

import torch

from spectrafan.fam import FAMComplex


def test_module_imports() -> None:
    """diagnose_fam imports cleanly and exposes diagnose_run."""
    from spectrafan import diagnose_fam

    assert hasattr(diagnose_fam, "diagnose_run")


def test_patched_forward_skip_fft_preserves_shape_and_dtype() -> None:
    """fam_skip_fft drops the FFT pathway but keeps the learned 1x1 projection."""
    from spectrafan.diagnose_fam import _patched_forward_skip_fft

    fam = FAMComplex(channels=8)
    x = torch.randn(2, 8, 16, 16)
    out = _patched_forward_skip_fft(fam, x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # Must equal fam.final(x) (the 1x1 conv) exactly — no FFT pathway.
    expected = fam.final(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_patched_forward_zero_returns_identity() -> None:
    """fam_zero returns the input unchanged — full FAM-block ablation."""
    from spectrafan.diagnose_fam import _patched_forward_zero

    fam = FAMComplex(channels=8)
    x = torch.randn(2, 8, 16, 16)
    out = _patched_forward_zero(fam, x)
    assert out.shape == x.shape
    assert torch.allclose(out, x)
