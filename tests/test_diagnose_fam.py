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


def test_apply_and_restore_forward_round_trip() -> None:
    """After restore, every FAM's forward behaves exactly as the original."""
    from spectrafan.diagnose_fam import (
        _apply_forward,
        _patched_forward_zero,
        _restore_forward,
    )
    from spectrafan.unet import FANetMini

    model = FANetMini()
    x = torch.randn(1, 3, 32, 32)

    baseline = model(x)

    _apply_forward(model, _patched_forward_zero)
    patched = model(x)
    # Identity patch on FAMs changes the network output (skips go through
    # unfiltered), so this must differ from baseline.
    assert not torch.allclose(patched, baseline)

    _restore_forward(model)
    restored = model(x)
    assert torch.allclose(restored, baseline, atol=1e-6)


def test_apply_forward_touches_every_fam() -> None:
    """_apply_forward binds onto every FAMComplex in the model."""
    from spectrafan.diagnose_fam import _apply_forward, _patched_forward_zero, _restore_forward
    from spectrafan.fam import FAMComplex
    from spectrafan.unet import FANetMini

    model = FANetMini()
    fams = [m for m in model.modules() if isinstance(m, FAMComplex)]
    assert len(fams) == 3  # FANetMini has 3 scales

    _apply_forward(model, _patched_forward_zero)
    try:
        for fam in fams:
            assert fam.forward.__func__ is _patched_forward_zero
    finally:
        _restore_forward(model)
