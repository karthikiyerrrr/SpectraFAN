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


def test_collector_starts_empty_and_records_per_call() -> None:
    """Each instrumented forward call appends one row tagged with the FAM's scale_idx."""
    from spectrafan.diagnose_fam import (
        _FamStatsCollector,
        _make_instrumented_forward,
    )

    fam = FAMComplex(channels=8)
    collector = _FamStatsCollector()
    instrumented = _make_instrumented_forward(collector, scale_idx=1)

    x = torch.randn(2, 8, 16, 16)
    out = instrumented(fam, x)

    assert out.shape == x.shape
    assert len(collector.rows) == 1
    row = collector.rows[0]
    expected_cols = {
        "scale_idx",
        "batch_idx",
        "input_norm",
        "contribution_norm",
        "output_norm",
        "branch_real_out_norm",
        "branch_imag_out_norm",
        "branch_real_dead_rate",
        "branch_imag_dead_rate",
        "fft_real_dc_share",
        "fft_real_p99_to_median",
    }
    assert set(row) == expected_cols
    assert row["scale_idx"] == 1
    assert row["batch_idx"] == 0  # first call


def test_collector_increments_batch_idx_within_scale() -> None:
    """Subsequent calls on the same scale advance batch_idx; different scales have their own counters."""
    from spectrafan.diagnose_fam import _FamStatsCollector, _make_instrumented_forward

    fam = FAMComplex(channels=8)
    collector = _FamStatsCollector()
    forward_s0 = _make_instrumented_forward(collector, scale_idx=0)
    forward_s1 = _make_instrumented_forward(collector, scale_idx=1)

    x = torch.randn(2, 8, 16, 16)
    forward_s0(fam, x)
    forward_s0(fam, x)
    forward_s1(fam, x)

    by_scale = {(r["scale_idx"], r["batch_idx"]) for r in collector.rows}
    assert by_scale == {(0, 0), (0, 1), (1, 0)}


def test_instrumented_forward_matches_original() -> None:
    """Instrumented output equals the original FAMComplex.forward output."""
    from spectrafan.diagnose_fam import _FamStatsCollector, _make_instrumented_forward

    fam = FAMComplex(channels=8)
    collector = _FamStatsCollector()
    instrumented = _make_instrumented_forward(collector, scale_idx=0)

    x = torch.randn(2, 8, 16, 16)
    expected = fam.forward(x)  # original, unbound at this point
    actual = instrumented(fam, x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_collector_values_are_finite() -> None:
    """All recorded stats are finite floats in plausible ranges."""
    import math

    from spectrafan.diagnose_fam import _FamStatsCollector, _make_instrumented_forward

    fam = FAMComplex(channels=8)
    collector = _FamStatsCollector()
    instrumented = _make_instrumented_forward(collector, scale_idx=0)
    instrumented(fam, torch.randn(2, 8, 16, 16))

    row = collector.rows[0]
    for k, v in row.items():
        if k in ("scale_idx", "batch_idx"):
            continue
        assert math.isfinite(v), f"{k} is not finite: {v}"
    assert 0.0 <= row["branch_real_dead_rate"] <= 1.0
    assert 0.0 <= row["branch_imag_dead_rate"] <= 1.0
    assert 0.0 <= row["fft_real_dc_share"] <= 1.0
