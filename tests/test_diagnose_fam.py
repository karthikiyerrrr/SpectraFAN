"""Tests for spectrafan.diagnose_fam."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch
import yaml
from torch.utils.data import Dataset

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


def test_run_val_once_returns_iou_in_unit_interval() -> None:
    """_run_val_once runs the model once over the loader and returns val_iou in [0, 1]."""
    from torch.utils.data import DataLoader, TensorDataset

    from spectrafan.diagnose_fam import _run_val_once
    from spectrafan.unet import FANetMini

    images = torch.rand(8, 3, 32, 32)
    masks = (images.mean(1, keepdim=True) > 0.5).float()
    loader = DataLoader(TensorDataset(images, masks), batch_size=4)

    model = FANetMini()
    iou = _run_val_once(model, loader, device=torch.device("cpu"))
    assert isinstance(iou, float)
    assert 0.0 <= iou <= 1.0


class _FakeTEMDataset(Dataset):
    """Mirrors TEMImageNetDataset's constructor signature for diagnose_fam tests.

    Yields random (3, image_size, image_size) images and binary (1, H, W) masks.
    """

    def __init__(
        self,
        root,
        split: str,
        image_size: int,
        splits_dir=Path("data/splits/temimagenet_v1"),
        transforms=None,
        subset_size: int | None = None,
    ) -> None:
        n_total = 8
        n = min(subset_size, n_total) if subset_size is not None else n_total
        g = torch.Generator().manual_seed(hash(split) & 0xFFFFFFFF)
        self._rows = [{"stem": f"{split}_{i:03d}"} for i in range(n)]
        self._images = torch.rand(n, 3, image_size, image_size, generator=g)
        self._masks = (self._images.mean(1, keepdim=True) > 0.5).float()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[idx], self._masks[idx]


def _write_fanetmini_run_dir(run_dir: Path, image_size: int = 32) -> None:
    """Write a config.yaml + best.pt that diagnose_run can load."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        "model": {"name": "fanetmini"},
        "data": {
            "image_size": image_size,
            "batch_size": 4,
            "root": "data/raw/temimagenet",
            "splits_dir": "data/splits/temimagenet_v1",
        },
        "train": {"device": "cpu"},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
    from spectrafan.unet import FANetMini

    model = FANetMini()
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 65, "val_iou": 0.8215},
        run_dir / "best.pt",
    )


def test_diagnose_run_writes_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """diagnose_run writes fam_stats.parquet + fam_diagnosis.json with the spec's shape."""
    from spectrafan import diagnose_fam as diag_mod

    run_dir = tmp_path / "2026-05-25_000000_fanetmini"
    _write_fanetmini_run_dir(run_dir)
    monkeypatch.setattr(diag_mod, "TEMImageNetDataset", _FakeTEMDataset)

    diag_mod.diagnose_run(run_dir)

    parquet_path = run_dir / "fam_stats.parquet"
    json_path = run_dir / "fam_diagnosis.json"
    assert parquet_path.is_file()
    assert json_path.is_file()

    df = pl.read_parquet(parquet_path)
    assert set(df.columns) == {
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
    # FANetMini has 3 FAMs; _FakeTEMDataset has 8 images at batch_size=4 -> 2 batches.
    # So 3 scales * 2 batches = 6 rows.
    assert df.height == 6
    assert set(df["scale_idx"].to_list()) == {0, 1, 2}

    blob = json.loads(json_path.read_text())
    assert blob["epoch"] == 66  # ckpt["epoch"] is 65 -> 1-indexed = 66
    assert blob["n_fam_modules"] == 3
    assert set(blob["val_iou"]) == {"as_trained", "fam_skip_fft", "fam_zero"}
    for mode, value in blob["val_iou"].items():
        assert 0.0 <= value <= 1.0, f"{mode} val_iou out of range: {value}"


def test_diagnose_run_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """diagnose_run raises FileExistsError if either output already exists."""
    from spectrafan import diagnose_fam as diag_mod

    run_dir = tmp_path / "2026-05-25_000000_fanetmini"
    _write_fanetmini_run_dir(run_dir)
    monkeypatch.setattr(diag_mod, "TEMImageNetDataset", _FakeTEMDataset)
    (run_dir / "fam_stats.parquet").write_text("stale")

    with pytest.raises(FileExistsError, match="fam_stats.parquet"):
        diag_mod.diagnose_run(run_dir)


def test_diagnose_run_missing_best_pt_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """diagnose_run raises FileNotFoundError when best.pt is absent."""
    from spectrafan import diagnose_fam as diag_mod

    run_dir = tmp_path / "2026-05-25_000000_fanetmini"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"name": "fanetmini"},
                "data": {
                    "image_size": 32,
                    "batch_size": 4,
                    "root": "data/raw/temimagenet",
                    "splits_dir": "data/splits/temimagenet_v1",
                },
                "train": {"device": "cpu"},
            }
        )
    )
    monkeypatch.setattr(diag_mod, "TEMImageNetDataset", _FakeTEMDataset)

    with pytest.raises(FileNotFoundError, match="best.pt"):
        diag_mod.diagnose_run(run_dir)


def test_diagnose_run_works_for_fanet_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """diagnose_run handles FANet runs (4 FAMs instead of 3) without changes."""
    from spectrafan import diagnose_fam as diag_mod

    run_dir = tmp_path / "2026-05-25_000000_full_repro"
    run_dir.mkdir()
    cfg_dict = {
        "model": {"name": "fanet", "channels": [8, 16, 32, 64], "bottleneck": 128},
        "data": {
            "image_size": 32,
            "batch_size": 4,
            "root": "data/raw/temimagenet",
            "splits_dir": "data/splits/temimagenet_v1",
        },
        "train": {"device": "cpu"},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
    from spectrafan.unet import FANet

    model = FANet(channels=(8, 16, 32, 64), bottleneck=128)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 49, "val_iou": 0.8243},
        run_dir / "best.pt",
    )
    monkeypatch.setattr(diag_mod, "TEMImageNetDataset", _FakeTEMDataset)

    diag_mod.diagnose_run(run_dir)

    df = pl.read_parquet(run_dir / "fam_stats.parquet")
    assert set(df["scale_idx"].to_list()) == {0, 1, 2, 3}  # FANet has 4 FAMs
    blob = json.loads((run_dir / "fam_diagnosis.json").read_text())
    assert blob["n_fam_modules"] == 4
