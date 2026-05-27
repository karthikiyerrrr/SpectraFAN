"""Tests for spectrafan.predict."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import Dataset


def test_find_latest_run_resolves_most_recent_match(tmp_path: Path) -> None:
    """Given multiple runs/*_<suffix>/ dirs, find_latest_run returns the one
    with the highest mtime that matches the suffix."""
    from spectrafan.analysis.predict import find_latest_run

    older = tmp_path / "2026-05-20_010101_fanetmini"
    newer = tmp_path / "2026-05-22_020202_fanetmini"
    other = tmp_path / "2026-05-23_030303_full_repro"
    for d in (older, newer, other):
        d.mkdir()
    # Force mtimes (mkdir order is not guaranteed to set them in the desired sequence).
    now = time.time()
    import os

    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now - 10, now - 10))
    os.utime(other, (now, now))  # newest overall, but wrong suffix

    assert find_latest_run(tmp_path, "fanetmini") == newer


def test_find_latest_run_no_match_raises(tmp_path: Path) -> None:
    """find_latest_run raises FileNotFoundError when no dir matches the suffix."""
    from spectrafan.analysis.predict import find_latest_run

    (tmp_path / "2026-05-20_010101_full_repro").mkdir()

    with pytest.raises(FileNotFoundError, match="no .*_fanetmini/ dirs found"):
        find_latest_run(tmp_path, "fanetmini")


class _FakeTEMDataset(Dataset):
    """Mirrors TEMImageNetDataset's constructor signature for predict.py tests.

    Yields random (3, image_size, image_size) images and binary (1, H, W) masks.
    `_rows[i]["stem"]` is exposed so predict.py's id extraction works unchanged.
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
        n_total = 40
        n = min(subset_size, n_total) if subset_size is not None else n_total
        # Deterministic per split so different splits give different content.
        g = torch.Generator().manual_seed(hash(split) & 0xFFFFFFFF)
        self._rows = [{"stem": f"{split}_{i:03d}"} for i in range(n)]
        self._images = torch.rand(n, 3, image_size, image_size, generator=g)
        self._masks = (self._images.mean(1, keepdim=True) > 0.5).float()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[idx], self._masks[idx]


def _write_fanetmini_run_dir(run_dir: Path, image_size: int = 32) -> None:
    """Write a config.yaml + best.pt that predict_run can load."""
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
    from spectrafan.models.unet import FANetMini

    model = FANetMini()
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 4, "val_iou": 0.5123},
        run_dir / "best.pt",
    )


def test_predict_writes_expected_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """predict_run writes predictions.npz + test_metrics.json with the spec's shapes."""
    from spectrafan.analysis import predict as predict_mod

    run_dir = tmp_path / "2026-05-23_000000_fanetmini"
    _write_fanetmini_run_dir(run_dir, image_size=32)

    monkeypatch.setattr(predict_mod, "TEMImageNetDataset", _FakeTEMDataset)

    predict_mod.predict_run(run_dir)

    npz_path = run_dir / "predictions.npz"
    json_path = run_dir / "test_metrics.json"
    assert npz_path.is_file()
    assert json_path.is_file()

    with np.load(npz_path) as artifact:
        keys = set(artifact.files)
        assert keys == {
            "val_images",
            "val_masks",
            "val_preds",
            "val_ids",
            "test_images",
            "test_masks",
            "test_preds",
            "test_ids",
        }
        assert artifact["val_images"].shape == (16, 1, 32, 32)
        assert artifact["val_images"].dtype == np.float32
        assert artifact["val_masks"].shape == (16, 1, 32, 32)
        assert artifact["val_masks"].dtype == np.uint8
        assert artifact["val_preds"].shape == (16, 1, 32, 32)
        assert artifact["val_preds"].dtype == np.uint8
        assert artifact["val_ids"].shape == (16,)
        assert artifact["val_ids"].dtype.kind == "U"
        assert artifact["test_images"].shape == (16, 1, 32, 32)
        assert artifact["test_ids"][0].startswith("test_")

    metrics = json.loads(json_path.read_text())
    assert set(metrics) == {"epoch", "val_iou", "test_iou", "test_dice", "test_px_acc", "test_size"}
    assert metrics["epoch"] == 5  # ckpt["epoch"] is 4 -> 1-indexed = 5
    assert metrics["val_iou"] == pytest.approx(0.5123)
    assert metrics["test_size"] == 40  # _FakeTEMDataset's n_total when subset_size is None
    assert 0.0 <= metrics["test_iou"] <= 1.0
    assert 0.0 <= metrics["test_dice"] <= 1.0
    assert 0.0 <= metrics["test_px_acc"] <= 1.0


def _write_fanet_run_dir(run_dir: Path, image_size: int = 32) -> None:
    """Write a config.yaml + best.pt for paper-FANet with tiny widths (fast test)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        "model": {
            "name": "fanet",
            "channels": [8, 16, 32, 64],
            "bottleneck": 128,
        },
        "data": {
            "image_size": image_size,
            "batch_size": 4,
            "root": "data/raw/temimagenet",
            "splits_dir": "data/splits/temimagenet_v1",
        },
        "train": {"device": "cpu"},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))
    from spectrafan.models.unet import FANet

    model = FANet(channels=(8, 16, 32, 64), bottleneck=128)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 9, "val_iou": 0.6234},
        run_dir / "best.pt",
    )


def test_predict_works_for_fanet_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """predict_run dispatches through build_model and works for FANet runs too."""
    from spectrafan.analysis import predict as predict_mod

    run_dir = tmp_path / "2026-05-23_000000_full_repro"
    _write_fanet_run_dir(run_dir, image_size=32)

    monkeypatch.setattr(predict_mod, "TEMImageNetDataset", _FakeTEMDataset)

    predict_mod.predict_run(run_dir)

    assert (run_dir / "predictions.npz").is_file()
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    assert metrics["epoch"] == 10
    assert metrics["val_iou"] == pytest.approx(0.6234)


def test_predict_cli_requires_run_or_latest(monkeypatch) -> None:
    """argparse rejects invocation with neither --run nor --latest."""
    from spectrafan.analysis import predict as predict_mod

    monkeypatch.setattr("sys.argv", ["spectrafan.predict"])
    with pytest.raises(SystemExit):
        predict_mod.main()


def test_predict_cli_rejects_both_run_and_latest(monkeypatch, tmp_path: Path) -> None:
    """argparse rejects invocation with both --run and --latest (mutually exclusive)."""
    from spectrafan.analysis import predict as predict_mod

    monkeypatch.setattr(
        "sys.argv",
        ["spectrafan.predict", "--run", str(tmp_path), "--latest", "fanetmini"],
    )
    with pytest.raises(SystemExit):
        predict_mod.main()


def test_predict_cli_runs_predict_with_run_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI with --run <path> invokes predict_run on that path."""
    from spectrafan.analysis import predict as predict_mod

    run_dir = tmp_path / "2026-05-23_000000_fanetmini"
    _write_fanetmini_run_dir(run_dir, image_size=32)
    monkeypatch.setattr(predict_mod, "TEMImageNetDataset", _FakeTEMDataset)
    monkeypatch.setattr("sys.argv", ["spectrafan.predict", "--run", str(run_dir)])

    predict_mod.main()

    assert (run_dir / "predictions.npz").is_file()
    assert (run_dir / "test_metrics.json").is_file()


def test_predict_cli_latest_resolves_against_runs_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI with --latest <suffix> --runs-root <dir> resolves and runs predict."""
    from spectrafan.analysis import predict as predict_mod

    run_dir = tmp_path / "2026-05-23_000000_fanetmini"
    _write_fanetmini_run_dir(run_dir, image_size=32)
    monkeypatch.setattr(predict_mod, "TEMImageNetDataset", _FakeTEMDataset)
    monkeypatch.setattr(
        "sys.argv",
        [
            "spectrafan.predict",
            "--latest",
            "fanetmini",
            "--runs-root",
            str(tmp_path),
        ],
    )

    predict_mod.main()

    assert (run_dir / "predictions.npz").is_file()
