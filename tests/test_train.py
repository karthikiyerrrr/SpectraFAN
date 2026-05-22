"""Tests for spectrafan.train."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch
from torch.utils.data import Dataset

from spectrafan.train import (
    AugConfig,
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    TrainConfig,
    fit,
)


class _SyntheticPairs(Dataset):
    """Tiny in-memory dataset: 8 random (3, 32, 32) images + binary masks."""

    def __init__(self, n: int = 8, size: int = 32) -> None:
        torch.manual_seed(0)
        self.images = torch.rand(n, 3, size, size)
        # Mask is a soft threshold on a low-frequency version of the image so the model
        # has something learnable, not pure noise.
        smoothed = torch.nn.functional.avg_pool2d(self.images.mean(1, keepdim=True), 8, 1, 4)
        smoothed = smoothed[..., :size, :size]  # crop back to (size, size) after padding
        self.masks = (smoothed > smoothed.median()).float()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx]


def _tiny_cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            channels=(8, 16, 32, 64),
            bottleneck=128,
            fam_conv_kind="depthwise",
        ),
        data=DataConfig(
            dataset="synthetic",
            image_size=32,
            batch_size=4,
            subset_size=None,
            val_subset_size=None,
            splits_dir=Path("data/splits/temimagenet_v1"),
        ),
        aug=AugConfig(),
        optim=OptimConfig(lr=1e-3),  # bumped from 1e-5 so 3 epochs is enough to move the loss
        train=TrainConfig(
            epochs=3,
            seed=0,
            device="cpu",
            run_root=tmp_path / "runs",
            loss_ce_weight=0.5,
            loss_dice_weight=0.5,
        ),
    )


def test_fit_one_epoch_decreases_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three epochs on a tiny synthetic Dataset; train loss must decrease."""
    ds = _SyntheticPairs()
    # Inject our synthetic dataset by monkey-patching the dataset factory.
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    run_dir = fit(cfg)

    metrics_path = run_dir / "metrics.parquet"
    assert metrics_path.is_file()
    df = pl.read_parquet(metrics_path)
    assert df.height == 3
    assert {"epoch", "lr", "train_loss", "val_loss", "train_iou", "val_iou"} <= set(df.columns)
    train_losses = df["train_loss"].to_list()
    assert train_losses[-1] < train_losses[0], f"train loss did not decrease: {train_losses}"

    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "config.yaml").is_file()


def test_fit_with_amp_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() with amp=True runs end-to-end on CPU (autocast is a no-op there) and produces finite metrics."""
    ds = _SyntheticPairs()
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    cfg.train.amp = True
    run_dir = fit(cfg)

    df = pl.read_parquet(run_dir / "metrics.parquet")
    assert df.height == cfg.train.epochs
    for col in ("train_loss", "val_loss", "train_iou", "val_iou"):
        values = df[col].to_list()
        assert all(v == v for v in values), f"{col} has NaN: {values}"


def _latest_run_dir(root: Path) -> Path:
    if not root.is_dir():
        pytest.skip(f"{root} does not exist; execute the smoke trial first")
    runs = sorted((p for p in root.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not runs:
        pytest.skip(f"no runs in {root}; execute the smoke trial first")
    return runs[-1]


@pytest.mark.slow
def test_smoke_trial_acceptance() -> None:
    """Validate the most recent run in runs/ meets the smoke trial gate.

    Pre-condition: ``uv run python -m spectrafan.train --config configs/smoke.yaml`` has
    been executed at least once. This test does NOT launch training.
    """
    run_dir = _latest_run_dir(Path("runs"))
    metrics_path = run_dir / "metrics.parquet"
    assert metrics_path.is_file(), f"no metrics.parquet under {run_dir}"

    df = pl.read_parquet(metrics_path)
    val_iou = df["val_iou"].to_list()

    assert df.height == 10, f"expected 10 epochs, got {df.height}"
    assert val_iou[2] > val_iou[1] > val_iou[0], (
        f"val_iou not strictly increasing over the first 3 epochs: {val_iou[:3]}"
    )
    assert val_iou[-1] > 0.5, f"final val_iou {val_iou[-1]:.4f} <= 0.5"

    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    # Round-trip the checkpoint to make sure it's loadable.
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt


def test_epoch_checkpoints_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With checkpoint_every=2 and 4 epochs, epoch_002.pt and epoch_004.pt are saved."""
    ds = _SyntheticPairs()
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    cfg.train.epochs = 4
    cfg.train.checkpoint_every = 2
    run_dir = fit(cfg)

    assert (run_dir / "epoch_002.pt").is_file()
    assert (run_dir / "epoch_004.pt").is_file()
    assert not (run_dir / "epoch_001.pt").exists()
    assert not (run_dir / "epoch_003.pt").exists()
    # best.pt + last.pt still saved every epoch
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "last.pt").is_file()


def test_env_json_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() writes env.json with expected keys."""
    import json as _json

    ds = _SyntheticPairs()
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    run_dir = fit(cfg)

    env_path = run_dir / "env.json"
    assert env_path.is_file()
    env = _json.loads(env_path.read_text())
    expected_keys = {
        "python_version", "torch_version", "cuda_available", "cuda_version",
        "device", "platform", "git_sha", "git_dirty", "hostname", "amp_enabled",
    }
    assert expected_keys <= set(env), f"missing keys: {expected_keys - set(env)}"
    assert env["device"] == "cpu"
    assert env["amp_enabled"] is False


def test_set_global_seed_enables_determinism() -> None:
    """set_global_seed turns on torch.use_deterministic_algorithms (warn_only)."""
    from spectrafan.train import set_global_seed

    set_global_seed(0)
    assert torch.are_deterministic_algorithms_enabled()
