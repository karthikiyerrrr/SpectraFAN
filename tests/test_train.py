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
