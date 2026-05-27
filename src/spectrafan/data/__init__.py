"""Datasets and the factory seam that selects one by name.

A thin factory now (one dataset). It graduates to a Registry when a second
dataset lands — register builders the way models do.
"""

from __future__ import annotations

from spectrafan.config import DataConfig
from spectrafan.data.temimagenet import TEMImageNetDataset
from spectrafan.data.transforms import Transform, eval_transforms, train_transforms

_KNOWN = ["temimagenet"]


def build_dataset(
    data_cfg: DataConfig,
    split: str,
    transforms: Transform | None = None,
    subset_size: int | None = None,
):
    """Construct the dataset named by ``data_cfg.dataset`` for ``split``."""
    if data_cfg.dataset == "temimagenet":
        return TEMImageNetDataset(
            root=data_cfg.root,
            split=split,
            image_size=data_cfg.image_size,
            splits_dir=data_cfg.splits_dir,
            transforms=transforms,
            subset_size=subset_size,
            input_norm=data_cfg.input_norm,
            in_channels=data_cfg.in_channels,
        )
    raise ValueError(f"unknown dataset: {data_cfg.dataset!r} (known: {_KNOWN})")


__all__ = ["TEMImageNetDataset", "build_dataset", "eval_transforms", "train_transforms"]
