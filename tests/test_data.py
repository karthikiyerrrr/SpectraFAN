"""Tests for spectrafan.data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spectrafan.data import TEMImageNetDataset, build_split, list_pairs, load_pair, load_split


def _write_png(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=128).save(path)


def test_list_pairs_matches_by_stem_and_warns_on_orphans(tmp_path: Path, caplog) -> None:
    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "circularMask"

    for stem in ("00001", "00002", "00003"):
        _write_png(image_dir / f"{stem}.png")
        _write_png(mask_dir / f"{stem}.png")

    # orphans on both sides
    _write_png(image_dir / "00099.png")
    _write_png(mask_dir / "00077.png")

    with caplog.at_level(logging.WARNING):
        df = list_pairs(tmp_path)

    assert df["stem"].to_list() == ["00001", "00002", "00003"]
    assert set(df.columns) == {
        "stem",
        "image_path",
        "mask_path",
        "image_bytes",
        "mask_bytes",
    }
    assert all((df["image_bytes"] > 0).to_list())
    assert all((df["mask_bytes"] > 0).to_list())

    warn_text = " ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "orphan" in warn_text.lower()
    assert "2" in warn_text  # 1 orphan image + 1 orphan mask


def test_load_pair_returns_float_image_and_binary_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    mask_path = tmp_path / "msk.png"
    Image.new("L", (16, 16), color=200).save(image_path)
    # mask drawn with both 0 and 255 pixels so the threshold actually matters
    mask = Image.new("L", (16, 16), color=0)
    for x in range(8):
        for y in range(8):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    image, mask_arr = load_pair(image_path, mask_path)

    assert image.dtype == np.float32
    assert image.shape == (16, 16)
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert np.isclose(image.mean(), 200 / 255, atol=1e-3)

    assert mask_arr.dtype == np.uint8
    assert mask_arr.shape == (16, 16)
    assert set(np.unique(mask_arr).tolist()) <= {0, 1}
    assert mask_arr.sum() == 64  # the 8x8 white quadrant


def test_build_split_partitions_and_is_deterministic() -> None:
    stems = [f"{i:05d}" for i in range(100)]
    train1, val1, test1 = build_split(stems, seed=0)
    train2, val2, test2 = build_split(stems, seed=0)

    assert (len(train1), len(val1), len(test1)) == (80, 10, 10)
    assert set(train1) | set(val1) | set(test1) == set(stems)
    assert not (set(train1) & set(val1))
    assert not (set(train1) & set(test1))
    assert not (set(val1) & set(test1))

    # Determinism across re-runs.
    assert (train1, val1, test1) == (train2, val2, test2)

    # Insensitivity to input order.
    train3, val3, test3 = build_split(list(reversed(stems)), seed=0)
    assert (train1, val1, test1) == (train3, val3, test3)


def test_load_split_reads_stems(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "train.txt").write_text("00001\n00002\n00003\n")
    assert load_split(splits_dir, "train") == ["00001", "00002", "00003"]


def test_dataset_returns_correct_shapes(tmp_path: Path) -> None:
    """Dataset returns (3, S, S) float image and (1, S, S) {0,1}-valued float mask."""
    # Fixture: dataset root with image/ and circularMask/ subdirs and a split file.
    root = tmp_path / "dataset"
    (root / "image").mkdir(parents=True)
    (root / "circularMask").mkdir(parents=True)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    stems = [f"{i:05d}" for i in range(5)]
    for stem in stems:
        Image.new("L", (64, 64), color=128).save(root / "image" / f"{stem}.png")
        # Mask with a non-trivial pattern so binarization is exercised.
        mask = Image.new("L", (64, 64), color=0)
        for x in range(32):
            for y in range(32):
                mask.putpixel((x, y), 255)
        mask.save(root / "circularMask" / f"{stem}.png")
    (splits_dir / "train.txt").write_text("\n".join(stems) + "\n")

    ds = TEMImageNetDataset(
        root=root,
        split="train",
        image_size=32,  # center-crop from 64
        splits_dir=splits_dir,
    )

    assert len(ds) == 5
    image, mask = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min().item() and image.max().item() <= 1.0
    assert mask.shape == (1, 32, 32)
    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_dataset_subset_size_truncates(tmp_path: Path) -> None:
    """``subset_size`` truncates the stem list to the first N entries."""
    root = tmp_path / "dataset"
    (root / "image").mkdir(parents=True)
    (root / "circularMask").mkdir(parents=True)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    stems = [f"{i:05d}" for i in range(10)]
    for stem in stems:
        Image.new("L", (32, 32), color=128).save(root / "image" / f"{stem}.png")
        Image.new("L", (32, 32), color=255).save(root / "circularMask" / f"{stem}.png")
    (splits_dir / "train.txt").write_text("\n".join(stems) + "\n")

    ds = TEMImageNetDataset(
        root=root,
        split="train",
        image_size=32,
        splits_dir=splits_dir,
        subset_size=3,
    )
    assert len(ds) == 3


def test_temimagenet_dataset_per_image_zscore_normalizes(tmp_path: Path) -> None:
    """input_norm='per_image_zscore' yields zero-mean, unit-variance per image."""
    from spectrafan.data import TEMImageNetDataset

    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "circularMask"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    # write a non-uniform grayscale image so std > 0
    for stem in ("00001",):
        path = image_dir / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.tile(np.linspace(0, 255, 16, dtype=np.uint8), (16, 1))
        Image.fromarray(arr).save(path)
        _write_png(mask_dir / f"{stem}.png", size=(16, 16))
    (splits_dir / "train.txt").write_text("00001\n")

    ds = TEMImageNetDataset(
        root=tmp_path,
        split="train",
        image_size=16,
        splits_dir=splits_dir,
        input_norm="per_image_zscore",
    )
    img, _ = ds[0]
    assert torch.isfinite(img).all()
    # per-image z-score: each channel of the replicated tensor is the same image, so
    # the per-image stats are identical across channels. Check overall.
    assert abs(img.mean().item()) < 1e-5
    assert abs(img.std().item() - 1.0) < 1e-3


def test_temimagenet_dataset_input_norm_default_is_none(tmp_path: Path) -> None:
    """Default input_norm preserves baseline behavior: image stays in [0, 1]."""
    from spectrafan.data import TEMImageNetDataset

    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "circularMask"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _write_png(image_dir / "00001.png")
    _write_png(mask_dir / "00001.png")
    (splits_dir / "train.txt").write_text("00001\n")

    ds = TEMImageNetDataset(
        root=tmp_path,
        split="train",
        image_size=8,
        splits_dir=splits_dir,
    )
    img, _ = ds[0]
    assert 0.0 <= img.min().item() and img.max().item() <= 1.0


def test_temimagenet_dataset_in_channels_one_returns_single_channel(tmp_path: Path) -> None:
    """in_channels=1 returns a (1, H, W) image without replicating to 3 channels."""
    from spectrafan.data import TEMImageNetDataset

    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "circularMask"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _write_png(image_dir / "00001.png")
    _write_png(mask_dir / "00001.png")
    (splits_dir / "train.txt").write_text("00001\n")

    ds = TEMImageNetDataset(
        root=tmp_path,
        split="train",
        image_size=8,
        splits_dir=splits_dir,
        in_channels=1,
    )
    img, _ = ds[0]
    assert img.shape == (1, 8, 8)


def test_temimagenet_dataset_in_channels_default_three(tmp_path: Path) -> None:
    """Default in_channels=3 preserves baseline behavior (replication to 3 channels)."""
    from spectrafan.data import TEMImageNetDataset

    image_dir = tmp_path / "image"
    mask_dir = tmp_path / "circularMask"
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    _write_png(image_dir / "00001.png")
    _write_png(mask_dir / "00001.png")
    (splits_dir / "train.txt").write_text("00001\n")

    ds = TEMImageNetDataset(
        root=tmp_path,
        split="train",
        image_size=8,
        splits_dir=splits_dir,
    )
    img, _ = ds[0]
    assert img.shape == (3, 8, 8)
