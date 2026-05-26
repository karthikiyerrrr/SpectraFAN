"""Dataset loaders for TEMImageNet v1.3 (public crystal STEM library).

Three stateless helpers and a torch Dataset:
- list_pairs: enumerate matched (image, mask) PNG pairs under a dataset root.
- load_pair: load one pair as numpy arrays in canonical dtypes/ranges.
- build_split / load_split: deterministic train/val/test splitting by stem.
- TEMImageNetDataset: torch Dataset wrapping the above for training loops.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def list_pairs(root: Path, target: str = "circularMask") -> pl.DataFrame:
    """Enumerate matched (image, mask) PNG pairs under ``root``.

    Pairs are matched by filename stem between ``root/image/`` and
    ``root/{target}/``. Unmatched files on either side are dropped from the
    result and the total orphan count is logged at WARN level.

    Returns a polars DataFrame with columns
    ``stem, image_path, mask_path, image_bytes, mask_bytes`` sorted by stem.
    """
    image_dir = root / "image"
    mask_dir = root / target

    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing image dir: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"missing mask dir: {mask_dir}")

    images = {p.stem: p for p in image_dir.glob("*.png")}
    masks = {p.stem: p for p in mask_dir.glob("*.png")}

    matched = sorted(set(images) & set(masks))
    orphan_count = (len(images) - len(matched)) + (len(masks) - len(matched))
    if orphan_count:
        logger.warning(
            "%d orphan file(s) without a counterpart were dropped from %s",
            orphan_count,
            root,
        )

    rows = [
        {
            "stem": stem,
            "image_path": str(images[stem].resolve()),
            "mask_path": str(masks[stem].resolve()),
            "image_bytes": images[stem].stat().st_size,
            "mask_bytes": masks[stem].stat().st_size,
        }
        for stem in matched
    ]
    return pl.DataFrame(
        rows,
        schema={
            "stem": pl.Utf8,
            "image_path": pl.Utf8,
            "mask_path": pl.Utf8,
            "image_bytes": pl.Int64,
            "mask_bytes": pl.Int64,
        },
    )


def load_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one (image, mask) pair as numpy arrays.

    Returns
    -------
    image : np.ndarray, float32, single-channel, values in [0, 1]
    mask  : np.ndarray, uint8,   single-channel, values in {0, 1}

    Multi-channel inputs are collapsed to luminance. The mask is thresholded at
    >0 so any non-zero encoding (0/255, 0/1, anti-aliased edges) reduces to
    binary.
    """
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("L"))
    with Image.open(mask_path) as im:
        mask_raw = np.asarray(im.convert("L"))

    if image.dtype == np.uint8:
        image_f = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image_f = image.astype(np.float32) / 65535.0
    else:
        image_f = image.astype(np.float32)

    mask = (mask_raw > 0).astype(np.uint8)
    return image_f, mask


def build_split(
    stems: list[str],
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle ``stems`` deterministically and slice into (train, val, test).

    The input is sorted before shuffling so the result depends only on ``stems``
    as a set and ``seed`` -- not on the input order.
    """
    if not abs(sum(fractions) - 1.0) < 1e-9:
        raise ValueError(f"fractions must sum to 1.0; got {fractions} summing to {sum(fractions)}")

    shuffled = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * fractions[0])
    n_val = round(n * fractions[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def load_split(splits_dir: Path, name: str) -> list[str]:
    """Read ``splits_dir/{name}.txt`` and return one stem per non-empty line."""
    path = splits_dir / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing split file: {path}")
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Split = Literal["train", "val", "test"]
InputNorm = Literal["none", "per_image_zscore"]
Transform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# Internal crop/resize helpers
# ---------------------------------------------------------------------------


def _center_crop_or_resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a 2D array to ``size x size``. Resize up (bilinear) if smaller."""
    h, w = arr.shape
    if h >= size and w >= size:
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top : top + size, left : left + size]
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))


def _center_crop_or_resize_mask(arr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a 2D array to ``size x size``. Resize up (nearest) if smaller."""
    h, w = arr.shape
    if h >= size and w >= size:
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top : top + size, left : left + size]
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.NEAREST))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TEMImageNetDataset(Dataset):
    """Torch Dataset over matched (image, mask) pairs in a committed split.

    Returns
    -------
    image : torch.Tensor, float32, shape (3, image_size, image_size), values in [0, 1].
            Grayscale source replicated to 3 channels for the FANet RGB stem.
    mask  : torch.Tensor, float32, shape (1, image_size, image_size), values in {0.0, 1.0}.

    The default ``splits_dir`` is relative to the process CWD — invoke the
    training CLI from the repo root, or pass an absolute path explicitly.
    """

    def __init__(
        self,
        root: Path,
        split: Split,
        image_size: int,
        splits_dir: Path = Path("data/splits/temimagenet_v1"),
        transforms: Transform | None = None,
        subset_size: int | None = None,
        input_norm: InputNorm = "none",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.input_norm = input_norm

        stems = load_split(Path(splits_dir), split)
        if subset_size is not None:
            stems = stems[:subset_size]

        df = list_pairs(self.root)
        index = {row["stem"]: row for row in df.iter_rows(named=True)}
        missing = [s for s in stems if s not in index]
        if missing:
            raise ValueError(
                f"{len(missing)} split stems are missing from {self.root}; first few: {missing[:5]}"
            )
        self._rows = [index[s] for s in stems]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._rows[idx]
        image_np, mask_np = load_pair(Path(row["image_path"]), Path(row["mask_path"]))
        if image_np.shape != mask_np.shape:
            raise ValueError(f"image/mask shape mismatch: {image_np.shape} vs {mask_np.shape}")
        if image_np.shape[0] != image_np.shape[1]:
            raise ValueError(
                f"non-square input {image_np.shape}; TEMImageNetDataset only supports square sources"
            )
        image_np = _center_crop_or_resize(image_np, self.image_size)
        mask_np = _center_crop_or_resize_mask(mask_np, self.image_size)

        if self.input_norm == "per_image_zscore":
            mean = float(image_np.mean())
            std = float(image_np.std())
            image_np = (image_np - mean) / (std + 1e-6)

        # image_np is float32 in [0, 1]; replicate to 3 channels for FANet's RGB stem.
        image = torch.from_numpy(image_np).float().unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        # mask_np is uint8 in {0, 1}; cast to float32 for BCE loss.
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)  # (1, H, W)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask
