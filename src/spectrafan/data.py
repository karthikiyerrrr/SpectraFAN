"""Dataset loaders for TEMImageNet v1.3 (public crystal STEM library).

Two stateless helpers:
- list_pairs: enumerate matched (image, mask) PNG pairs under a dataset root.
- load_pair: load one pair as numpy arrays in canonical dtypes/ranges.

No torch.utils.data.Dataset here yet; training-time wiring lives with the
baseline-reproduction notebook.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

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

    ordered = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    return train, val, test


def load_split(splits_dir: Path, name: str) -> list[str]:
    """Read ``splits_dir/{name}.txt`` and return one stem per non-empty line."""
    path = splits_dir / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing split file: {path}")
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]
