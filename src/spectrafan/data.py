"""Dataset loaders for TEMImageNet v1.3 (public crystal STEM library).

Two stateless helpers:
- list_pairs: enumerate matched (image, mask) PNG pairs under a dataset root.
- load_pair: load one pair as numpy arrays in canonical dtypes/ranges.

No torch.utils.data.Dataset here yet; training-time wiring lives with the
baseline-reproduction notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from PIL import Image  # noqa: F401 — used in load_pair (Task 3)

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
