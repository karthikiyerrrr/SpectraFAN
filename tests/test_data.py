"""Tests for spectrafan.data."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from spectrafan.data import list_pairs


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
