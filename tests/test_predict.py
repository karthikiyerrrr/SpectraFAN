"""Tests for spectrafan.predict."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def test_find_latest_run_resolves_most_recent_match(tmp_path: Path) -> None:
    """Given multiple runs/*_<suffix>/ dirs, find_latest_run returns the one
    with the highest mtime that matches the suffix."""
    from spectrafan.predict import find_latest_run

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
    from spectrafan.predict import find_latest_run

    (tmp_path / "2026-05-20_010101_full_repro").mkdir()

    with pytest.raises(FileNotFoundError, match="no .*_fanetmini/ dirs found"):
        find_latest_run(tmp_path, "fanetmini")
