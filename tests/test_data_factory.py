"""Tests for the dataset factory seam."""

from __future__ import annotations

import pytest

from spectrafan.config import DataConfig
from spectrafan.data import build_dataset


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="temimagenet"):
        build_dataset(DataConfig(dataset="nope"), split="train")
