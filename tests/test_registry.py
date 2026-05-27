"""Tests for the generic Registry."""

from __future__ import annotations

import pytest

from spectrafan.registry import Registry


def test_register_and_build():
    reg = Registry("widget")

    @reg.register("a")
    def _a(x):
        return x + 1

    assert reg.build("a", 1) == 2
    assert reg.keys() == ["a"]


def test_duplicate_key_raises():
    reg = Registry("widget")

    @reg.register("a")
    def _a():
        return 1

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("a")
        def _a2():
            return 2


def test_unknown_key_lists_available():
    reg = Registry("widget")

    @reg.register("known")
    def _k():
        return 1

    with pytest.raises(ValueError, match="known"):
        reg.build("missing")
