"""Tests for spectrafan.configs (shared YAML loader)."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectrafan.configs import apply_overrides, deep_merge, load_raw_config


def test_load_raw_config_merges_extends(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("a: 1\nb:\n  c: 2\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("extends: base.yaml\nb:\n  d: 3\n")

    raw = load_raw_config(overlay)

    assert raw == {"a": 1, "b": {"c": 2, "d": 3}}


def test_load_raw_config_applies_dotted_overrides(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("a:\n  b: 1\n")

    raw = load_raw_config(cfg, overrides=["a.b=2", "a.c=hello"])

    assert raw == {"a": {"b": 2, "c": "hello"}}


def test_load_raw_config_no_extends(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("x: 1\n")

    assert load_raw_config(cfg) == {"x": 1}


def test_deep_merge_overlay_wins_on_scalars() -> None:
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_deep_merge_recurses_on_dicts() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_apply_overrides_rejects_missing_equals() -> None:
    with pytest.raises(ValueError, match="override must be key=value"):
        apply_overrides({}, ["bad"])


def test_apply_overrides_does_not_mutate_input() -> None:
    d = {"a": {"b": 1}}
    original_inner = d["a"]
    result = apply_overrides(d, ["a.c=2"])
    assert "c" not in original_inner, "apply_overrides mutated nested dict in input"
    assert result == {"a": {"b": 1, "c": 2}}
    assert d == {"a": {"b": 1}}, "apply_overrides mutated top-level input"


def test_load_raw_config_only_resolves_one_level_of_extends(tmp_path: Path) -> None:
    """Pin the current single-level behavior. If a base has `extends:`, that
    directive is not followed; it just appears in the merged dict."""
    grandparent = tmp_path / "grandparent.yaml"
    grandparent.write_text("gp_key: 1\n")
    parent = tmp_path / "parent.yaml"
    parent.write_text("extends: grandparent.yaml\nparent_key: 2\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\nchild_key: 3\n")

    raw = load_raw_config(child)

    # parent_key and child_key are merged in. gp_key is NOT — the nested
    # extends inside parent.yaml is passed through as a literal key.
    assert raw == {"extends": "grandparent.yaml", "parent_key": 2, "child_key": 3}
