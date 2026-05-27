"""Backward-compatibility shim: load_raw_config for spectrafan.profile.

The config schema and loader moved to spectrafan.config (Task 1.2).
This module remains to avoid touching profile.py, which calls load_raw_config
to get a plain dict for its own ProfileConfig construction.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    return yaml.safe_load(value)


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    out = copy.deepcopy(d)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value; got {override!r}")
        key, _, value = override.partition("=")
        _set_dotted(out, key.strip(), _coerce(value.strip()))
    return out


def load_raw_config(path: Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML file, resolve one level of `extends:`, apply overrides."""
    raw = yaml.safe_load(path.read_text()) or {}
    if "extends" in raw:
        base_path = path.parent / raw.pop("extends")
        base = yaml.safe_load(base_path.read_text()) or {}
        raw = deep_merge(base, raw)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return raw
