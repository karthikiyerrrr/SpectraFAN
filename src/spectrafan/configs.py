"""Shared YAML config loader with `extends:` support and dotted overrides.

Used by both spectrafan.train and spectrafan.profile so the two CLIs share
exactly the same merge + override semantics. The dict shape returned here is
domain-agnostic; each caller projects it onto its own dataclass(es).
"""

from __future__ import annotations

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
    """Parse override values as YAML scalars (bool/int/float/str/list)."""
    return yaml.safe_load(value)


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    out = dict(d)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value; got {override!r}")
        key, _, value = override.partition("=")
        _set_dotted(out, key.strip(), _coerce(value.strip()))
    return out


def load_raw_config(path: Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML file, resolve a single `extends:` parent, apply overrides."""
    raw = yaml.safe_load(path.read_text()) or {}
    if "extends" in raw:
        base_path = path.parent / raw.pop("extends")
        base = yaml.safe_load(base_path.read_text()) or {}
        raw = deep_merge(base, raw)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return raw
