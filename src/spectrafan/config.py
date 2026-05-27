"""Structured config schema and OmegaConf-backed loader.

The dataclasses here are the single source of truth for run configuration.
OmegaConf treats them as a typed schema: it merges the schema with one or more
YAML layers (resolved through a multi-level ``extends:`` chain) plus
``key.subkey=value`` dotted overrides, validates, and returns a populated
dataclass instance.

This module intentionally omits ``from __future__ import annotations``:
OmegaConf introspects real field types at runtime, and stringized (PEP 563)
annotations break that. Every field uses an OmegaConf-supported type — str,
int, float, bool, Optional[...], list, or a nested dataclass. Paths are stored
as str and wrapped in Path() at use sites; the former Literal fields are plain
str validated by the registries/builders that consume them.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    name: str = "fanet"
    channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck: int = 1024
    fam_conv_kind: str = "depthwise"
    skip_transform: str = "fam_complex"
    output_norm: str = "bn"


@dataclass
class DataConfig:
    dataset: str = "temimagenet"
    root: str = "data/raw/temimagenet"
    image_size: int = 512
    batch_size: int = 4
    subset_size: int | None = None
    val_subset_size: int | None = None
    splits_dir: str = "data/splits/temimagenet_v1"
    num_workers: int = 2
    input_norm: str = "none"
    in_channels: int = 3


@dataclass
class AugConfig:
    p_flip: float = 0.5
    max_rot_deg: float = 15.0
    zoom_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    noise_sigma: float = 0.01


@dataclass
class OptimConfig:
    optimizer: str = "rmsprop"
    lr: float = 1.0e-5
    decay: float = 0.99
    weight_decay: float = 1.0e-8
    momentum: float = 0.999
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    schedule: str = "exponential"
    warmup_epochs: int = 0
    min_lr: float = 0.0


@dataclass
class LossConfig:
    ce_weight: float = 0.5
    dice_weight: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 200
    seed: int = 0
    device: str = "auto"
    run_root: str = "runs"
    loss: LossConfig = field(default_factory=LossConfig)
    amp: bool = False
    checkpoint_every: int = 10
    deterministic: bool = True


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --- Profile-tool schema (loaded by spectrafan.analysis.profile) --------------


@dataclass
class ProfileSweepEntry:
    image_size: int = 256
    batch_size: int = 4


@dataclass
class ProfileSection:
    warmup_iters: int = 20
    measure_iters: int = 100
    include_chrome_trace: bool = True
    # typed list-of-dataclass requires omegaconf>=2.2 (pinned >=2.3 in pyproject)
    configs: list[ProfileSweepEntry] = field(default_factory=list)


@dataclass
class ProfileConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    profile: ProfileSection = field(default_factory=ProfileSection)


# --- Loader -------------------------------------------------------------------


def _resolve_extends_chain(path: Path) -> list[Path]:
    """Return YAML paths base-first: root ancestor down to ``path``.

    Each file may declare ``extends: <sibling-relative-path>`` naming its
    parent. The chain is followed to its root and returned reversed so the
    caller merges in override order (children win). Cycles raise.
    """
    chain: list[Path] = []
    seen: set[Path] = set()
    cur = Path(path).resolve()
    while True:
        if cur in seen:
            raise ValueError(f"cyclic extends chain at {cur}")
        seen.add(cur)
        chain.append(cur)
        raw = yaml.safe_load(cur.read_text()) or {}
        parent = raw.get("extends")
        if not parent:
            break
        cur = (cur.parent / parent).resolve()
    return list(reversed(chain))


def _load_layer(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    raw.pop("extends", None)
    return raw


def load_config(path: Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a run config: schema + multi-level extends chain + dotted overrides."""
    layers = [OmegaConf.create(_load_layer(p)) for p in _resolve_extends_chain(Path(path))]
    merged = OmegaConf.merge(OmegaConf.structured(RunConfig), *layers)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(merged)


def load_profile_config(path: Path, overrides: list[str] | None = None) -> ProfileConfig:
    """Load a profile config: tolerates unrelated keys (data/aug/...) inherited
    from a base like default.yaml, keeping only model + profile."""
    layers = [OmegaConf.create(_load_layer(p)) for p in _resolve_extends_chain(Path(path))]
    merged = OmegaConf.merge(*layers)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    subset = OmegaConf.create(
        {"model": merged.get("model") or {}, "profile": merged.get("profile") or {}}
    )
    typed = OmegaConf.merge(OmegaConf.structured(ProfileConfig), subset)
    return OmegaConf.to_object(typed)


def run_config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    """Plain, YAML-safe dict (lists/str/nested dicts) for serialization.

    Valid only for instances produced by the schema dataclasses (RunConfig and
    its nested dataclasses); passing an arbitrary dict or OmegaConf node will
    raise a structured-config error.
    """
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
