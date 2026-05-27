"""Tests for spectrafan.config (OmegaConf structured loader)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from spectrafan.config import RunConfig, load_config, run_config_to_dict


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def test_defaults_match_schema(tmp_path):
    cfg_path = _write(tmp_path, "empty.yaml", "{}\n")
    cfg = load_config(cfg_path)
    assert isinstance(cfg, RunConfig)
    assert cfg.model.name == "fanet"
    assert cfg.model.skip_transform == "fam_complex"
    assert cfg.train.loss.ce_weight == 0.5


def test_single_extends_overrides_base(tmp_path):
    _write(tmp_path, "base.yaml", "model:\n  name: fanet\ndata:\n  batch_size: 4\n")
    child = _write(tmp_path, "child.yaml", "extends: base.yaml\ndata:\n  batch_size: 16\n")
    cfg = load_config(child)
    assert cfg.model.name == "fanet"
    assert cfg.data.batch_size == 16


def test_multi_level_extends(tmp_path):
    _write(tmp_path, "a.yaml", "data:\n  image_size: 256\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\nmodel:\n  name: fanetmini\n")
    c = _write(tmp_path, "c.yaml", "extends: b.yaml\ndata:\n  batch_size: 16\n")
    cfg = load_config(c)
    assert cfg.data.image_size == 256  # from a (two levels up)
    assert cfg.model.name == "fanetmini"  # from b
    assert cfg.data.batch_size == 16  # from c


def test_dotted_overrides(tmp_path):
    cfg_path = _write(tmp_path, "x.yaml", "{}\n")
    cfg = load_config(cfg_path, overrides=["train.epochs=5", "optim.lr=1e-3"])
    assert cfg.train.epochs == 5
    assert cfg.optim.lr == pytest.approx(1e-3)


def test_unknown_key_rejected(tmp_path):
    cfg_path = _write(tmp_path, "bad.yaml", "model:\n  nonexistent: 1\n")
    with pytest.raises(ConfigKeyError):
        load_config(cfg_path)


def test_round_trip_dict_is_yaml_friendly(tmp_path):
    cfg = load_config(_write(tmp_path, "x.yaml", "{}\n"))
    d = run_config_to_dict(cfg)
    assert isinstance(d["model"]["channels"], list)
    assert isinstance(d["data"]["root"], str)
    assert d["train"]["loss"]["ce_weight"] == 0.5


def test_load_profile_config_keeps_only_model_and_profile(tmp_path):
    from spectrafan.config import ProfileConfig, load_profile_config

    _write(
        tmp_path,
        "base.yaml",
        """
        model:
          name: fanet
          channels: [64, 128, 256, 512]
        data:
          batch_size: 8
        train:
          epochs: 7
        """,
    )
    prof = _write(
        tmp_path,
        "prof.yaml",
        """
        extends: base.yaml
        profile:
          warmup_iters: 5
          measure_iters: 50
        """,
    )
    cfg = load_profile_config(prof)
    assert isinstance(cfg, ProfileConfig)
    assert cfg.model.name == "fanet"  # inherited model survives
    assert cfg.profile.warmup_iters == 5
    assert cfg.profile.measure_iters == 50
    assert not hasattr(cfg, "data")  # unrelated inherited keys tolerated + dropped


def test_load_profile_config_parses_sweep_entries(tmp_path):
    from spectrafan.config import ProfileSweepEntry, load_profile_config

    prof = _write(
        tmp_path,
        "prof.yaml",
        """
        profile:
          configs:
            - image_size: 256
              batch_size: 4
            - image_size: 512
              batch_size: 2
        """,
    )
    cfg = load_profile_config(prof)
    assert len(cfg.profile.configs) == 2
    assert all(isinstance(e, ProfileSweepEntry) for e in cfg.profile.configs)
    assert cfg.profile.configs[0].image_size == 256
    assert cfg.profile.configs[1].batch_size == 2
