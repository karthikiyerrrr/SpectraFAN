"""Tests for the one-time run-config migrator."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from migrate_run_configs import migrate_config_dict  # noqa: E402


def test_injects_skip_transform_default():
    old = {
        "model": {
            "name": "fanetmini",
            "channels": [32, 64, 128],
            "bottleneck": 256,
            "fam_conv_kind": "depthwise",
            "output_norm": "bn",
        },
        "data": {
            "dataset": "temimagenet",
            "root": "data/raw/temimagenet",
            "image_size": 256,
            "batch_size": 16,
            "splits_dir": "data/splits/temimagenet_v1",
            "num_workers": 8,
            "input_norm": "none",
            "in_channels": 3,
        },
        "aug": {"p_flip": 0.5, "max_rot_deg": 15.0, "zoom_range": [0.9, 1.1], "noise_sigma": 0.01},
        "optim": {
            "optimizer": "adamw",
            "lr": 1.0e-3,
            "decay": 0.99,
            "weight_decay": 1.0e-4,
            "momentum": 0.999,
            "betas": [0.9, 0.999],
            "schedule": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1.0e-5,
        },
        "train": {
            "epochs": 200,
            "seed": 0,
            "device": "auto",
            "run_root": "runs",
            "loss": {"ce_weight": 0.5, "dice_weight": 0.5},
            "amp": True,
            "checkpoint_every": 25,
            "deterministic": True,
        },
    }
    new = migrate_config_dict(old)
    assert new["model"]["skip_transform"] == "fam_complex"


def test_migrated_dict_loads_under_new_schema(tmp_path):
    from spectrafan.config import load_config

    old = {
        "model": {"name": "fanetmini"},
        "train": {"loss": {"ce_weight": 0.3, "dice_weight": 0.7}},
    }
    new = migrate_config_dict(old)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(new))
    cfg = load_config(p)
    assert cfg.model.name == "fanetmini"
    assert cfg.model.skip_transform == "fam_complex"
    assert cfg.train.loss.dice_weight == 0.7


def test_migrates_flat_loss_to_nested():
    old = {
        "model": {"name": "fanetmini"},
        "train": {"epochs": 200, "loss_ce_weight": 0.3, "loss_dice_weight": 0.7},
    }
    new = migrate_config_dict(old)
    assert "loss_ce_weight" not in new["train"]
    assert "loss_dice_weight" not in new["train"]
    assert new["train"]["loss"] == {"ce_weight": 0.3, "dice_weight": 0.7}


def test_legacy_flat_loss_config_loads_after_migration(tmp_path):
    from spectrafan.config import load_config

    # Shape produced by the pre-refactor serializer: flat loss keys, no skip_transform.
    old = {
        "model": {
            "name": "fanetmini",
            "channels": [64, 128, 256, 512],
            "bottleneck": 1024,
            "fam_conv_kind": "depthwise",
            "output_norm": "bn",
        },
        "data": {
            "dataset": "temimagenet",
            "root": "data/raw/temimagenet",
            "image_size": 256,
            "batch_size": 16,
            "subset_size": None,
            "val_subset_size": None,
            "splits_dir": "data/splits/temimagenet_v1",
            "num_workers": 8,
            "input_norm": "none",
            "in_channels": 3,
        },
        "aug": {"p_flip": 0.5, "max_rot_deg": 15.0, "zoom_range": [0.9, 1.1], "noise_sigma": 0.01},
        "optim": {
            "optimizer": "adamw",
            "lr": 1.0e-3,
            "decay": 0.99,
            "weight_decay": 1.0e-4,
            "momentum": 0.999,
            "betas": [0.9, 0.999],
            "schedule": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1.0e-5,
        },
        "train": {
            "epochs": 200,
            "seed": 0,
            "device": "auto",
            "run_root": "runs",
            "loss_ce_weight": 0.5,
            "loss_dice_weight": 0.5,
            "amp": True,
            "checkpoint_every": 25,
            "deterministic": True,
        },
    }
    import yaml

    new = migrate_config_dict(old)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(new))
    cfg = load_config(p)  # must NOT raise
    assert cfg.model.name == "fanetmini"
    assert cfg.model.skip_transform == "fam_complex"
    assert cfg.train.loss.ce_weight == 0.5
    assert cfg.train.loss.dice_weight == 0.5
