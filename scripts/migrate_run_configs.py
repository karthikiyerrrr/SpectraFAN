"""One-time migrator for run config.yaml files into the current schema.

Reads a finished run's config.yaml, fills in fields the current schema expects
(notably model.skip_transform), validates by round-tripping through
spectrafan.config.load_config, backs up the original to config.yaml.bak, and
rewrites config.yaml in the canonical schema shape.

    uv run python scripts/migrate_run_configs.py runs/<id> [runs/<id2> ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def migrate_config_dict(old: dict) -> dict:
    """Return a copy of ``old`` upgraded to the current schema defaults.

    Injects ``model.skip_transform`` and migrates the legacy flat
    ``train.loss_ce_weight`` / ``train.loss_dice_weight`` keys into the current
    nested ``train.loss.{ce_weight,dice_weight}`` form.
    """
    new = {k: dict(v) if isinstance(v, dict) else v for k, v in old.items()}

    model = dict(new.get("model") or {})
    model.setdefault("skip_transform", "fam_complex")
    new["model"] = model

    train = dict(new.get("train") or {})
    if "loss_ce_weight" in train or "loss_dice_weight" in train:
        loss = dict(train.get("loss") or {})
        if "loss_ce_weight" in train:
            loss.setdefault("ce_weight", train.pop("loss_ce_weight"))
        if "loss_dice_weight" in train:
            loss.setdefault("dice_weight", train.pop("loss_dice_weight"))
        train["loss"] = loss
        new["train"] = train

    return new


def migrate_run(run_dir: Path) -> None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"no config.yaml in {run_dir}")
    old = yaml.safe_load(cfg_path.read_text()) or {}
    new = migrate_config_dict(old)

    # Validate before writing: must load cleanly under the new schema.
    from spectrafan.config import load_config

    tmp = run_dir / "config.migrated.tmp.yaml"
    tmp.write_text(yaml.safe_dump(new))
    try:
        load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    backup = run_dir / "config.yaml.bak"
    if not backup.exists():
        backup.write_text(cfg_path.read_text())
    cfg_path.write_text(yaml.safe_dump(new))
    print(f"migrated: {cfg_path} (backup at {backup})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="migrate_run_configs")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for rd in args.run_dirs:
        migrate_run(rd)


if __name__ == "__main__":
    sys.exit(main())
