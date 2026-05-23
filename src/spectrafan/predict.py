"""Generate inference artifacts for a finished training run.

CLI:
    uv run python -m spectrafan.predict --run runs/<id>
    uv run python -m spectrafan.predict --latest fanetmini

Writes <run>/predictions.npz (16 val + 16 test sample preds) and
<run>/test_metrics.json (IoU/Dice/pixel-acc over the full test split).
"""

from __future__ import annotations

from pathlib import Path


def find_latest_run(runs_root: Path, suffix: str) -> Path:
    """Most-recently-mtime'd `runs_root/*_<suffix>/` directory.

    Raises FileNotFoundError if no matching directory exists.
    """
    candidates = sorted(
        (p for p in runs_root.glob(f"*_{suffix}") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no {runs_root}/*_{suffix}/ dirs found")
    return candidates[-1]
