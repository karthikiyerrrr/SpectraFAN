"""Post-hoc FAM diagnostic.

Loads <run_dir>/best.pt, runs the val split three times (as_trained,
fam_skip_fft, fam_zero), and writes <run_dir>/fam_stats.parquet plus
<run_dir>/fam_diagnosis.json. See docs/superpowers/specs/2026-05-25-fam-postmortem-design.md.

CLI:
    uv run python -m spectrafan.diagnose_fam --run runs/<id>
    uv run python -m spectrafan.diagnose_fam --latest fanetmini
"""

from __future__ import annotations

from pathlib import Path


def diagnose_run(run_dir: Path) -> None:
    """Placeholder; filled in by later tasks."""
    raise NotImplementedError
