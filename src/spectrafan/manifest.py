"""Per-run output manifest (``run.json``).

A single manifest per run directory records which pipeline stages have written to
it, the headline scalar metrics, and the artifact files present. The train,
predict, profile, and diagnose tools each call :func:`write_manifest` after
emitting their artifacts; calls are additive, so a run dir that is trained then
predicted then diagnosed accumulates all three stages in one file.

The viewer notebooks call :func:`load_manifest` to decide which sections to
render. For runs produced before manifests existed, :func:`load_manifest`
synthesizes an equivalent view from the artifact files on disk so legacy runs
still render.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST_NAME = "run.json"

# Canonical stage order (also the only valid stage names).
STAGE_ORDER = ["train", "predict", "diagnose", "profile"]

# Artifact files whose presence implies a stage ran (legacy inference only).
_STAGE_ARTIFACTS = {
    "train": ["metrics.parquet"],
    "predict": ["test_metrics.json", "predictions.npz"],
    "diagnose": ["fam_diagnosis.json", "fam_stats.parquet"],
    "profile": ["timings.parquet", "summary.json"],
}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _list_artifacts(run_dir: Path) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir() if p.is_file() and p.name != MANIFEST_NAME)


def _order_stages(stages) -> list[str]:
    seen = set(stages)
    return [s for s in STAGE_ORDER if s in seen]


def write_manifest(
    run_dir: Path,
    stage: str,
    *,
    model: str | None = None,
    dataset: str | None = None,
    headline: dict | None = None,
) -> dict:
    """Create or update ``run.json`` in ``run_dir`` for a completed ``stage``.

    Unions ``stage`` into ``stages`` (canonical order, idempotent), merges
    ``headline`` scalars, refreshes ``artifacts`` from disk, and stamps
    ``updated`` (and ``created`` on first write). Returns the written manifest.
    """
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage: {stage!r} (valid: {STAGE_ORDER})")
    run_dir = Path(run_dir)
    path = run_dir / MANIFEST_NAME

    if path.is_file():
        manifest = json.loads(path.read_text())
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "model": None,
            "dataset": None,
            "stages": [],
            "headline": {},
            "artifacts": [],
            "created": _now(),
            "updated": None,
        }

    manifest["stages"] = _order_stages([*manifest["stages"], stage])
    if model is not None:
        manifest["model"] = model
    if dataset is not None:
        manifest["dataset"] = dataset
    if headline:
        manifest["headline"] = {**manifest["headline"], **headline}
    manifest["artifacts"] = _list_artifacts(run_dir)
    manifest["updated"] = _now()

    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def infer_stages(run_dir: Path) -> list[str]:
    """Stages implied by artifact presence (for runs lacking a manifest)."""
    run_dir = Path(run_dir)
    present = {p.name for p in run_dir.iterdir() if p.is_file()}
    return _order_stages(
        stage for stage, files in _STAGE_ARTIFACTS.items() if any(f in present for f in files)
    )


def load_manifest(run_dir: Path) -> dict:
    """Return the run's manifest, synthesizing a minimal one from artifacts when
    ``run.json`` is absent (legacy runs). The synthesized manifest has an empty
    headline and null model/dataset."""
    run_dir = Path(run_dir)
    path = run_dir / MANIFEST_NAME
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "model": None,
        "dataset": None,
        "stages": infer_stages(run_dir),
        "headline": {},
        "artifacts": _list_artifacts(run_dir),
        "created": None,
        "updated": None,
    }
