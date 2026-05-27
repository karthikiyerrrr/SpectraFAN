# Notebooks

Live notebooks for SpectraFAN. Two [marimo](https://marimo.io/) viewers explore runs that
sync back from Colab/Drive, and one committed Colab launcher produces those runs.

## Marimo viewers (artifact readers — no torch, no dataset)

| Notebook | Purpose |
| --- | --- |
| `explore_run.py` | Inspect one run: training curves, held-out test, FAM diagnosis, profiling — whichever the run's `run.json` says it produced. |
| `compare_runs.py` | Overlay curves and a side-by-side summary across multiple runs. |

Both read `run.json` (see `spectrafan.manifest`) and degrade gracefully for older runs that
predate manifests. Point them at a Drive-synced runs directory:

```bash
export SPECTRAFAN_RUNS_DIR="/path/to/your/Drive/.../SpectraFAN/runs"
uv run marimo edit notebooks/explore_run.py
```

Conventions: **polars** for tables, **plotly** for figures, and hide cells that only build
UI/plot objects — surface the rendered result.

## Colab launcher

`colab_spectrafan.ipynb` (committed boilerplate) mounts Drive, clones, syncs the pinned env,
downloads the dataset, symlinks `runs/` to Drive, then offers labeled task cells: **Train**,
**Predict (emit artifacts)**, **Profile**. Copy it to a personal `colab_*.ipynb` (gitignored)
and set `DRIVE_ROOT` in cell 3 to your Drive layout. Training emits a run dir under
`$DRIVE_ROOT/runs/`; run **Predict** after **Train** so the viewers have prediction artifacts.

## Archive

`archive/` holds the reproduction-phase notebooks (`01`–`07`), kept for history.
