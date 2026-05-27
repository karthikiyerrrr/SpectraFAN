# SpectraFAN

A research project building on FANet (Liu et al., *Materials Today Nano* 33, 2026),
applied to the public TEMImageNet / AtomSegNet crystal-STEM dataset
(Lin et al. 2021).

## Status

We are reproducing FANet, adapted to TEMImageNet (256×256 STEM images).
The dataset is the one load-bearing commitment of this project — the FANet
recipe is being recreated as faithfully as the dataset allows, with
modifications only where TEMImageNet's resolution, channel count, or labels
require them. The reproduction matches the training recipe and is closing
the remaining accuracy gap.

Beyond reproduction, the research will be directed towards **optimization,
efficiency, and/or speed**. The exact approach is deliberately undecided
and will be explored after the reproduction is locked. The repo is
intentionally neutral about which specific direction within that scope
comes next.

## Tech stack

- **Python 3.11**, managed with **[uv](https://docs.astral.sh/uv/)**
- **PyTorch** for models, training, and measurement
- **polars** for dataset metadata and run logs (not pandas)
- **plotly** for visualization
- **[marimo](https://marimo.io/)** for notebooks — reactive, stored as plain `.py`

## Getting started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/<you>/SpectraFAN.git
cd SpectraFAN
uv sync
```

`uv sync` reads `.python-version` (pinned to 3.11), provisions a matching interpreter,
creates a virtual environment, and installs the package in editable mode along with
all dependencies.

Pull the TEMImageNet / AtomSegNet dataset:

```bash
uv run python data/download.py
```

Explore a run (once one exists under `runs/`):

```bash
uv run marimo edit notebooks/explore_run.py
```

## Project layout

```text
SpectraFAN/
├── src/spectrafan/      # the package
│   ├── models/          # FANet, FANetMini, FAM, blocks
│   ├── data/            # TEMImageNet dataset loader and transforms
│   ├── training/        # training loop, losses, metrics
│   └── analysis/        # predict, profile, diagnose_fam
├── scripts/             # build_split.py (data preparation)
├── notebooks/           # marimo
├── tests/               # numerical + regression tests
├── configs/             # one YAML per run
├── data/                # download script + raw/processed (gitignored)
└── pyproject.toml
```

The `src/` layout means `spectrafan` must be installed (which `uv sync` handles) —
you can't `import spectrafan` from the repo root.

## Common commands

```bash
# Environment
uv sync                                # install / refresh dependencies
uv add <package>                       # add a runtime dependency
uv add --dev <package>                 # add a dev dependency

# Training and analysis tools
uv run spectrafan-train --config configs/smoke.yaml
uv run spectrafan-predict --run runs/<id>
uv run spectrafan-profile --config configs/smoke.yaml
uv run spectrafan-diagnose --run runs/<id>

# Notebooks
uv run marimo edit notebooks/<file>    # edit a notebook
uv run marimo run notebooks/<file>     # run a notebook as a read-only app

# Testing and linting
uv run pytest                          # run the test suite
uv run ruff check .                    # lint
uv run ruff format .                   # format
```

## References

- Liu et al., *FANet: A frequency-domain attention network for ...*, Materials Today Nano 33 (2026).
- Lin et al., *Sci. Rep.* (2021) — TEMImageNet / AtomSegNet crystal STEM library.

## License

MIT.
