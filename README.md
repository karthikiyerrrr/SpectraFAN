# SpectraFAN

Frequency-domain attention for crystallographic image segmentation.

A two-phase research project on top of FANet (Liu et al., *Materials Today Nano* 33, 2026):

1. **Phase 1 — latency foundation.** Make the Frequency Attention Module materially
   faster at inference and in training with no accuracy loss, via a real-input FFT,
   kernel fusion + graph capture, and a real-valued (DCT-domain) variant.
2. **Phase 2 — crystallographic frequency prior.** Inject what lattice symmetry tells
   us about *where* the signal lives in frequency space, and test the resulting
   gains in data efficiency and sim-to-real transfer.

The phases are deliberately decoupled — Phase 1 is a measurable, shippable result on
its own; Phase 2 is a research bet that lands *on top of* an already-improved model.

## Tech stack

- **Python 3.11**, managed with **[uv](https://docs.astral.sh/uv/)**
- **PyTorch** for models, profiling, and `torch.compile` / CUDA graphs
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

Pull the public crystal STEM dataset (TEMImageNet / AtomSegNet, Lin et al. 2021):

```bash
uv run python data/download.py
```

Open the first notebook:

```bash
uv run marimo edit notebooks/01_repro_baseline.py
```

## Project layout

```text
SpectraFAN/
├── src/spectrafan/      # the package
│   ├── fam.py           # FAM variants: complex | real (rfft) | dct
│   ├── unet.py          # U-Net backbone with FAM on skip connections
│   ├── priors.py        # Phase 2 frequency priors
│   ├── data.py          # dataset loaders
│   ├── train.py         # training loop, metrics, losses
│   ├── profile.py       # torch.profiler harness
│   └── compile.py       # torch.compile / CUDA-graph wrappers
├── notebooks/           # marimo, in plan order
├── tests/               # numerical equivalence + regression tests
├── configs/             # one YAML per run — swap --fam and --prior cleanly
├── data/                # download script + raw/processed (gitignored)
└── pyproject.toml
```

The `src/` layout means `spectrafan` must be installed (which `uv sync` handles) —
you can't `import spectrafan` from the repo root. Intentional: catches packaging
issues early, matches modern Python practice.

## Common commands

```bash
# Environment
uv sync                                # install / refresh dependencies
uv add <package>                       # add a runtime dependency
uv add --dev <package>                 # add a dev dependency

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
- FcaNet — fixed DCT bases for channel attention (contrast point for the real-valued FAM variant).

## License

MIT.
