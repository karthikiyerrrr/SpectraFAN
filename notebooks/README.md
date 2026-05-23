# Notebooks

Exploratory work for SpectraFAN. All version-controlled notebooks are
[marimo](https://marimo.io/) `.py` files — reactive, plain-text diffable.
Colab launcher notebooks (`colab_*.ipynb`) are gitignored because they hold
per-user Drive paths and auth state; the canonical cell content lives in
`03b_full_repro.py` and `04_profile_fam.py` under their "How to reproduce a
run" accordions.

Edit via marimo-pair:

```bash
uv run marimo edit notebooks/<file>.py
```

Conventions:

- Use **polars** for tabular data and run logs (not pandas).
- Use **plotly** for visualization.
- Hide cells that only instantiate UI elements or generate plot objects; surface
  the rendered result.

Current notebooks (reproduction phase):

| # | Notebook | Purpose |
| --- | --- | --- |
| 01 | `01_data.py` | TEMImageNet inventory, paired image/mask viewer, log-magnitude FFT view |
| 02 | `02_model.py` | FANet (FAMComplex + U-Net) build, calibrated to Table 1 |
| 03a | `03a_repro_baseline.py` | Smoke / baseline training-run inspector: curves + 4-sample prediction grid |
| 03b | `03b_full_repro.py` | Full-reproduction inspector for Colab-produced runs, plus held-out test-set evaluation. Contains the Colab launcher recipe under an accordion. |
| 04 | `04_profile_fam.py` | Per-op timing inspector for `spectrafan.profile` runs. Contains the Colab launcher recipe under an accordion. |

Further notebooks will be added once the reproduction is locked and the
research direction is chosen.
