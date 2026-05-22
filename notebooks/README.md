# Notebooks

Exploratory work for SpectraFAN. All notebooks are [marimo](https://marimo.io/) `.py`
files — reactive, version-controllable, no `.ipynb`.

Edit via marimo-pair:

```bash
uv run marimo edit notebooks/<file>.py
```

Conventions:

- Use **polars** for tabular data and run logs (not pandas).
- Use **plotly** for visualization.
- Hide cells that only instantiate UI elements or generate plot objects; surface
  the rendered result.

Planned notebook order tracks the project plan:

| # | Notebook | Purpose |
| --- | --- | --- |
| 01 | `01_data.py` | TEMImageNet inventory, paired image/mask viewer, and log-magnitude FFT view |
| 02 | `02_model.py` | FANet (FAMComplex + U-Net) recreation, calibrated to Table 1 |
| 03a | `03a_repro_baseline.py` | Inspect a smoke / reproduction training run: curves + 4-sample prediction grid |
| colab | `colab_full_repro.ipynb` | Colab launcher: clones repo, mounts Drive, runs the 50-epoch full reproduction training |
| 03b | `03b_full_repro.py` | Local inspector for the full-reproduction run produced by `colab_full_repro.ipynb`, plus held-out test-set evaluation |
| 04 | `04_profile_fam.py` | Per-op profiling — settle transform-bound vs. FLOP-bound |
| 05 | `05_rfft_fam.py` | Step A: rfft FAM + boundary-handling numerical equivalence |
| 06 | `06_fused_fam.py` | Step B: torch.compile + CUDA-graph latency deltas |
| 07 | `07_dct_fam.py` | Step C: real-valued DCT FAM, accuracy vs. baseline |
| 08 | `08_frequency_prior.py` | Phase 2: data-efficiency sweeps with the crystallographic prior |
