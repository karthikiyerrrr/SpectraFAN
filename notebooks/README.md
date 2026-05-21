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
|---|----------|---------|
| 01 | `repro_baseline.py` | Reproduce FANet Table 1 on the public crystal dataset |
| 02 | `profile_fam.py`    | Per-op profiling — settle transform-bound vs. FLOP-bound |
| 03 | `rfft_fam.py`       | Step A: rfft FAM + boundary-handling numerical equivalence |
| 04 | `fused_fam.py`      | Step B: torch.compile + CUDA-graph latency deltas |
| 05 | `dct_fam.py`        | Step C: real-valued DCT FAM, accuracy vs. baseline |
| 06 | `frequency_prior.py`| Phase 2: data-efficiency sweeps with the crystallographic prior |
