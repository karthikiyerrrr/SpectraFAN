import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 04 — FAM per-op profiling inspector

        Inspect a `configs/profile.yaml` profiling run produced on Colab. Renders the
        FAM share of forward time, a stacked bar of forward time by category, and a
        per-skip breakdown table.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "How to reproduce a run (Colab L4/A100)": mo.md(
                r"""
                Create a Jupyter notebook in any Google Drive folder (e.g.
                `My Drive/SpectraFAN/colab_profile.ipynb` — `colab_*.ipynb` is
                gitignored). Paste these eight cells, edit `DRIVE_ROOT` in cell 3
                to match your Drive layout, pick an L4 or A100 runtime, and run all.
                Artifacts land in `<DRIVE_ROOT>/runs/<timestamp>_profile/`.

                **Cell 1 (markdown):** title / one-line description.

                **Cell 2 (code):** mount Drive.
                ```python
                from google.colab import drive
                drive.mount('/content/drive')
                ```

                **Cell 3 (code):** anchor Drive root.
                ```python
                import os
                DRIVE_ROOT = '/content/drive/MyDrive/SpectraFAN'  # edit to your layout
                os.environ['DRIVE_ROOT'] = DRIVE_ROOT
                assert os.path.isdir(DRIVE_ROOT), f'expected Drive folder not found: {DRIVE_ROOT}'
                ```

                **Cell 4 (code):** clone repo into Colab local disk (idempotent).
                ```python
                !test -d /content/SpectraFAN || git clone https://github.com/karthikiyerrrr/SpectraFAN.git /content/SpectraFAN
                ```

                **Cell 5 (code):** install pinned env from `uv.lock`.
                ```python
                !cd /content/SpectraFAN && pip install uv -q && uv python install 3.11 && uv sync --frozen --no-dev
                ```

                **Cell 6 (code):** download TEMImageNet into `data/raw/` (~2 GB, idempotent).
                ```python
                !cd /content/SpectraFAN && uv run python data/download.py
                ```

                **Cell 7 (code):** symlink `runs/` to Drive so artifacts persist + sync.
                ```python
                !cd /content/SpectraFAN && mkdir -p "$DRIVE_ROOT/runs" && ln -sfn "$DRIVE_ROOT/runs" runs
                ```

                **Cell 8 (code):** profile. Use `!` line magic and `python -u` so stdout streams live.
                ```python
                !cd /content/SpectraFAN && uv run python -u -m spectrafan.profile \
                    --config configs/profile.yaml
                ```
                """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    import json
    import os
    from pathlib import Path

    import plotly.graph_objects as go
    import polars as pl

    # Default to the in-repo runs/. Set SPECTRAFAN_RUNS_DIR to a Drive-synced path
    # (or anywhere else) to inspect runs produced outside this checkout.
    RUNS_DIR = Path(os.environ.get("SPECTRAFAN_RUNS_DIR", "runs"))
    return RUNS_DIR, go, json, pl


@app.cell(hide_code=True)
def _(RUNS_DIR, mo):
    if RUNS_DIR.exists():
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.endswith("_profile")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    else:
        run_dirs = []
    run_picker = mo.ui.dropdown(
        options={p.name: p for p in run_dirs},
        value=run_dirs[0].name if run_dirs else None,
        label="Run dir",
    )
    (
        run_picker
        if run_dirs
        else mo.md(
            f"**No `_profile` runs found under `{RUNS_DIR}`.** Run `colab_profile.ipynb` first."
        )
    )
    return run_dirs, run_picker


@app.cell(hide_code=True)
def _(mo, run_dirs, run_picker):
    mo.stop(not run_dirs, mo.md("_Waiting for a run to be available._"))
    run_dir = run_picker.value
    mo.md(f"**Active run:** `{run_dir.name}`")
    return (run_dir,)


@app.cell(hide_code=True)
def _(json, pl, run_dir):
    df = pl.read_parquet(run_dir / "timings.parquet")
    summary = json.loads((run_dir / "summary.json").read_text())
    env = json.loads((run_dir / "env.json").read_text())
    return df, env, summary


@app.cell
def _(mo, summary):
    mo.md(f"""
    FAMs take **{summary["fams_pct_of_fwd"]:.1f}%** of forward time at
    ({summary["image_size"]}, {summary["batch_size"]}), split into
    **{summary["transform_pct_of_fams"]:.1f}%** transform (FFT + iFFT),
    **{summary["branches_pct_of_fams"]:.1f}%** branches, and the remainder in the final
    1x1 + residual. Non-FAM stages take **{100 - summary["fams_pct_of_fwd"]:.1f}%**.
    """)
    return


@app.cell(hide_code=True)
def _(df, go, pl):
    in_fam_cats = ["fft", "branches", "ifft", "final"]
    non_fam_cats = ["encoder", "decoder", "head", "other"]
    stack_order = ["fft", "branches", "ifft", "final", "encoder", "decoder", "head", "other"]

    # Aggregate in-FAM categories (sum across all FAMs); non-FAM categories already aggregated.
    in_fam_df = (
        df.filter(pl.col("category").is_in(in_fam_cats))
        .group_by(["image_size", "batch_size", "category"])
        .agg(pl.col("median_us").sum())
    )
    non_fam_df = df.filter(pl.col("category").is_in(non_fam_cats)).select(
        ["image_size", "batch_size", "category", "median_us"]
    )
    stack_df = pl.concat([in_fam_df, non_fam_df]).sort(["image_size", "batch_size", "category"])

    # Build one bar per (image_size, batch_size), one trace per category.
    sweep = (
        stack_df.select(["image_size", "batch_size"]).unique().sort(["image_size", "batch_size"])
    )
    x_labels = [f"({r['image_size']}, {r['batch_size']})" for r in sweep.iter_rows(named=True)]

    fig_stack = go.Figure()
    for cat in stack_order:
        ys = []
        for r in sweep.iter_rows(named=True):
            row = stack_df.filter(
                (pl.col("image_size") == r["image_size"])
                & (pl.col("batch_size") == r["batch_size"])
                & (pl.col("category") == cat)
            )
            ys.append(row["median_us"][0] if not row.is_empty() else 0.0)
        texts = [f"{y / 1000:.2f} ms" for y in ys]
        fig_stack.add_bar(name=cat, x=x_labels, y=ys, text=texts, textposition="inside")
    fig_stack.update_layout(
        barmode="stack",
        title="Forward time by category (median, microseconds)",
        xaxis_title="(image_size, batch_size)",
        yaxis_title="median_us",
    )
    fig_stack
    return


@app.cell(hide_code=True)
def _(df, mo, pl):
    # Build a per-skip table for each sweep config.
    sweep_configs = (
        df.select(["image_size", "batch_size"]).unique().sort(["image_size", "batch_size"])
    )
    in_fam_cats_t = ["fft", "branches", "ifft", "final"]

    tables = []
    for cfg_row in sweep_configs.iter_rows(named=True):
        cfg_df = df.filter(
            (pl.col("image_size") == cfg_row["image_size"])
            & (pl.col("batch_size") == cfg_row["batch_size"])
        )
        total_fwd_us = cfg_df.filter(pl.col("category") == "total_fwd")["median_us"][0]

        fam_rows = cfg_df.filter(pl.col("category") == "fam").sort("module_path")
        rows = []
        for fam in fam_rows.iter_rows(named=True):
            mp = fam["module_path"]
            internal = cfg_df.filter(
                (pl.col("module_path") == mp) & (pl.col("category").is_in(in_fam_cats_t))
            )
            by_cat = {r["category"]: r["median_us"] for r in internal.iter_rows(named=True)}
            total_us = sum(by_cat.values())
            rows.append(
                {
                    "skip": mp,
                    "spatial_hw": fam["spatial_hw"],
                    "fft_us": round(by_cat.get("fft", 0.0), 2),
                    "branches_us": round(by_cat.get("branches", 0.0), 2),
                    "ifft_us": round(by_cat.get("ifft", 0.0), 2),
                    "final_us": round(by_cat.get("final", 0.0), 2),
                    "total_us": round(total_us, 2),
                    "pct_of_fwd": (
                        round(100.0 * total_us / total_fwd_us, 2) if total_fwd_us > 0 else 0.0
                    ),
                }
            )
        df_subset = pl.DataFrame(rows)
        tables.append(
            mo.vstack(
                [
                    mo.md(
                        f"### Per-skip breakdown — image_size={cfg_row['image_size']}, batch_size={cfg_row['batch_size']}"
                    ),
                    mo.ui.table(df_subset),
                ]
            )
        )
    mo.vstack(tables)
    return


@app.cell(hide_code=True)
def _(env, json, mo, run_dir):
    mo.accordion(
        {
            "Config + environment": mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("### config.yaml"),
                            mo.md(f"```yaml\n{(run_dir / 'config.yaml').read_text()}\n```"),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("### env.json"),
                            mo.md(f"```json\n{json.dumps(env, indent=2)}\n```"),
                        ]
                    ),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo, run_dir):
    trace_path = run_dir / "chrome_trace.json"
    (
        mo.download(
            data=trace_path.read_bytes(),
            filename="chrome_trace.json",
            label="Download Chrome trace",
        )
        if trace_path.is_file()
        else mo.md("_No `chrome_trace.json` in this run._")
    )
    return


if __name__ == "__main__":
    app.run()
