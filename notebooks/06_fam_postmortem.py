import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 06 — FAM post-mortem

        Verdict on whether the Frequency Attention Module is contributing to the
        FANetMini ceiling. Reads `fam_stats.parquet` + `fam_diagnosis.json`
        emitted by `spectrafan.diagnose_fam` — no local inference.
        """
    )

    return (mo,)


@app.cell(hide_code=True)
def _():
    import json
    import os
    from pathlib import Path

    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl

    RUNS_DIR = Path(os.environ.get("SPECTRAFAN_RUNS_DIR", "runs"))

    return RUNS_DIR, go, json, pl, px


@app.cell(hide_code=True)
def _(RUNS_DIR, mo):
    if RUNS_DIR.exists():
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "fam_diagnosis.json").is_file()),
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
            f"**No diagnosed runs found under `{RUNS_DIR}`.** Run "
            "`uv run python -m spectrafan.diagnose_fam --latest fanetmini` first."
        )
    )

    return run_dirs, run_picker


@app.cell(hide_code=True)
def _(mo, run_dirs, run_picker):
    mo.stop(not run_dirs, mo.md("_Waiting for a diagnosed run to be available._"))
    run_dir = run_picker.value
    mo.md(f"**Active run:** `{run_dir.name}`")

    return (run_dir,)


@app.cell(hide_code=True)
def _(json, mo, run_dir):
    blob = json.loads((run_dir / "fam_diagnosis.json").read_text())
    iou = blob["val_iou"]
    mo.md(
        f"""
        ### Counterfactual val_iou

        Checkpoint: `best.pt` (epoch {blob["epoch"]}, {blob["n_fam_modules"]} FAMs, {blob["n_val_batches"]} val batches).

        | mode | val_iou | delta vs as_trained |
        | --- | ---: | ---: |
        | `as_trained` | **{iou["as_trained"]:.4f}** | — |
        | `fam_skip_fft` | {iou["fam_skip_fft"]:.4f} | {iou["fam_skip_fft"] - iou["as_trained"]:+.4f} |
        | `fam_zero` | {iou["fam_zero"]:.4f} | {iou["fam_zero"] - iou["as_trained"]:+.4f} |
        """
    )

    return (blob,)


@app.cell(hide_code=True)
def _(pl, run_dir):
    stats_df = pl.read_parquet(run_dir / "fam_stats.parquet")
    stats_df

    return (stats_df,)


@app.cell(hide_code=True)
def _(pl, px, stats_df):
    ratio = stats_df.with_columns(
        (pl.col("contribution_norm") / pl.col("input_norm")).alias("contribution_ratio")
    )
    fig = px.box(
        ratio.to_pandas(),
        x="scale_idx",
        y="contribution_ratio",
        points="all",
        title="FFT pathway contribution norm / input norm, by scale",
        labels={
            "scale_idx": "FAM scale (0=shallow)",
            "contribution_ratio": "||spatial_hat|| / ||x||",
        },
    )
    fig.add_hline(y=0.1, line_dash="dash", annotation_text="0.1 (rule-of-thumb healthy)")
    fig

    return


@app.cell(hide_code=True)
def _(go, pl, stats_df):
    medians = (
        stats_df.group_by("scale_idx")
        .agg(
            [
                pl.col("branch_real_dead_rate").median().alias("real"),
                pl.col("branch_imag_dead_rate").median().alias("imag"),
            ]
        )
        .sort("scale_idx")
    )
    scales = medians["scale_idx"].to_list()
    fig = go.Figure()
    fig.add_bar(x=scales, y=medians["real"].to_list(), name="real branch")
    fig.add_bar(x=scales, y=medians["imag"].to_list(), name="imag branch")
    fig.update_layout(
        title="Median post-ReLU dead-activation rate, by scale",
        xaxis_title="FAM scale",
        yaxis_title="fraction of activations == 0",
        yaxis_range=[0, 1],
        barmode="group",
    )
    fig

    return


@app.cell(hide_code=True)
def _(px, stats_df):
    fig = px.box(
        stats_df.to_pandas(),
        x="scale_idx",
        y="fft_real_dc_share",
        points="all",
        title="DC-bin share of FFT-real energy, by scale (1.0 = single bin dominates)",
        labels={"scale_idx": "FAM scale", "fft_real_dc_share": "|freq.real[0,0]|^2 / total"},
    )
    fig.update_yaxes(range=[0, 1])
    fig

    return


@app.cell(hide_code=True)
def _(blob, mo, pl, stats_df):
    iou = blob["val_iou"]
    delta_skip = iou["fam_skip_fft"] - iou["as_trained"]
    delta_zero = iou["fam_zero"] - iou["as_trained"]
    median_ratio_by_scale = (
        stats_df.with_columns((pl.col("contribution_norm") / pl.col("input_norm")).alias("ratio"))
        .group_by("scale_idx")
        .agg(pl.col("ratio").median().alias("median_ratio"))
        .sort("scale_idx")
    )
    ratios = dict(
        zip(
            median_ratio_by_scale["scale_idx"].to_list(),
            median_ratio_by_scale["median_ratio"].to_list(),
            strict=True,
        )
    )
    min_ratio = min(ratios.values())
    max_ratio = max(ratios.values())

    if abs(delta_zero) < 0.005:
        verdict = (
            "**Entire FAM block is dead.** `fam_zero` matches `as_trained` within "
            f"{delta_zero:+.4f}. The 1x1 final conv is also wasted. Next step: "
            "FAM redesign (see decision table in the spec)."
        )
    elif abs(delta_skip) < 0.005:
        verdict = (
            "**FFT pathway is dead.** `fam_skip_fft` matches `as_trained` within "
            f"{delta_skip:+.4f} — the 1x1 projection is doing all the work. "
            "Next step: FAM redesign (see decision table in the spec)."
        )
    elif delta_skip > 0:
        verdict = (
            f"**FFT pathway is actively hurting.** `fam_skip_fft` is *better* "
            f"by {delta_skip:+.4f}. Strong evidence for redesign or removal."
        )
    elif delta_skip < -0.05 and delta_zero < -0.05 and min_ratio > 0.1:
        verdict = (
            f"**FAM is healthy.** Both ablations hurt by > 0.05 and contribution "
            f"ratio at all scales > 0.1 (min={min_ratio:.3f}, max={max_ratio:.3f}). "
            "The ceiling is recipe-bound. Next step: resume the frozen B/C/D sweep."
        )
    else:
        verdict = (
            f"**Ambiguous.** delta_skip={delta_skip:+.4f}, delta_zero={delta_zero:+.4f}, "
            f"contribution ratio min={min_ratio:.3f} max={max_ratio:.3f}. Per-scale ratios: "
            f"{', '.join(f'scale {k}: {v:.3f}' for k, v in ratios.items())}. "
            "Re-read the decision table in the spec — this case is between buckets."
        )
    mo.md(f"### Verdict\n\n{verdict}")

    return
