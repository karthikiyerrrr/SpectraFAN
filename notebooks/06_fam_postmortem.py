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

    import plotly.graph_objects as go
    import polars as pl

    RUNS_DIR = Path(os.environ.get("SPECTRAFAN_RUNS_DIR", "runs"))
    return RUNS_DIR, go, json, pl


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
    _iou = blob["val_iou"]
    mo.md(
        f"""
        ### Counterfactual val_iou

        Checkpoint: `best.pt` (epoch {blob["epoch"]}, {blob["n_fam_modules"]} FAMs, {blob["n_val_batches"]} val batches).

        | mode | val_iou | delta vs as_trained |
        | --- | ---: | ---: |
        | `as_trained` | **{_iou["as_trained"]:.4f}** | — |
        | `fam_skip_fft` | {_iou["fam_skip_fft"]:.4f} | {_iou["fam_skip_fft"] - _iou["as_trained"]:+.4f} |
        | `fam_zero` | {_iou["fam_zero"]:.4f} | {_iou["fam_zero"] - _iou["as_trained"]:+.4f} |
        """
    )
    return (blob,)


@app.cell(hide_code=True)
def _(pl, run_dir):
    stats_df = pl.read_parquet(run_dir / "fam_stats.parquet")
    stats_df
    return (stats_df,)


@app.cell(hide_code=True)
def _(go, pl, stats_df):
    _ratio = stats_df.with_columns(
        (pl.col("contribution_norm") / pl.col("input_norm")).alias("contribution_ratio")
    )
    _fig = go.Figure()
    for _scale in sorted(set(_ratio["scale_idx"].to_list())):
        _sub = _ratio.filter(pl.col("scale_idx") == _scale)
        _fig.add_box(
            y=_sub["contribution_ratio"].to_list(),
            name=f"scale {_scale}",
            boxpoints="all",
            pointpos=0,
        )
    _fig.add_hline(y=0.1, line_dash="dash", annotation_text="0.1 (rule-of-thumb healthy)")
    _fig.update_layout(
        title="FFT pathway contribution norm / input norm, by scale",
        xaxis_title="FAM scale (0=shallow)",
        yaxis_title="||spatial_hat|| / ||x||",
        showlegend=False,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(go, pl, stats_df):
    _medians = (
        stats_df.group_by("scale_idx")
        .agg(
            [
                pl.col("branch_real_dead_rate").median().alias("real"),
                pl.col("branch_imag_dead_rate").median().alias("imag"),
            ]
        )
        .sort("scale_idx")
    )
    _scales = _medians["scale_idx"].to_list()
    _fig = go.Figure()
    _fig.add_bar(x=_scales, y=_medians["real"].to_list(), name="real branch")
    _fig.add_bar(x=_scales, y=_medians["imag"].to_list(), name="imag branch")
    _fig.update_layout(
        title="Median post-ReLU dead-activation rate, by scale",
        xaxis_title="FAM scale",
        yaxis_title="fraction of activations == 0",
        yaxis_range=[0, 1],
        barmode="group",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(go, pl, stats_df):
    _fig = go.Figure()
    for _scale in sorted(set(stats_df["scale_idx"].to_list())):
        _sub = stats_df.filter(pl.col("scale_idx") == _scale)
        _fig.add_box(
            y=_sub["fft_real_dc_share"].to_list(),
            name=f"scale {_scale}",
            boxpoints="all",
            pointpos=0,
        )
    _fig.update_layout(
        title="DC-bin share of FFT-real energy, by scale (1.0 = single bin dominates)",
        xaxis_title="FAM scale",
        yaxis_title="|freq.real[0,0]|^2 / total",
        yaxis_range=[0, 1],
        showlegend=False,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(blob, mo, pl, stats_df):
    _iou = blob["val_iou"]
    _delta_skip = _iou["fam_skip_fft"] - _iou["as_trained"]
    _delta_zero = _iou["fam_zero"] - _iou["as_trained"]
    _median_ratio_by_scale = (
        stats_df.with_columns((pl.col("contribution_norm") / pl.col("input_norm")).alias("ratio"))
        .group_by("scale_idx")
        .agg(pl.col("ratio").median().alias("median_ratio"))
        .sort("scale_idx")
    )
    _ratios = dict(
        zip(
            _median_ratio_by_scale["scale_idx"].to_list(),
            _median_ratio_by_scale["median_ratio"].to_list(),
            strict=True,
        )
    )
    _min_ratio = min(_ratios.values())
    _max_ratio = max(_ratios.values())

    if abs(_delta_zero) < 0.005:
        _verdict = (
            "**Entire FAM block is dead.** `fam_zero` matches `as_trained` within "
            f"{_delta_zero:+.4f}. The 1x1 final conv is also wasted. Next step: "
            "FAM redesign (see decision table in the spec)."
        )
    elif abs(_delta_skip) < 0.005:
        _verdict = (
            "**FFT pathway is dead.** `fam_skip_fft` matches `as_trained` within "
            f"{_delta_skip:+.4f} — the 1x1 projection is doing all the work. "
            "Next step: FAM redesign (see decision table in the spec)."
        )
    elif _delta_skip > 0:
        _verdict = (
            f"**FFT pathway is actively hurting.** `fam_skip_fft` is *better* "
            f"by {_delta_skip:+.4f}. Strong evidence for redesign or removal."
        )
    elif _delta_skip < -0.05 and _delta_zero < -0.05 and _min_ratio > 0.1:
        _verdict = (
            f"**FAM is healthy.** Both ablations hurt by > 0.05 and contribution "
            f"ratio at all scales > 0.1 (min={_min_ratio:.3f}, max={_max_ratio:.3f}). "
            "The ceiling is recipe-bound. Next step: resume the frozen B/C/D sweep."
        )
    else:
        _verdict = (
            f"**Ambiguous.** delta_skip={_delta_skip:+.4f}, delta_zero={_delta_zero:+.4f}, "
            f"contribution ratio min={_min_ratio:.3f} max={_max_ratio:.3f}. Per-scale ratios: "
            f"{', '.join(f'scale {k}: {v:.3f}' for k, v in _ratios.items())}. "
            "Re-read the decision table in the spec — this case is between buckets."
        )
    mo.md("### Verdict\n\n" + _verdict)
    return


if __name__ == "__main__":
    app.run()
