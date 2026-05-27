import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Explore run

        Single-run inspector for a synced `runs/` directory. Reads `run.json` and renders
        only the sections the run produced — training curves, held-out test, FAM diagnosis,
        profiling. Pure artifact viewer: no model load, no dataset, no torch.

        Point at a Drive-synced path by exporting `SPECTRAFAN_RUNS_DIR` before launching marimo.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _():
    import json
    import os
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go
    import plotly.subplots as sp
    import polars as pl

    from spectrafan.manifest import load_manifest

    RUNS_DIR = Path(os.environ.get("SPECTRAFAN_RUNS_DIR", "runs"))
    return RUNS_DIR, go, json, load_manifest, np, pl, sp


@app.cell(hide_code=True)
def _(RUNS_DIR, mo):
    if RUNS_DIR.exists():
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "config.yaml").is_file()),
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
    run_picker if run_dirs else mo.md(f"**No runs found under `{RUNS_DIR}`.**")
    return run_dirs, run_picker


@app.cell(hide_code=True)
def _(load_manifest, mo, run_dirs, run_picker):
    mo.stop(not run_dirs, mo.md("_Waiting for a run._"))
    run_dir = run_picker.value
    manifest = load_manifest(run_dir)
    stages = manifest["stages"]
    mo.md(
        f"**{manifest['run_id']}** &nbsp;|&nbsp; model: `{manifest['model'] or '—'}` "
        f"&nbsp;|&nbsp; dataset: `{manifest['dataset'] or '—'}` "
        f"&nbsp;|&nbsp; stages: {', '.join(f'`{s}`' for s in stages) or '—'}"
    )
    return manifest, run_dir, stages


@app.cell(hide_code=True)
def _(mo, run_dir):
    mo.stop(not (run_dir / "config.yaml").is_file())
    config_text = (run_dir / "config.yaml").read_text()
    env_path = run_dir / "env.json"
    env_text = env_path.read_text() if env_path.is_file() else "(no env.json)"
    mo.hstack(
        [
            mo.vstack([mo.md("### config.yaml"), mo.md(f"```yaml\n{config_text}\n```")]),
            mo.vstack([mo.md("### env.json"), mo.md(f"```json\n{env_text}\n```")]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, stages):
    mo.stop("train" not in stages)
    mo.md("## Training curves")
    return


@app.cell(hide_code=True)
def _(go, mo, pl, run_dir, stages):
    mo.stop("train" not in stages)
    metrics_df = pl.read_parquet(run_dir / "metrics.parquet")

    def _curve(cols, title, yrange=None, log_y=False):
        e = metrics_df["epoch"].to_list()
        fig = go.Figure()
        for name in cols:
            if name in metrics_df.columns:
                fig.add_scatter(x=e, y=metrics_df[name].to_list(), mode="lines", name=name)
        fig.update_layout(title=title, xaxis_title="epoch", template="plotly_white", height=320)
        if yrange:
            fig.update_yaxes(range=yrange)
        if log_y:
            fig.update_yaxes(type="log")
        return fig

    mo.vstack(
        [
            _curve(["train_iou", "val_iou"], "IoU", yrange=[0, 1]),
            _curve(["train_dice", "val_dice"], "Dice", yrange=[0, 1]),
            _curve(["train_loss", "val_loss"], "Loss"),
            _curve(["lr"], "Learning rate", log_y=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, stages):
    mo.stop("predict" not in stages)
    mo.md("## Held-out test-set evaluation")
    return


@app.cell(hide_code=True)
def _(json, mo, run_dir):
    mo.stop(not (run_dir / "test_metrics.json").is_file())
    tm = json.loads((run_dir / "test_metrics.json").read_text())
    mo.md(
        f"""
        `best.pt` (epoch {tm["epoch"]}, val_iou={tm["val_iou"]:.4f}) on the test split
        ({tm["test_size"]} pairs).

        | metric | value |
        | --- | --- |
        | IoU | **{tm["test_iou"]:.4f}** |
        | Dice | **{tm["test_dice"]:.4f}** |
        | Pixel accuracy | **{tm["test_px_acc"]:.4f}** |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, np, run_dir, sp):
    mo.stop(not (run_dir / "predictions.npz").is_file())
    artifact = np.load(run_dir / "predictions.npz")

    def grid_figure(images, masks, preds, title):
        """4x3 (image | mask | pred) heatmap grid from (N, 1, H, W) arrays (first 4)."""
        fig = sp.make_subplots(
            rows=4,
            cols=3,
            subplot_titles=["image", "mask", "pred"] * 4,
            vertical_spacing=0.04,
            horizontal_spacing=0.02,
        )
        for row in range(4):
            fig.add_heatmap(z=images[row, 0], row=row + 1, col=1, showscale=False, colorscale="gray")
            fig.add_heatmap(z=masks[row, 0], row=row + 1, col=2, showscale=False, colorscale="gray")
            fig.add_heatmap(z=preds[row, 0], row=row + 1, col=3, showscale=False, colorscale="gray")
        fig.update_layout(height=1100, title=title, showlegend=False)
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
        return fig

    grid_figure(
        artifact["test_images"],
        artifact["test_masks"],
        artifact["test_preds"],
        "test predictions @ best.pt (first 4)",
    )
    return


@app.cell(hide_code=True)
def _(mo, stages):
    mo.stop("diagnose" not in stages)
    mo.md("## FAM diagnosis")
    return


@app.cell(hide_code=True)
def _(json, mo, run_dir):
    mo.stop(not (run_dir / "fam_diagnosis.json").is_file())
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
def _(mo, pl, run_dir):
    mo.stop(not (run_dir / "fam_stats.parquet").is_file())
    stats_df = pl.read_parquet(run_dir / "fam_stats.parquet")
    stats_df
    return (stats_df,)


@app.cell(hide_code=True)
def _(go, mo, pl, run_dir, stats_df):
    mo.stop(not (run_dir / "fam_stats.parquet").is_file())
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
def _(go, mo, pl, run_dir, stats_df):
    mo.stop(not (run_dir / "fam_stats.parquet").is_file())
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
    _fig2 = go.Figure()
    _fig2.add_bar(x=_scales, y=_medians["real"].to_list(), name="real branch")
    _fig2.add_bar(x=_scales, y=_medians["imag"].to_list(), name="imag branch")
    _fig2.update_layout(
        title="Median post-ReLU dead-activation rate, by scale",
        xaxis_title="FAM scale",
        yaxis_title="fraction of activations == 0",
        yaxis_range=[0, 1],
        barmode="group",
    )
    _fig2
    return


@app.cell(hide_code=True)
def _(go, mo, pl, run_dir, stats_df):
    mo.stop(not (run_dir / "fam_stats.parquet").is_file())
    _fig3 = go.Figure()
    for _scale in sorted(set(stats_df["scale_idx"].to_list())):
        _sub = stats_df.filter(pl.col("scale_idx") == _scale)
        _fig3.add_box(
            y=_sub["fft_real_dc_share"].to_list(),
            name=f"scale {_scale}",
            boxpoints="all",
            pointpos=0,
        )
    _fig3.update_layout(
        title="DC-bin share of FFT-real energy, by scale (1.0 = single bin dominates)",
        xaxis_title="FAM scale",
        yaxis_title="|freq.real[0,0]|^2 / total",
        yaxis_range=[0, 1],
        showlegend=False,
    )
    _fig3
    return


@app.cell(hide_code=True)
def _(blob, mo, pl, run_dir, stats_df):
    mo.stop(not (run_dir / "fam_diagnosis.json").is_file())
    mo.stop(not (run_dir / "fam_stats.parquet").is_file())
    _iou2 = blob["val_iou"]
    _delta_skip = _iou2["fam_skip_fft"] - _iou2["as_trained"]
    _delta_zero = _iou2["fam_zero"] - _iou2["as_trained"]
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


@app.cell(hide_code=True)
def _(mo, stages):
    mo.stop("profile" not in stages)
    mo.md("## Profiling")
    return


@app.cell(hide_code=True)
def _(json, mo, run_dir):
    mo.stop(not (run_dir / "summary.json").is_file())
    s = json.loads((run_dir / "summary.json").read_text())
    rows = "\n".join(f"| {k} | {v} |" for k, v in s.items())
    mo.md("### Summary\n\n| key | value |\n| --- | --- |\n" + rows)
    return


@app.cell(hide_code=True)
def _(go, mo, pl, run_dir):
    mo.stop(not (run_dir / "timings.parquet").is_file())
    df = pl.read_parquet(run_dir / "timings.parquet")

    in_fam_cats = ["fft", "branches", "ifft", "final"]
    non_fam_cats = ["encoder", "decoder", "head", "other"]
    stack_order = ["fft", "branches", "ifft", "final", "encoder", "decoder", "head", "other"]

    in_fam_df = (
        df.filter(pl.col("category").is_in(in_fam_cats))
        .group_by(["image_size", "batch_size", "category"])
        .agg(pl.col("median_us").sum())
    )
    non_fam_df = df.filter(pl.col("category").is_in(non_fam_cats)).select(
        ["image_size", "batch_size", "category", "median_us"]
    )
    stack_df = pl.concat([in_fam_df, non_fam_df]).sort(["image_size", "batch_size", "category"])

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


if __name__ == "__main__":
    app.run()
