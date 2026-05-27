import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # 07 — Compare adapted-track runs

        Overlays training curves and surfaces a side-by-side summary across runs in
        `runs/`. Picks up every directory under `runs/` that contains a `metrics.parquet`,
        so dropping in E3, E4, … just works — re-run the discovery cell after pulling
        a new run from Drive.
        """
    )

    return (mo,)


@app.cell(hide_code=True)
def _():
    from __future__ import annotations

    from pathlib import Path
    import json

    import polars as pl

    RUNS_DIR = Path("runs")
    BASELINE_NAME = "2026-05-23_235101_fanetmini"
    DECISION_DELTA = 0.005  # spec rule: keep if best val_iou >= baseline + 0.005


    def run_label(run_dir: Path) -> str:
        """Short label for plots. Strips the `YYYY-MM-DD_HHMMSS_` prefix when present."""
        parts = run_dir.name.split("_", 2)
        return parts[2] if len(parts) == 3 else run_dir.name


    def discover_runs(runs_dir: Path = RUNS_DIR) -> list[Path]:
        """All run dirs that contain a metrics.parquet, sorted by name (chronological)."""
        return sorted(p.parent for p in runs_dir.glob("*/metrics.parquet"))


    run_dirs = discover_runs()
    {p.name: run_label(p) for p in run_dirs}

    return (
        BASELINE_NAME,
        DECISION_DELTA,
        Path,
        RUNS_DIR,
        json,
        pl,
        run_dirs,
        run_label,
    )


@app.cell(hide_code=True)
def _(BASELINE_NAME, mo, run_dirs, run_label):
    _default_pick_dirs = [
        p for p in run_dirs
        if p.name == BASELINE_NAME or "_fanetmini_adapted_E" in p.name
    ]
    run_picker = mo.ui.multiselect(
        # display key = short label, returned value = full run dir name
        options={run_label(p): p.name for p in run_dirs},
        value=[run_label(p) for p in _default_pick_dirs],
        label="Runs to compare",
    )
    run_picker

    return (run_picker,)


@app.cell(hide_code=True)
def _(Path, RUNS_DIR, json, pl, run_label, run_picker):
    selected_dirs = [RUNS_DIR / name for name in run_picker.value]


    def _load_test_metrics(run_dir: Path) -> dict | None:
        f = run_dir / "test_metrics.json"
        return json.loads(f.read_text()) if f.is_file() else None


    runs: dict[str, dict] = {}
    for d in selected_dirs:
        label = run_label(d)
        df = pl.read_parquet(d / "metrics.parquet")
        runs[label] = {"dir": d, "metrics": df, "test": _load_test_metrics(d)}

    list(runs)

    return runs, selected_dirs


@app.cell(hide_code=True)
def _(pl, runs: dict[str, dict]):
    def _summarize(label: str, info: dict) -> dict:
        df = info["metrics"]
        last10 = df.tail(10)
        best_idx = df["val_iou"].arg_max()
        test = info["test"] or {}
        return {
            "run": label,
            "epochs": int(df["epoch"][-1]),
            "best_val_iou": float(df["val_iou"][best_idx]),
            "best_epoch": int(df["epoch"][best_idx]),
            "final10_val_mean": float(last10["val_iou"].mean()),
            "final10_val_std": float(last10["val_iou"].std()),
            "final_train_loss": float(df["train_loss"][-1]),
            "final_train_iou": float(df["train_iou"][-1]),
            "final_gap": float(df["train_iou"][-1] - df["val_iou"][-1]),
            "mean_wall_sec": float(df["wall_sec"].mean()) if "wall_sec" in df.columns else None,
            "test_iou": test.get("test_iou"),
        }


    summary_df = pl.DataFrame([_summarize(label, info) for label, info in runs.items()])
    summary_df

    return


@app.cell(hide_code=True)
def _(
    BASELINE_NAME,
    DECISION_DELTA,
    RUNS_DIR,
    mo,
    run_label,
    run_picker,
    runs: dict[str, dict],
):
    baseline_best = None
    if BASELINE_NAME in run_picker.value:
        baseline_label = run_label(RUNS_DIR / BASELINE_NAME)
        baseline_best = float(runs[baseline_label]["metrics"]["val_iou"].max())

    if baseline_best is not None:
        threshold_value = baseline_best + DECISION_DELTA
        threshold_md = mo.md(
            f"Baseline best val_iou: **{baseline_best:.4f}** &nbsp;|&nbsp; "
            f"Decision threshold (Δ ≥ +{DECISION_DELTA}): **{threshold_value:.4f}**"
        )
    else:
        threshold_value = None
        threshold_md = mo.md(f"_Baseline `{BASELINE_NAME}` not selected — threshold line hidden._")

    threshold_md

    return baseline_best, threshold_value


@app.cell(hide_code=True)
def _(runs: dict[str, dict]):
    import plotly.graph_objects as go

    _COLOR_CYCLE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    run_colors = {label: _COLOR_CYCLE[i % len(_COLOR_CYCLE)] for i, label in enumerate(runs)}


    def overlay(metric: str, *, title: str, yaxis: str, log_y: bool = False) -> go.Figure:
        """Overlay a single metric across all selected runs, with consistent colors."""
        fig = go.Figure()
        for label, info in runs.items():
            df = info["metrics"]
            if metric not in df.columns:
                continue
            fig.add_scatter(
                x=df["epoch"].to_list(),
                y=df[metric].to_list(),
                mode="lines",
                name=label,
                line=dict(color=run_colors[label]),
            )
        fig.update_layout(
            title=title, xaxis_title="epoch", yaxis_title=yaxis,
            template="plotly_white", height=380, hovermode="x unified",
        )
        if log_y:
            fig.update_yaxes(type="log")
        return fig


    return (overlay,)


@app.cell(hide_code=True)
def _(baseline_best, overlay, threshold_value):
    _fig_val = overlay("val_iou", title="val_iou", yaxis="val_iou")
    if threshold_value is not None:
        _fig_val.add_hline(
            y=threshold_value, line_dash="dash", line_color="#444",
            annotation_text=f"threshold {threshold_value:.4f}", annotation_position="top right",
        )
        _fig_val.add_hline(
            y=baseline_best, line_dash="dot", line_color="#888",
            annotation_text=f"baseline {baseline_best:.4f}", annotation_position="bottom right",
        )
    _fig_val

    return


@app.cell(hide_code=True)
def _(overlay):
    overlay("train_iou", title="train_iou", yaxis="train_iou")

    return


@app.cell(hide_code=True)
def _(overlay):
    overlay("train_loss", title="train_loss", yaxis="train_loss")

    return


@app.cell(hide_code=True)
def _(overlay):
    overlay("lr", title="lr schedule", yaxis="lr", log_y=True)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train vs val curves — single run
    """)
    return


@app.cell(hide_code=True)
def _(mo, run_label, selected_dirs):
    single_run_picker = mo.ui.dropdown(
        options={run_label(d): d.name for d in selected_dirs},
        value=run_label(selected_dirs[-1]) if selected_dirs else None,
        label="Run",
    )
    single_run_picker

    return (single_run_picker,)


@app.cell(hide_code=True)
def _(RUNS_DIR, run_label, runs: dict[str, dict], single_run_picker):
    from plotly.subplots import make_subplots

    _single_label = run_label(RUNS_DIR / single_run_picker.value)
    _df = runs[_single_label]["metrics"]
    _epochs = _df["epoch"].to_list()

    _pairs = [("iou", "train_iou", "val_iou"), ("dice", "train_dice", "val_dice"), ("loss", "train_loss", "val_loss")]
    _pairs = [(t, tr, va) for (t, tr, va) in _pairs if tr in _df.columns and va in _df.columns]

    _fig_tv = make_subplots(rows=len(_pairs), cols=1, subplot_titles=[t for (t, _, _) in _pairs], shared_xaxes=True, vertical_spacing=0.08)
    for _i, (_title, _tr, _va) in enumerate(_pairs, start=1):
        _fig_tv.add_scatter(x=_epochs, y=_df[_tr].to_list(), mode="lines", name=f"train ({_title})", line=dict(color="#1f77b4"), legendgroup=_title, row=_i, col=1)
        _fig_tv.add_scatter(x=_epochs, y=_df[_va].to_list(), mode="lines", name=f"val ({_title})", line=dict(color="#ff7f0e"), legendgroup=_title, row=_i, col=1)
    _fig_tv.update_layout(title=f"train vs val — {_single_label}", template="plotly_white", height=260 * len(_pairs), hovermode="x unified")
    _fig_tv.update_xaxes(title_text="epoch", row=len(_pairs), col=1)
    _fig_tv

    return


if __name__ == "__main__":
    app.run()
