import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 05 — FANetMini run inspector

        Inspect a `configs/fanetmini.yaml` training run produced on Colab. Renders curves,
        config + env.json, sample predictions, and held-out test-set metrics. Reads
        pre-computed artifacts (`predictions.npz`, `test_metrics.json`) emitted by
        `spectrafan.predict` — no local inference.
        """
    )

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "How to reproduce a run (Colab A100)": mo.md(
                r"""
                Create a Jupyter notebook in any Google Drive folder (e.g.
                `My Drive/SpectraFAN/colab_fanetmini.ipynb` — `colab_*.ipynb` is
                gitignored). Paste these eight cells in order, edit `DRIVE_ROOT` in
                cell 3 to match your Drive layout, pick an A100 runtime, and run
                all. Artifacts land in `<DRIVE_ROOT>/runs/<timestamp>_fanetmini/`.
                Mount that Drive locally and point this notebook at it by
                exporting `SPECTRAFAN_RUNS_DIR=<your-drive-runs-path>` before
                launching marimo.

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

                **Cell 8 (code):** train + dump inference artifacts. `!` line magic
                with `python -u` so stdout streams live; `&&` chains predict only
                if train succeeds.
                ```python
                !cd /content/SpectraFAN && \
                  uv run python -u -m spectrafan.train --config configs/fanetmini.yaml && \
                  uv run python -m spectrafan.predict --latest fanetmini
                ```
                On a Colab session disconnect, reconnect, re-run cells 2-7
                (idempotent), then add `--resume runs/<timestamp>_fanetmini/last.pt`
                to the train line in cell 8.
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

    import numpy as np
    import plotly.graph_objects as go
    import plotly.subplots as sp
    import polars as pl

    # Default to the in-repo runs/. Set SPECTRAFAN_RUNS_DIR to a Drive-synced path
    # (or anywhere else) to inspect runs produced outside this checkout.
    RUNS_DIR = Path(os.environ.get("SPECTRAFAN_RUNS_DIR", "runs"))

    return RUNS_DIR, go, json, np, pl, sp


@app.cell(hide_code=True)
def _(RUNS_DIR, mo):
    if RUNS_DIR.exists():
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir() and "_fanetmini" in p.name),
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
            f"**No `fanetmini` runs found under `{RUNS_DIR}`.** Run `colab_fanetmini.ipynb` first."
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
def _(mo, run_dir):
    config_text = (run_dir / "config.yaml").read_text()
    env_text = (
        (run_dir / "env.json").read_text()
        if (run_dir / "env.json").is_file()
        else "(no env.json — run predates this artifact)"
    )

    mo.hstack(
        [
            mo.vstack([mo.md("### config.yaml"), mo.md(f"```yaml\n{config_text}\n```")]),
            mo.vstack([mo.md("### env.json"), mo.md(f"```json\n{env_text}\n```")]),
        ]
    )

    return


@app.cell(hide_code=True)
def _(pl, run_dir):
    metrics_df = pl.read_parquet(run_dir / "metrics.parquet")
    metrics_df

    return (metrics_df,)


@app.cell(hide_code=True)
def _(go, metrics_df):
    _epochs = metrics_df["epoch"].to_list()
    fig_iou = go.Figure()
    fig_iou.add_scatter(
        x=_epochs, y=metrics_df["train_iou"].to_list(), mode="lines", name="train_iou"
    )
    fig_iou.add_scatter(x=_epochs, y=metrics_df["val_iou"].to_list(), mode="lines", name="val_iou")
    fig_iou.update_layout(title="IoU", xaxis_title="epoch", yaxis_title="iou", yaxis_range=[0, 1])
    fig_iou

    return


@app.cell(hide_code=True)
def _(go, metrics_df):
    _epochs_d = metrics_df["epoch"].to_list()
    fig_dice = go.Figure()
    fig_dice.add_scatter(
        x=_epochs_d, y=metrics_df["train_dice"].to_list(), mode="lines", name="train_dice"
    )
    fig_dice.add_scatter(
        x=_epochs_d, y=metrics_df["val_dice"].to_list(), mode="lines", name="val_dice"
    )
    fig_dice.update_layout(
        title="Dice", xaxis_title="epoch", yaxis_title="dice", yaxis_range=[0, 1]
    )
    fig_dice

    return


@app.cell(hide_code=True)
def _(go, metrics_df):
    _epochs_l = metrics_df["epoch"].to_list()
    fig_loss = go.Figure()
    fig_loss.add_scatter(
        x=_epochs_l, y=metrics_df["train_loss"].to_list(), mode="lines", name="train_loss"
    )
    fig_loss.add_scatter(
        x=_epochs_l, y=metrics_df["val_loss"].to_list(), mode="lines", name="val_loss"
    )
    fig_loss.update_layout(
        title="Loss (0.5 * BCE + 0.5 * Dice)", xaxis_title="epoch", yaxis_title="loss"
    )
    fig_loss

    return


@app.cell(hide_code=True)
def _(go, metrics_df):
    fig_lr = go.Figure()
    fig_lr.add_scatter(
        x=metrics_df["epoch"].to_list(), y=metrics_df["lr"].to_list(), mode="lines", name="lr"
    )
    fig_lr.update_layout(
        title="Learning rate (ExponentialLR γ=0.99)",
        xaxis_title="epoch",
        yaxis_title="lr",
        yaxis_type="log",
    )
    fig_lr

    return


@app.cell(hide_code=True)
def _(json, mo, np, run_dir):
    pred_path = run_dir / "predictions.npz"
    metrics_path = run_dir / "test_metrics.json"
    has_artifacts = pred_path.is_file() and metrics_path.is_file()

    if has_artifacts:
        artifact = np.load(pred_path)
        test_metrics = json.loads(metrics_path.read_text())
        missing_msg = None
    else:
        artifact = None
        test_metrics = None
        missing_msg = mo.md(
            "**No predictions artifact for this run.** Run "
            f"`uv run python -m spectrafan.predict --run {run_dir}` "
            "from a machine with the dataset to generate predictions."
        )

    return artifact, has_artifacts, missing_msg, test_metrics


@app.cell(hide_code=True)
def _(has_artifacts, missing_msg, mo, test_metrics):
    mo.stop(not has_artifacts, missing_msg)
    mo.md(
        f"""
        ### Held-out test-set evaluation

        Evaluated `best.pt` (epoch {test_metrics["epoch"]}, val_iou={test_metrics["val_iou"]:.4f}) on the **test split** ({test_metrics["test_size"]} pairs).
        Test IoU within ~0.05 of val IoU is the spec's sanity check.

        | metric | value |
        | --- | --- |
        | IoU | **{test_metrics["test_iou"]:.4f}** |
        | Dice | **{test_metrics["test_dice"]:.4f}** |
        | Pixel accuracy | **{test_metrics["test_px_acc"]:.4f}** |
        """
    )

    return


@app.cell(hide_code=True)
def _(artifact, has_artifacts, mo, sp):
    mo.stop(not has_artifacts)

    def grid_figure(images, masks, preds, title):
        """4x3 (image | mask | pred) heatmap grid from (N, 1, H, W) arrays (uses first 4)."""
        fig = sp.make_subplots(
            rows=4,
            cols=3,
            subplot_titles=["image", "mask", "pred"] * 4,
            vertical_spacing=0.04,
            horizontal_spacing=0.02,
        )
        for row in range(4):
            fig.add_heatmap(
                z=images[row, 0], row=row + 1, col=1, showscale=False, colorscale="gray"
            )
            fig.add_heatmap(z=masks[row, 0], row=row + 1, col=2, showscale=False, colorscale="gray")
            fig.add_heatmap(z=preds[row, 0], row=row + 1, col=3, showscale=False, colorscale="gray")
        fig.update_layout(height=1100, title=title, showlegend=False)
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
        return fig

    grid_figure(
        artifact["val_images"],
        artifact["val_masks"],
        artifact["val_preds"],
        "val predictions @ best.pt (first 4 of 16)",
    )

    return (grid_figure,)


@app.cell(hide_code=True)
def _(artifact, grid_figure, has_artifacts, mo):
    mo.stop(not has_artifacts)
    grid_figure(
        artifact["test_images"],
        artifact["test_masks"],
        artifact["test_preds"],
        "test predictions @ best.pt (first 4 of 16)",
    )

    return
