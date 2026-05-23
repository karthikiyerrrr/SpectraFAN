import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 03b — FANet full reproduction inspector

        Inspect a `configs/full_repro.yaml` training run produced on Colab. Renders curves,
        config + env.json, sample predictions, and a held-out test-set evaluation.
        """
    )

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "How to reproduce a run (Colab L4/A100)": mo.md(
                r"""
                Create a Jupyter notebook in your Drive folder
                `My Drive/03 Projects/02 SpectraFAN/` named `colab_full_repro.ipynb`
                (kept out of version control on purpose). Paste these eight cells in order,
                pick an L4 or A100 runtime, and run all. Artifacts land in
                `My Drive/03 Projects/02 SpectraFAN/runs/<timestamp>_full_repro/`, which
                Google Drive desktop mirrors to your laptop where this notebook reads them.

                **Cell 1 (markdown):** title / one-line description.

                **Cell 2 (code):** mount Drive.
                ```python
                from google.colab import drive
                drive.mount('/content/drive')
                ```

                **Cell 3 (code):** anchor Drive root.
                ```python
                import os
                DRIVE_ROOT = '/content/drive/MyDrive/03 Projects/02 SpectraFAN'
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

                **Cell 8 (code):** train. Use `!` line magic and `python -u` so stdout streams live.
                ```python
                !cd /content/SpectraFAN && uv run python -u -m spectrafan.train --config configs/full_repro.yaml
                ```
                On a Colab session disconnect, reconnect, re-run cells 2-7 (idempotent), then
                add `--resume runs/<timestamp>_full_repro/last.pt` to cell 8 and re-run it.

                To extend a finished run from 50 → 100 epochs, override the epoch count:
                ```python
                !cd /content/SpectraFAN && uv run python -u -m spectrafan.train \
                    --config configs/full_repro.yaml \
                    --override train.epochs=100 \
                    --resume runs/<timestamp>_full_repro/last.pt
                ```
                """
            ),
        }
    )

    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import plotly.graph_objects as go
    import plotly.subplots as sp
    import polars as pl
    import torch
    from torch.utils.data import DataLoader

    from spectrafan.data import TEMImageNetDataset
    from spectrafan.metrics import RunningMetrics
    from spectrafan.train import load_config
    from spectrafan.transforms import eval_transforms
    from spectrafan.unet import FANet

    # Primary location is the in-repo runs/, mirrored from the Drive output of the Colab launcher.
    RUNS_DIR_CANDIDATES = [
        Path("runs"),
        Path.home()
        / "Library/CloudStorage/GoogleDrive-kbi102003@gmail.com/My Drive/03 Projects/02 SpectraFAN/runs",
    ]
    RUNS_DIR = next((p for p in RUNS_DIR_CANDIDATES if p.exists()), Path("runs"))

    return (
        DataLoader,
        FANet,
        RUNS_DIR,
        RunningMetrics,
        TEMImageNetDataset,
        eval_transforms,
        go,
        load_config,
        pl,
        sp,
        torch,
    )


@app.cell(hide_code=True)
def _(RUNS_DIR, mo):
    if RUNS_DIR.exists():
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.endswith("_full_repro")),
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
            f"**No `_full_repro` runs found under `{RUNS_DIR}`.** Run `colab_full_repro.ipynb` first."
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
def _(
    FANet,
    TEMImageNetDataset,
    eval_transforms,
    load_config,
    mo,
    run_dir,
    sp,
    torch,
):
    cfg = load_config(run_dir / "config.yaml")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = FANet(
        channels=tuple(cfg.model.channels),
        bottleneck=cfg.model.bottleneck,
        output_norm=cfg.model.output_norm,
        conv_kind=cfg.model.fam_conv_kind,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def predict_samples(split: str, n: int):
        """Run best.pt on n samples from `split` and return (images, masks, binarized preds) as lists of numpy arrays."""
        ds = TEMImageNetDataset(
            root=cfg.data.root,
            split=split,
            image_size=cfg.data.image_size,
            splits_dir=cfg.data.splits_dir,
            transforms=eval_transforms(),
            subset_size=n,
        )
        imgs, msks, prds = [], [], []
        with torch.no_grad():
            for idx in range(len(ds)):
                x, y = ds[idx]
                logit = model(x.unsqueeze(0))
                p = torch.sigmoid(logit)[0, 0].cpu().numpy()
                imgs.append(x[0].numpy())
                msks.append(y[0].numpy())
                prds.append((p > 0.5).astype("float32"))
        return imgs, msks, prds

    def grid_figure(imgs, msks, prds, title):
        """Build a 4x3 (image, mask, pred) heatmap grid."""
        fig = sp.make_subplots(
            rows=4,
            cols=3,
            subplot_titles=["image", "mask", "pred"] * 4,
            vertical_spacing=0.04,
            horizontal_spacing=0.02,
        )
        for row in range(4):
            fig.add_heatmap(z=imgs[row], row=row + 1, col=1, showscale=False, colorscale="gray")
            fig.add_heatmap(z=msks[row], row=row + 1, col=2, showscale=False, colorscale="gray")
            fig.add_heatmap(z=prds[row], row=row + 1, col=3, showscale=False, colorscale="gray")
        fig.update_layout(height=1100, title=title, showlegend=False)
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
        return fig

    mo.md(
        f"Loaded `best.pt` from **epoch {ckpt['epoch'] + 1}** with **val_iou = {ckpt['val_iou']:.4f}**."
    )

    return cfg, ckpt, grid_figure, model, predict_samples


@app.cell(hide_code=True)
def _(ckpt, grid_figure, predict_samples):
    val_imgs, val_msks, val_prds = predict_samples("val", 4)
    grid_figure(
        val_imgs, val_msks, val_prds, f"val predictions @ best.pt (epoch {ckpt['epoch'] + 1})"
    )

    return


@app.cell(hide_code=True)
def _(
    DataLoader,
    RunningMetrics,
    TEMImageNetDataset,
    cfg,
    ckpt,
    eval_transforms,
    mo,
    model,
    torch,
):
    def _eval_test_set():
        ds = TEMImageNetDataset(
            root=cfg.data.root,
            split="test",
            image_size=cfg.data.image_size,
            splits_dir=cfg.data.splits_dir,
            transforms=eval_transforms(),
        )
        loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
        rm = RunningMetrics()
        with torch.no_grad():
            for xb, yb in loader:
                rm.update(model(xb), yb)
        return rm.compute(), len(ds)

    test_metrics, test_size = _eval_test_set()

    mo.md(
        f"""
        ### Held-out test-set evaluation

        Evaluated `best.pt` (epoch {ckpt["epoch"] + 1}, val_iou={ckpt["val_iou"]:.4f}) on the **test split** ({test_size} pairs).
        Test IoU within ~0.05 of val IoU is the spec's sanity check.

        | metric | value |
        | --- | --- |
        | IoU | **{test_metrics["iou"]:.4f}** |
        | Dice | **{test_metrics["dice"]:.4f}** |
        | Pixel accuracy | **{test_metrics["px_acc"]:.4f}** |
        """
    )

    return


@app.cell(hide_code=True)
def _(ckpt, grid_figure, predict_samples):
    test_imgs, test_msks, test_prds = predict_samples("test", 4)
    grid_figure(
        test_imgs, test_msks, test_prds, f"test predictions @ best.pt (epoch {ckpt['epoch'] + 1})"
    )

    return


if __name__ == "__main__":
    app.run()
