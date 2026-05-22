import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import torch
    import yaml
    from plotly.subplots import make_subplots

    from spectrafan.data import TEMImageNetDataset
    from spectrafan.unet import FANet

    RUNS_DIR = Path("runs")
    DEVICE = "cpu"
    return (
        DEVICE,
        FANet,
        RUNS_DIR,
        TEMImageNetDataset,
        go,
        make_subplots,
        mo,
        np,
        pl,
        torch,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Reproduction baseline

    Interactive view of a smoke / reproduction training run: pick a run, inspect its config, compare loss / IoU / Dice curves, and spot-check predictions from the best checkpoint.
    """)
    return


@app.cell(hide_code=True)
def _(RUNS_DIR, mo):
    if RUNS_DIR.exists():
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    else:
        run_dirs = []

    run_picker = mo.ui.dropdown(
        options={p.name: p for p in run_dirs},
        value=run_dirs[0].name if run_dirs else None,
        label="Run",
    )
    (
        run_picker
        if run_dirs
        else mo.md(
            "**No runs found in `runs/`.** "
            "Run `uv run python -m spectrafan.train --config configs/smoke.yaml` first."
        )
    )
    return run_dirs, run_picker


@app.cell(hide_code=True)
def _(mo, run_dirs, run_picker):
    mo.stop(not run_dirs, mo.md("_Waiting for a run to be available._"))
    run_dir = run_picker.value
    run_dir
    return (run_dir,)


@app.cell(hide_code=True)
def _(pl, run_dir, yaml):
    with open(run_dir / "config.yaml") as f:
        run_config = yaml.safe_load(f)

    config_rows = []
    for section, params in run_config.items():
        for k, v in params.items():
            config_rows.append({"section": section, "key": k, "value": str(v)})
    config_df = pl.DataFrame(config_rows)
    config_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Training curves
    """)
    return


@app.cell
def _(pl, run_dir):
    metrics_df = pl.read_parquet(run_dir / "metrics.parquet")
    metrics_df
    return (metrics_df,)


@app.cell(hide_code=True)
def _(go, metrics_df):
    _epochs = metrics_df["epoch"].to_numpy()
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_epochs,
            y=metrics_df["train_loss"].to_numpy(),
            mode="lines+markers",
            name="train",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_epochs,
            y=metrics_df["val_loss"].to_numpy(),
            mode="lines+markers",
            name="val",
        )
    )
    _fig.update_layout(
        title="Loss",
        xaxis_title="epoch",
        yaxis_title="loss",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(go, metrics_df):
    _epochs = metrics_df["epoch"].to_numpy()
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_epochs,
            y=metrics_df["train_iou"].to_numpy(),
            mode="lines+markers",
            name="train",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_epochs,
            y=metrics_df["val_iou"].to_numpy(),
            mode="lines+markers",
            name="val",
        )
    )
    _fig.update_layout(
        title="IoU",
        xaxis_title="epoch",
        yaxis_title="IoU",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(go, metrics_df):
    _epochs = metrics_df["epoch"].to_numpy()
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_epochs,
            y=metrics_df["train_dice"].to_numpy(),
            mode="lines+markers",
            name="train",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_epochs,
            y=metrics_df["val_dice"].to_numpy(),
            mode="lines+markers",
            name="val",
        )
    )
    _fig.update_layout(
        title="Dice",
        xaxis_title="epoch",
        yaxis_title="Dice",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Predictions from `best.pt`
    """)
    return


@app.cell
def _(DEVICE, FANet, run_dir, torch):
    ckpt = torch.load(run_dir / "best.pt", map_location=DEVICE, weights_only=False)
    m_cfg = ckpt["config"]["model"]
    model = FANet(
        channels=tuple(m_cfg["channels"]),
        bottleneck=m_cfg["bottleneck"],
        conv_kind=m_cfg["fam_conv_kind"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    f"loaded best.pt @ epoch {ckpt['epoch']}, val_iou={ckpt['val_iou']:.4f}"
    return ckpt, model


@app.cell
def _(TEMImageNetDataset, ckpt, model, torch):
    d_cfg = ckpt["config"]["data"]
    val_dataset = TEMImageNetDataset(
        root=d_cfg["root"],
        split="val",
        image_size=d_cfg["image_size"],
        splits_dir=d_cfg["splits_dir"],
        transforms=None,
        subset_size=4,
    )
    with torch.no_grad():
        sample_imgs = []
        sample_masks = []
        sample_probs = []
        for i in range(len(val_dataset)):
            img, msk = val_dataset[i]
            logits = model(img.unsqueeze(0))
            prob = torch.sigmoid(logits).squeeze().numpy()
            sample_imgs.append(img[0].numpy())
            sample_masks.append(msk.squeeze().numpy())
            sample_probs.append(prob)
    f"prepared {len(sample_imgs)} samples; image shape {sample_imgs[0].shape}"
    return sample_imgs, sample_masks, sample_probs


@app.cell(hide_code=True)
def _(go, make_subplots, np, sample_imgs, sample_masks, sample_probs):
    _n = len(sample_imgs)
    _cols = ["image", "mask", "pred prob", "pred (>=0.5)"]
    _fig = make_subplots(
        rows=_n,
        cols=4,
        column_titles=_cols,
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )
    for _i in range(_n):
        _panels = [
            sample_imgs[_i],
            sample_masks[_i],
            sample_probs[_i],
            (sample_probs[_i] >= 0.5).astype(np.float32),
        ]
        for _j, _panel in enumerate(_panels):
            _fig.add_trace(
                go.Heatmap(
                    z=np.flipud(_panel),
                    colorscale="gray",
                    showscale=False,
                    zmin=0.0,
                    zmax=1.0,
                ),
                row=_i + 1,
                col=_j + 1,
            )

    for _ax in _fig.layout.annotations:
        _ax.font = dict(size=12)
    _fig.update_xaxes(visible=False)
    _fig.update_yaxes(visible=False, scaleanchor=None)
    _fig.update_layout(
        height=220 * _n,
        width=920,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
