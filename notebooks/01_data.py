import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl

    from spectrafan.data import list_pairs, load_pair

    DATA_ROOT = Path("data/raw/temimagenet")
    return DATA_ROOT, Path, go, list_pairs, load_pair, mo, np, pl, px


@app.cell
def _(DATA_ROOT, list_pairs):
    df = list_pairs(DATA_ROOT)
    df.head(10)
    return (df,)


@app.cell
def _(df, pl):
    summary = df.select(
        pairs=pl.len(),
        image_bytes_min=pl.col("image_bytes").min(),
        image_bytes_median=pl.col("image_bytes").median(),
        image_bytes_max=pl.col("image_bytes").max(),
        mask_bytes_median=pl.col("mask_bytes").median(),
    )
    summary
    return


@app.cell
def _(df, go):
    go.Figure(
        data=[go.Histogram(x=df["image_bytes"].to_numpy(), nbinsx=60)],
    ).update_layout(
        title="Image file-size distribution (bytes)",
        xaxis_title="image_bytes",
        yaxis_title="count",
        bargap=0.02,
    )
    return


@app.cell(hide_code=True)
def _(df, mo):
    sample_idx = mo.ui.slider(0, len(df) - 1, value=0, label="Sample index")
    sample_idx
    return (sample_idx,)


@app.cell
def _(Path, df, load_pair, np, px, sample_idx):
    row = df.row(sample_idx.value, named=True)
    image, mask = load_pair(Path(row["image_path"]), Path(row["mask_path"]))
    coverage = float(mask.sum()) / mask.size

    stem = row["stem"]
    title_pair = f"stem={stem}  |  mask coverage={coverage:.3%}"

    fig_pair = px.imshow(
        np.stack([image, mask.astype(np.float32)], axis=0),
        facet_col=0,
        facet_col_wrap=2,
        color_continuous_scale="gray",
        title=title_pair,
    )
    fig_pair.for_each_annotation(
        lambda a: a.update(text={"facet_col=0": "image", "facet_col=1": "mask"}.get(a.text, a.text))
    )
    fig_pair
    return image, stem


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "Crystal lattices have **sparse** frequency spectra. The bright spots in "
        "`log |FFT|` are the lattice periodicities; almost all the signal lives in "
        "a few discrete locations set by the crystal's reciprocal vectors. This is "
        "the structural prior we'll inject in Phase 2."
    )
    return


@app.cell
def _(image, mo, np, px, stem):
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))

    fig_img = px.imshow(image, color_continuous_scale="gray", title=f"image  ({stem})")
    fig_spec = px.imshow(spectrum, color_continuous_scale="gray", title="log(1 + |FFT|)")
    mo.hstack([fig_img, fig_spec], justify="start", widths="equal")
    return


if __name__ == "__main__":
    app.run()
