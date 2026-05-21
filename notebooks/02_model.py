import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import plotly.express as px
    import polars as pl
    import torch
    from fvcore.nn import FlopCountAnalysis

    from spectrafan.fam import FAMComplex
    from spectrafan.unet import FANet

    torch.manual_seed(0)
    _ = torch.set_grad_enabled(False)



@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # FANet model walkthrough

    A walk through the FANet architecture from Liu et al.,
    *Frequency domain attention network for S/TEM image segmentation*,
    **Materials Today Nano** 33 (2026), 100772.

    FANet is a U-Net with one **Frequency Attention Module (FAM)** sitting on
    each of the four skip connections. Each FAM transforms the encoder's
    feature map into the frequency domain, learns a filter there, and
    transforms back — so the decoder sees a frequency-aware version of the
    skip instead of the raw encoder output.

    This notebook traces **what happens to a tensor** as it moves through the
    model:

    1. Why frequency-domain attention is well suited to crystalline imagery.
    2. What the FAM does to a feature map, stage by stage.
    3. How the frequency content of a map actually changes after FAM filtering.
    4. Where compute happens across the four skip scales.
    5. How the four FAM-filtered skips flow through the encoder / decoder.
    """)

    return


@app.cell(hide_code=True)
def motivation_md():
    mo.md(r"""
    ## Why frequency-domain attention?

    A crystal's atomic lattice is periodic. Its Fourier magnitude spectrum is
    a small set of sharp peaks at positions set by the lattice's reciprocal
    vectors — literally the diffraction pattern. A spatial 3×3 convolution
    has to relearn that structure from local neighborhoods over and over;
    an attention module that operates in frequency space can address those
    peaks directly, with a global receptive field.

    The synthetic lattice below makes the argument visible. Two interfering
    sinusoids in real space produce four sharp peaks in the magnitude spectrum
    (counting the conjugate pair). For real STEM/TEM imagery the same
    structure holds, just with more peaks and more noise.
    """)

    return


@app.cell
def lattice_demo():
    def make_lattice(size=128, freqs=((6, 0), (0, 8)), phases=(0.0, 0.0)):
        """Sum of cosines on a 2D grid — a toy crystal lattice."""
        y, x = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, size),
            torch.linspace(0, 2 * torch.pi, size),
            indexing="ij",
        )
        img = torch.zeros(size, size)
        for (fy, fx), phi in zip(freqs, phases, strict=True):
            img = img + torch.cos(fy * y + fx * x + phi)
        return img


    lattice = make_lattice()
    lattice_spectrum = torch.fft.fftshift(torch.fft.fft2(lattice)).abs().log1p()


    def _square_fig(arr, title, cmap):
        f = px.imshow(arr, color_continuous_scale=cmap, title=title, aspect="equal")
        f.update_layout(
            coloraxis_showscale=False,
            width=440,
            height=440,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return f


    mo.hstack(
        [
            _square_fig(lattice.numpy(), "real space — toy crystal lattice", "gray"),
            _square_fig(lattice_spectrum.numpy(), "frequency space — log|FFT|", "magma"),
        ],
        justify="start",
        gap=1,
    )

    return (lattice,)


@app.cell(hide_code=True)
def fam_md():
    mo.md(r"""
    ## What FAMComplex does to a feature map

    A FAM doesn't change the shape of its input — `(B, C, H, W)` in,
    `(B, C, H, W)` out. What it changes is the *frequency content*.

    ```
    x ──► fft2 ──► split real/imag ──► per-part branch ──► combine ──► ifft2 ──►
       └──────────────── residual add ────────────────────────────► + ──► conv1×1 ──►
    ```

    Stage by stage:

    - **fft2** turns each `(H, W)` spatial map into a complex spectrum of the
      same shape. Every "pixel" in the result is now a frequency, and its
      value says how much of that frequency is present in the input.
    - **split real / imag** is so that two ordinary real-valued conv branches
      can run on each half — a complex 2D conv would need bespoke kernels.
    - **branch** is the learned frequency filter: two depthwise 3×3 convs
      sliding over the frequency grid, with BatchNorm and ReLU between them.
      It learns to amplify or suppress patterns in the spectrum. The real and
      imaginary branches have *independent* weights.
    - **combine + ifft2** reassembles the filtered spectrum and projects back
      to real space.
    - **residual add** keeps the original feature map alongside the filtered
      contribution, so the FAM behaves as a *correction* on top of the
      encoder's output rather than a replacement.
    - **conv1×1** mixes channels at the end.

    The same forward pass below, run explicitly on a `(1, 4, 8, 8)` input so
    every intermediate stays legible:
    """)

    return


@app.cell
def fam_steps():
    fam_demo = FAMComplex(channels=4, conv_kind="depthwise").eval()

    x_demo = torch.randn(1, 4, 8, 8)
    freq = torch.fft.fft2(x_demo)
    r, i = freq.real, freq.imag
    r_filtered = fam_demo.branch_real(r)
    i_filtered = fam_demo.branch_imag(i)
    freq_hat = torch.complex(r_filtered, i_filtered)
    spatial_hat = torch.fft.ifft2(freq_hat).real
    y_demo = fam_demo.final(x_demo + spatial_hat)


    def _row(stage, t):
        return {
            "stage": stage,
            "shape": str(tuple(t.shape)),
            "dtype": str(t.dtype).removeprefix("torch."),
        }


    steps_df = pl.DataFrame(
        [
            _row("input  x", x_demo),
            _row("fft2(x)  →  freq", freq),
            _row("freq.real  →  r", r),
            _row("freq.imag  →  i", i),
            _row("branch_real(r)  →  r'", r_filtered),
            _row("branch_imag(i)  →  i'", i_filtered),
            _row("complex(r', i')  →  freq_hat", freq_hat),
            _row("ifft2(freq_hat).real  →  spatial_hat", spatial_hat),
            _row("conv1×1(x + spatial_hat)  →  y", y_demo),
        ]
    )
    steps_df

    return


@app.cell(hide_code=True)
def fam_on_lattice_md():
    mo.md(r"""
    ### FAM in action on the lattice

    Push the lattice through a freshly-initialized single-channel FAM and
    compare the spectrum before and after. Even with random filter weights,
    the output spectrum is visibly redistributed — energy spreads, the sharp
    peaks soften. A trained FAM would learn to *concentrate* energy at
    informative frequencies and suppress noise elsewhere.
    """)
    return


@app.cell(hide_code=True)
def fam_on_lattice(lattice):
    # Push the lattice through one FAM and compare frequency spectra.
    fam_single_channel = FAMComplex(channels=1, conv_kind="depthwise").eval()

    lattice_input = lattice[None, None]  # (1, 1, H, W)
    lattice_output = fam_single_channel(lattice_input)

    in_spec = torch.fft.fftshift(torch.fft.fft2(lattice_input[0, 0])).abs().log1p()
    out_spec = torch.fft.fftshift(torch.fft.fft2(lattice_output[0, 0])).abs().log1p()


    def _spec_fig(arr, title):
        f = px.imshow(arr, color_continuous_scale="magma", title=title, aspect="equal")
        f.update_layout(
            coloraxis_showscale=False,
            width=440,
            height=440,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return f


    mo.hstack(
        [
            _spec_fig(in_spec.numpy(), "input spectrum (log|FFT|)"),
            _spec_fig(out_spec.numpy(), "output spectrum (log|FFT|)"),
        ],
        justify="start",
        gap=1,
    )

    return


@app.cell(hide_code=True)
def per_scale_md():
    mo.md(r"""
    ## Where the compute happens

    The four FAMs run on four different skip tensors and have very uneven
    costs. The high-resolution, low-channel skip (`64×512×512`) dominates,
    because the FFT cost scales as `O(N²·log N²)` with spatial dimensions
    while the spectrum still has one frequency cell per spatial pixel for the
    depthwise convs to process.
    """)

    return


@app.cell
def per_scale_table():
    skip_specs = [
        (64, 512),   # stem output
        (128, 256),  # encoder 1
        (256, 128),  # encoder 2
        (512, 64),   # encoder 3
    ]


    def _fam_costs():
        rows = []
        for channels, spatial in skip_specs:
            fam = FAMComplex(channels=channels, conv_kind="depthwise").eval()
            skip_tensor = torch.zeros(1, channels, spatial, spatial)
            params = sum(p.numel() for p in fam.parameters())
            fc = FlopCountAnalysis(fam, skip_tensor)
            fc.unsupported_ops_warnings(False)
            fc.uncalled_modules_warnings(False)
            rows.append({
                "skip": f"FAM({channels})",
                "input shape": f"(1, {channels}, {spatial}, {spatial})",
                "params (K)": round(params / 1e3, 1),
                "GFLOPs @ 512² input": round(fc.total() / 1e9, 3),
            })
        return rows


    per_skip_df = pl.DataFrame(_fam_costs())
    per_skip_df

    return


@app.cell(hide_code=True)
def wiring_md():
    mo.md(r"""
    ## How the FAM-filtered skips flow through the U-Net

    Each encoder stage tees off a copy of its output into the matching FAM.
    The FAM-filtered version is what the decoder concatenates with its
    upsampled input — the raw encoder output never reaches the decoder. The
    bottleneck has no FAM.

    ```
    input ──► stem(3→64) ─┬──► skip0 ──► FAM(64) ──────────────────────┐
                          ▼                                              │
                  enc0(64→128) ─┬──► skip1 ──► FAM(128) ────────────┐    │
                                ▼                                    │    │
                       enc1(128→256) ─┬──► skip2 ──► FAM(256) ──┐    │    │
                                      ▼                          │    │    │
                            enc2(256→512) ─┬──► skip3 ──► FAM(512) ┐ │    │
                                            ▼                       │ │    │
                                 bottleneck(512→1024)               │ │    │
                                            ▼                       │ │    │
                                 dec0(1024→512) ◄── concat ─────────┘ │    │
                                            ▼                          │    │
                                 dec1(512→256)  ◄── concat ────────────┘    │
                                            ▼                                │
                                 dec2(256→128)  ◄── concat ──────────────────┘ (and FAM(128) above)
                                            ▼
                                 dec3(128→64)
                                            ▼
                                 out(64→1) → sigmoid
    ```

    Below: the actual `(B, C, H, W)` shape of the data at every stage of a
    forward pass on a `(1, 3, 512, 512)` input.
    """)

    return


@app.cell
def shape_walk():
    # Walk the FANet forward pass and record the shape at every stage.
    def _walk_fanet_shapes():
        model = FANet().eval()
        x_input = torch.zeros(1, 3, 512, 512)

        rows = []
        skips_local = []

        h = model.stem(x_input)
        skips_local.append(h)
        rows.append({"stage": "stem", "produces": "skip0", "shape": str(tuple(h.shape))})

        for idx, enc in enumerate(model.encoders):
            h = enc(h)
            skips_local.append(h)
            rows.append({"stage": f"encoders[{idx}]", "produces": f"skip{idx + 1}", "shape": str(tuple(h.shape))})

        h = model.bottleneck_module(h)
        rows.append({"stage": "bottleneck", "produces": "—", "shape": str(tuple(h.shape))})

        filtered_local = [fam(s) for fam, s in zip(model.fams, skips_local, strict=True)]
        for idx, (dec, skip) in enumerate(zip(model.decoders, reversed(filtered_local), strict=True)):
            h = dec(h, skip)
            rows.append({"stage": f"decoders[{idx}]", "produces": "—", "shape": str(tuple(h.shape))})

        h = model.out(h)
        rows.append({"stage": "out", "produces": "y", "shape": str(tuple(h.shape))})

        return rows


    wiring_df = pl.DataFrame(_walk_fanet_shapes())
    wiring_df

    return


if __name__ == "__main__":
    app.run()
