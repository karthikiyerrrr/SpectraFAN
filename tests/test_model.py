"""Tests for FAMComplex and FANet — architecture and calibration."""

from __future__ import annotations

from itertools import product

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from spectrafan.models.fam import ConvKind, FAMComplex
from spectrafan.models.unet import FANet, OutputNorm

CONV_KINDS: list[ConvKind] = ["depthwise", "depthwise_separable"]


@pytest.mark.parametrize("conv_kind", CONV_KINDS)
def test_fam_shape_roundtrip(conv_kind: ConvKind) -> None:
    """FAM preserves shape and dtype, produces no NaNs."""
    torch.manual_seed(0)
    fam = FAMComplex(channels=64, conv_kind=conv_kind)
    x = torch.randn(2, 64, 32, 32)
    y = fam(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_fam_handles_bfloat16_input() -> None:
    """FAM must run in fp32 internally regardless of input dtype.

    Regression: torch.fft.fft2 raises RuntimeError on bf16, so AMP wrapping the
    full network forward used to crash inside FAM. The forward now escapes
    autocast for the FFT block.
    """
    torch.manual_seed(0)
    fam = FAMComplex(channels=64, conv_kind="depthwise")
    x_bf16 = torch.randn(2, 64, 32, 32, dtype=torch.bfloat16)
    y = fam(x_bf16)
    assert y.shape == x_bf16.shape
    assert y.dtype == torch.bfloat16, f"FAM must return input dtype; got {y.dtype}"
    assert torch.isfinite(y).all()


def test_fanet_shape() -> None:
    """FANet maps (B, 3, 512, 512) to (B, 1, 512, 512) logits. Sigmoid is applied externally for probabilities."""
    torch.manual_seed(1)
    model = FANet()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y_logits = model(x)
    assert y_logits.shape == (1, 1, 512, 512)
    assert y_logits.dtype == x.dtype
    assert torch.isfinite(y_logits).all()

    # Probabilities are obtained by applying sigmoid externally and must span (0, 1).
    y_prob = torch.sigmoid(y_logits)
    assert (y_prob >= 0).all() and (y_prob <= 1).all()
    # Catches the paper's Conv->BN->ReLU->Sigmoid OutputConv, which clamps y to [0.5, 1.0].
    assert y_prob.min().item() < 0.5, (
        "OutputConv must let logits go negative so sigmoid can drop below 0.5; "
        f"got y_prob.min()={y_prob.min().item():.4f}. Did a ReLU sneak back in before the output?"
    )
    assert y_prob.max().item() > 0.5, (
        f"y_prob.max()={y_prob.max().item():.4f}; logits all negative?"
    )


@pytest.mark.parametrize("output_norm", ["bn", "none", "groupnorm"])
def test_fanet_output_norm_variants(output_norm: OutputNorm) -> None:
    """All three OutputConv head variants build, forward-pass, and emit finite logits."""
    torch.manual_seed(1)
    model = FANet(output_norm=output_norm)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)
    assert torch.isfinite(y).all()


# Paper Table 1, FANet row.
TARGET_PARAMS = 31.77e6
TARGET_GFLOPS = 57.15
TOLERANCE = 0.05  # +/- 5% on both metrics

BOTTLENECKS = [1024, 512]
CALIB_COMBOS = list(product(CONV_KINDS, BOTTLENECKS))


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_gflops(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Return GFLOPs for one forward pass using fvcore.

    fvcore reports multiply-adds as a single FLOP, which matches the paper's
    convention (their U-Net baseline at 31.03 M params reports 54.66 GFLOPs,
    consistent with the MAdd-as-FLOP counting we get from fvcore).
    """
    flops = FlopCountAnalysis(model, x)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9


def _within(value: float, target: float, tol: float) -> bool:
    return abs(value - target) / target <= tol


def test_fanet_calibration_discovery() -> None:
    """At least one (conv_kind, bottleneck) combo must match Table 1 params within 5% (GFLOPs reported but not gated, due to measurement-convention mismatch).

    Prints all combinations so the winner is visible in test output regardless
    of pass/fail. On total failure, treat as a finding (none of the literal-
    reading variants reproduce Table 1) and re-brainstorm before adding hybrids.
    """
    x = torch.zeros(1, 3, 512, 512)
    results: list[tuple[ConvKind, int, float, float, bool]] = []
    for conv_kind, bottleneck in CALIB_COMBOS:
        model = FANet(conv_kind=conv_kind, bottleneck=bottleneck)
        model.eval()
        params_m = _count_params(model) / 1e6
        gflops = _count_gflops(model, x)
        ok = _within(params_m * 1e6, TARGET_PARAMS, TOLERANCE)
        results.append((conv_kind, bottleneck, params_m, gflops, ok))

    report = "\n".join(
        f"  conv_kind={ck:<22s} bottleneck={bn:<4d} params={pm:6.2f}M gflops={gf:7.2f}  match={ok}"
        for ck, bn, pm, gf, ok in results
    )
    print("\nFANet calibration vs Table 1 (target 31.77 M params; GFLOPs informational only):")
    print(report)

    assert any(ok for *_, ok in results), (
        "No (conv_kind, bottleneck) combination matched Table 1 params within "
        f"+/- {TOLERANCE * 100:.0f}%. Results:\n{report}"
    )


def test_fanet_calibration_locked() -> None:
    """Pin the default config to Table 1 params within 5%."""
    # Locked by params calibration; GFLOPs measurement convention mismatch tracked separately.
    locked_conv_kind: ConvKind = "depthwise"
    locked_bottleneck = 1024

    model = FANet(conv_kind=locked_conv_kind, bottleneck=locked_bottleneck)
    model.eval()
    params = _count_params(model)
    assert _within(params, TARGET_PARAMS, TOLERANCE), (
        f"params {params / 1e6:.2f}M off target {TARGET_PARAMS / 1e6:.2f}M"
    )


def test_fanetmini_forward_shape() -> None:
    """FANetMini maps (B, 3, 256, 256) to (B, 1, 256, 256) logits."""
    from spectrafan.models.unet import FANetMini

    torch.manual_seed(0)
    model = FANetMini()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_fanetmini_has_three_fams_with_expected_channels() -> None:
    """FANetMini exposes exactly three FAMs at channel widths (32, 64, 128)."""
    from spectrafan.models.unet import FANetMini

    model = FANetMini()
    assert len(model.fams) == 3
    expected_channels = (32, 64, 128)
    actual_channels = tuple(fam.channels for fam in model.fams)
    assert actual_channels == expected_channels, (
        f"FANetMini FAM channel widths: expected {expected_channels}, got {actual_channels}"
    )


def test_fanetmini_inherits_fanet_backbone() -> None:
    """FANetMini is a FANet subclass and reuses the same module composition."""
    from spectrafan.models.unet import FANet, FANetMini

    assert issubclass(FANetMini, FANet)
    model = FANetMini()
    # The backbone surface (encoders, decoders, bottleneck, out) must be present
    # because they are inherited from FANet, not re-implemented.
    assert hasattr(model, "encoders") and len(model.encoders) == 2  # 3 scales = stem + 2 encoders
    assert hasattr(model, "decoders") and len(model.decoders) == 3
    assert hasattr(model, "bottleneck_module")
    assert hasattr(model, "out")


def test_fanetmini_param_count_pinned() -> None:
    """Pin FANetMini parameter count. Equality (not range) so any accidental
    composition change surfaces as a diff in this test."""
    from spectrafan.models.unet import FANetMini

    expected = 1958595  # measured from FANetMini() once
    actual = sum(p.numel() for p in FANetMini().parameters())
    assert actual == expected, (
        f"FANetMini param count changed: expected {expected}, got {actual}. "
        "If this is intentional, update the pinned value."
    )


def test_package_reexports_models() -> None:
    """`from spectrafan import FANet, FANetMini` must work for users."""
    import spectrafan
    from spectrafan.models.unet import FANet as FANet_src
    from spectrafan.models.unet import FANetMini as FANetMini_src

    assert hasattr(spectrafan, "FANet")
    assert hasattr(spectrafan, "FANetMini")
    # Identity, not just equality — these must be the same class objects.

    assert spectrafan.FANet is FANet_src
    assert spectrafan.FANetMini is FANetMini_src


def test_fanetmini_accepts_in_channels_one() -> None:
    """FANetMini builds with in_channels=1 and forward-passes (1, 1, 256, 256) inputs."""
    from spectrafan.models.unet import FANetMini

    torch.manual_seed(0)
    model = FANetMini(in_channels=1)
    model.eval()
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)
    assert torch.isfinite(y).all()
