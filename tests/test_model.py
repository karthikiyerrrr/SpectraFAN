"""Tests for FAMComplex and FANet — architecture and calibration."""

from __future__ import annotations

from itertools import product

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from spectrafan.fam import ConvKind, FAMComplex
from spectrafan.unet import FANet

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


def test_fanet_shape() -> None:
    """FANet maps (B, 3, 512, 512) to (B, 1, 512, 512) with values in [0, 1]."""
    torch.manual_seed(0)
    model = FANet()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 512, 512)
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    assert (y >= 0).all() and (y <= 1).all()


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
    """At least one (conv_kind, bottleneck) combo must match Table 1 within 5%.

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
        ok = _within(params_m * 1e6, TARGET_PARAMS, TOLERANCE) and _within(
            gflops, TARGET_GFLOPS, TOLERANCE
        )
        results.append((conv_kind, bottleneck, params_m, gflops, ok))

    report = "\n".join(
        f"  conv_kind={ck:<22s} bottleneck={bn:<4d} params={pm:6.2f}M gflops={gf:7.2f}  match={ok}"
        for ck, bn, pm, gf, ok in results
    )
    print("\nFANet calibration vs Table 1 (target 31.77 M params, 57.15 GFLOPs):")
    print(report)

    assert any(ok for *_, ok in results), (
        "No (conv_kind, bottleneck) combination matched Table 1 within "
        f"+/- {TOLERANCE * 100:.0f}%. Results:\n{report}"
    )


@pytest.mark.xfail(reason="winner not yet locked into configs/default.yaml", strict=True)
def test_fanet_calibration_locked() -> None:
    """Pin the chosen default config to Table 1 within 5%.

    This test is flipped from xfail to a hard assert in Task 7, once the
    discovery test surfaces the winning combination.
    """
    # Defaults updated in Task 7 — placeholders below match the FAMComplex /
    # FANet constructor defaults until then.
    locked_conv_kind: ConvKind = "depthwise"
    locked_bottleneck = 1024

    x = torch.zeros(1, 3, 512, 512)
    model = FANet(conv_kind=locked_conv_kind, bottleneck=locked_bottleneck)
    model.eval()
    params = _count_params(model)
    gflops = _count_gflops(model, x)
    assert _within(params, TARGET_PARAMS, TOLERANCE), (
        f"params {params / 1e6:.2f}M off target {TARGET_PARAMS / 1e6:.2f}M"
    )
    assert _within(gflops, TARGET_GFLOPS, TOLERANCE), (
        f"gflops {gflops:.2f} off target {TARGET_GFLOPS:.2f}"
    )
