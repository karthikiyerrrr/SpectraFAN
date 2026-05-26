"""Post-hoc FAM diagnostic.

Loads <run_dir>/best.pt, runs the val split three times (as_trained,
fam_skip_fft, fam_zero), and writes <run_dir>/fam_stats.parquet plus
<run_dir>/fam_diagnosis.json. See docs/superpowers/specs/2026-05-25-fam-postmortem-design.md.

CLI:
    uv run python -m spectrafan.diagnose_fam --run runs/<id>
    uv run python -m spectrafan.diagnose_fam --latest fanetmini
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType

import torch
from torch.utils.data import DataLoader

from spectrafan.fam import FAMComplex
from spectrafan.metrics import RunningMetrics


def diagnose_run(run_dir: Path) -> None:
    """Placeholder; filled in by later tasks."""
    raise NotImplementedError


def _patched_forward_skip_fft(self: FAMComplex, x: torch.Tensor) -> torch.Tensor:
    """Drop the FFT pathway, keep the learned 1x1 projection.

    Equivalent to running FAMComplex.forward with spatial_hat = 0. Returned
    tensor has the same shape and dtype as x.
    """
    return self.final(x)


def _patched_forward_zero(self: FAMComplex, x: torch.Tensor) -> torch.Tensor:
    """Pure identity — bounds the contribution of the entire FAM block."""
    return x


def _apply_forward(
    model: torch.nn.Module, fn: Callable[[FAMComplex, torch.Tensor], torch.Tensor]
) -> None:
    """Bind `fn` as the .forward of every FAMComplex module in `model`."""
    for module in model.modules():
        if isinstance(module, FAMComplex):
            module.forward = MethodType(fn, module)


def _restore_forward(model: torch.nn.Module) -> None:
    """Restore the original FAMComplex.forward on every FAMComplex in `model`."""
    for module in model.modules():
        if isinstance(module, FAMComplex):
            module.forward = MethodType(FAMComplex.forward, module)


@dataclass
class _FamStatsCollector:
    """Accumulates per-FAM per-batch stats during the as_trained val pass.

    `rows` is a list of dicts (one per (scale_idx, batch_idx) pair) ready
    to be turned into a polars DataFrame.
    `_batch_counters` is private: maps scale_idx -> next batch_idx. The
    instrumented forward bumps it on each call.
    """

    rows: list[dict[str, float]] = field(default_factory=list)
    _batch_counters: dict[int, int] = field(default_factory=dict)

    def next_batch_idx(self, scale_idx: int) -> int:
        idx = self._batch_counters.get(scale_idx, 0)
        self._batch_counters[scale_idx] = idx + 1
        return idx


def _rms(t: torch.Tensor) -> float:
    """Root-mean-square magnitude, scale-free across spatial resolutions."""
    return float(t.detach().float().pow(2).mean().sqrt().item())


def _dead_rate(t: torch.Tensor) -> float:
    """Fraction of activations exactly equal to zero (post-ReLU)."""
    return float((t.detach() == 0).float().mean().item())


def _make_instrumented_forward(
    collector: _FamStatsCollector, scale_idx: int
) -> Callable[[FAMComplex, torch.Tensor], torch.Tensor]:
    """Build a forward function that mirrors FAMComplex.forward but stashes stats.

    The function returned is bound onto a FAMComplex via MethodType in the
    `as_trained` val pass; it computes exactly the same output as the
    original forward.
    """

    def _forward(self: FAMComplex, x: torch.Tensor) -> torch.Tensor:
        # Mirror FAMComplex.forward exactly, with extra capture statements.
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            freq = torch.fft.fft2(x_fp32)

            # FFT real-side spectrum stats — captured before BN sees them.
            freq_real_abs = freq.real.abs().detach().float()
            dc_energy = freq_real_abs[..., 0, 0].pow(2).sum().item()
            total_energy = freq_real_abs.pow(2).sum().item()
            fft_real_dc_share = dc_energy / max(total_energy, 1e-12)
            flat = freq_real_abs.flatten()
            p99 = torch.quantile(flat, 0.99).item()
            median = flat.median().item()
            fft_real_p99_to_median = p99 / max(median, 1e-12)

            # Real and imag branches — capture each branch's output norm.
            # Dead-rate uses the post-ReLU tensor; recover it by stepping
            # through the branch manually so we don't need a hook.
            r_pre = self.branch_real.conv1(freq.real)
            r_bn = self.branch_real.bn(r_pre)
            r_relu = self.branch_real.act(r_bn)
            r_prime = self.branch_real.conv2(r_relu)
            branch_real_dead_rate = _dead_rate(r_relu)

            i_pre = self.branch_imag.conv1(freq.imag)
            i_bn = self.branch_imag.bn(i_pre)
            i_relu = self.branch_imag.act(i_bn)
            i_prime = self.branch_imag.conv2(i_relu)
            branch_imag_dead_rate = _dead_rate(i_relu)

            freq_hat = torch.complex(r_prime, i_prime)
            spatial_hat = torch.fft.ifft2(freq_hat).real
            out = self.final(x_fp32 + spatial_hat)

        collector.rows.append(
            {
                "scale_idx": scale_idx,
                "batch_idx": collector.next_batch_idx(scale_idx),
                "input_norm": _rms(x_fp32),
                "contribution_norm": _rms(spatial_hat),
                "output_norm": _rms(out),
                "branch_real_out_norm": _rms(r_prime),
                "branch_imag_out_norm": _rms(i_prime),
                "branch_real_dead_rate": branch_real_dead_rate,
                "branch_imag_dead_rate": branch_imag_dead_rate,
                "fft_real_dc_share": fft_real_dc_share,
                "fft_real_p99_to_median": fft_real_p99_to_median,
            }
        )
        return out.to(orig_dtype)

    return _forward


def _run_val_once(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """One forward pass over `loader`; returns val_iou as a float."""
    rm = RunningMetrics()
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rm.update(model(xb), yb)
    return float(rm.compute()["iou"])
