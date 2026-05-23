"""Per-op profiling harness for FANet.

Measures forward (and optionally backward) wall-clock time at three
granularities -- whole pass, per-FAM, per-FAM-internal (fft / branches /
ifft / final) -- over a sweep of (image_size, batch_size) configs.

The locked measurement protocol: 20 warm-up + 100 measured iterations,
`torch.cuda.synchronize()` before each timer batch, median + IQR reported.

See docs/superpowers/specs/2026-05-22-fam-profiling-design.md for the design.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from spectrafan.configs import load_raw_config
from spectrafan.fam import ConvKind, FAMComplex


@dataclass(frozen=True)
class ProfileSweepEntry:
    image_size: int
    batch_size: int


@dataclass
class ProfileConfig:
    warmup_iters: int = 20
    measure_iters: int = 100
    include_chrome_trace: bool = True
    configs: list[ProfileSweepEntry] = field(default_factory=list)
    # Inherited from default.yaml `model:` block, used to construct FANet:
    model_channels: tuple[int, ...] = (64, 128, 256, 512)
    model_bottleneck: int = 1024
    model_fam_conv_kind: ConvKind = "depthwise"


def load_profile_config(path: Path, overrides: list[str] | None = None) -> ProfileConfig:
    raw = load_raw_config(path, overrides=overrides)
    profile_raw = raw.get("profile", {}) or {}
    model_raw = raw.get("model", {}) or {}
    sweep = [
        ProfileSweepEntry(image_size=e["image_size"], batch_size=e["batch_size"])
        for e in profile_raw.get("configs", [])
    ]
    return ProfileConfig(
        warmup_iters=profile_raw.get("warmup_iters", ProfileConfig.warmup_iters),
        measure_iters=profile_raw.get("measure_iters", ProfileConfig.measure_iters),
        include_chrome_trace=profile_raw.get(
            "include_chrome_trace", ProfileConfig.include_chrome_trace
        ),
        configs=sweep,
        model_channels=tuple(model_raw.get("channels", ProfileConfig.model_channels)),
        model_bottleneck=model_raw.get("bottleneck", ProfileConfig.model_bottleneck),
        model_fam_conv_kind=model_raw.get("fam_conv_kind", ProfileConfig.model_fam_conv_kind),
    )


class Timer:
    """Context manager that times a code block on the given device.

    On CUDA, uses ``torch.cuda.Event(enable_timing=True)`` pairs and forces a
    ``torch.cuda.synchronize()`` on exit so ``elapsed_us`` reflects completed
    work, not queued work. On CPU/MPS, falls back to ``time.perf_counter_ns``.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.elapsed_us: float = 0.0
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._t0_ns: int | None = None

    def __enter__(self) -> Timer:
        if self.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._t0_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.device.type == "cuda":
            assert self._start_event is not None and self._end_event is not None
            self._end_event.record()
            torch.cuda.synchronize()
            if exc_type is None:
                self.elapsed_us = self._start_event.elapsed_time(self._end_event) * 1000.0
        else:
            assert self._t0_ns is not None
            if exc_type is None:
                self.elapsed_us = (time.perf_counter_ns() - self._t0_ns) / 1000.0


class FAMProfiled(nn.Module):
    """Wraps a FAMComplex, timing fft / branches / ifft / final separately.

    Mirrors FAMComplex.forward exactly. After each forward, ``self.last_timings``
    holds the four per-block elapsed times in microseconds.
    """

    def __init__(self, fam: FAMComplex, device: torch.device) -> None:
        super().__init__()
        self.fam = fam
        self.device = device
        self.last_timings: dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            with Timer(self.device) as t_fft:
                freq = torch.fft.fft2(x_fp32)
            with Timer(self.device) as t_branches:
                r_prime = self.fam.branch_real(freq.real)
                i_prime = self.fam.branch_imag(freq.imag)
            with Timer(self.device) as t_ifft:
                freq_hat = torch.complex(r_prime, i_prime)
                spatial_hat = torch.fft.ifft2(freq_hat).real
            with Timer(self.device) as t_final:
                out = self.fam.final(x_fp32 + spatial_hat)
        self.last_timings = {
            "fft": t_fft.elapsed_us,
            "branches": t_branches.elapsed_us,
            "ifft": t_ifft.elapsed_us,
            "final": t_final.elapsed_us,
        }
        return out.to(orig_dtype)
