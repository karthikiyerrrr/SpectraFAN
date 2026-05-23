"""Per-op profiling harness for FANet.

Measures forward (and optionally backward) wall-clock time at three
granularities -- whole pass, per-FAM, per-FAM-internal (fft / branches /
ifft / final) -- over a sweep of (image_size, batch_size) configs.

The locked measurement protocol: 20 warm-up + 100 measured iterations,
`torch.cuda.synchronize()` before each timer batch, median + IQR reported.

See docs/superpowers/specs/2026-05-22-fam-profiling-design.md for the design.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from statistics import median, pstdev

import polars as pl
import torch
from torch import nn

from spectrafan.configs import load_raw_config
from spectrafan.fam import ConvKind, FAMComplex
from spectrafan.unet import FANet


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


def _percentile(samples: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 1]); empty list -> 0.0."""
    if not samples:
        return 0.0
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "mean_us": 0.0,
            "median_us": 0.0,
            "p95_us": 0.0,
            "iqr_us": 0.0,
            "std_us": 0.0,
        }
    return {
        "mean_us": sum(samples) / len(samples),
        "median_us": median(samples),
        "p95_us": _percentile(samples, 0.95),
        "iqr_us": _percentile(samples, 0.75) - _percentile(samples, 0.25),
        "std_us": pstdev(samples) if len(samples) > 1 else 0.0,
    }


def _attach_block_timers(model: FANet, device: torch.device) -> tuple[dict[str, list[float]], list]:
    """Install pre/post forward hooks that record per-iteration block timings.

    Returns (samples_by_key, hook_handles). Keys: 'stem', 'enc_i', 'bottleneck',
    'dec_i', 'out'. The bottleneck block is folded into the encoder category by
    the caller (it sits between encoders, not as a skip).
    """
    samples: dict[str, list[float]] = {}
    pending: dict[str, Timer] = {}
    handles = []

    def _make_pair(name: str, module: nn.Module) -> None:
        samples[name] = []

        def pre(_m, _inp):
            t = Timer(device)
            t.__enter__()
            pending[name] = t

        def post(_m, _inp, _out):
            t = pending.pop(name)
            t.__exit__(None, None, None)
            samples[name].append(t.elapsed_us)

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    _make_pair("stem", model.stem)
    for i, enc in enumerate(model.encoders):
        _make_pair(f"enc_{i}", enc)
    _make_pair("bottleneck", model.bottleneck_module)
    for i, dec in enumerate(model.decoders):
        _make_pair(f"dec_{i}", dec)
    _make_pair("out", model.out)
    return samples, handles


def profile_one_config(
    image_size: int,
    batch_size: int,
    channels: tuple[int, ...],
    bottleneck: int,
    fam_conv_kind: ConvKind,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
    seed: int,
    include_backward: bool,
) -> pl.DataFrame:
    """Profile one (image_size, batch_size) configuration.

    Builds a FANet, wraps each FAM with ``FAMProfiled``, attaches forward hooks
    on stem + each encoder + bottleneck + each decoder + out, runs warm-up and
    measured iterations, and returns a polars DataFrame with one row per
    (category, module_path). When ``include_backward=True``, also emits a
    single ``total_bwd`` row.

    The ``encoder`` category includes the stem, all encoder blocks, AND the
    bottleneck module (treated as an opaque non-FAM block per the design).
    The ``other`` category is the residual ``total_fwd - fams_total - encoder
    - decoder - head`` and can go slightly negative on noisy CPUs.
    """
    torch.manual_seed(seed)
    model = FANet(
        in_channels=3,
        channels=channels,
        bottleneck=bottleneck,
        conv_kind=fam_conv_kind,
    ).to(device)
    model.train()

    # Spatial resolution at each skip (stem keeps input HW; each encoder halves it).
    skip_hw = [image_size]
    for _ in channels[1:]:
        skip_hw.append(skip_hw[-1] // 2)

    # Swap FAMs for profiled wrappers.
    profiled_fams = [FAMProfiled(fam, device=device) for fam in model.fams]
    model.fams = nn.ModuleList(profiled_fams)

    block_samples, hook_handles = _attach_block_timers(model, device)
    total_fwd_samples: list[float] = []
    fams_total_samples: list[float] = []
    per_fam_samples: list[list[float]] = [[] for _ in profiled_fams]
    fam_internal_samples: list[dict[str, list[float]]] = [
        {"fft": [], "branches": [], "ifft": [], "final": []} for _ in profiled_fams
    ]

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    total_bwd_samples: list[float] = []
    try:
        # Warm-up (drop these samples).
        for _ in range(warmup_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            _ = model(x)
            for k in block_samples:
                block_samples[k].clear()
            for fam in profiled_fams:
                fam.last_timings.clear()

        # Measurement loop (forward, optionally backward).
        for _ in range(measure_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            if include_backward:
                x_iter = x.clone().requires_grad_(True)
            else:
                x_iter = x
            with Timer(device) as t_total:
                out = model(x_iter)
            total_fwd_samples.append(t_total.elapsed_us)

            if include_backward:
                loss = out.sum()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                with Timer(device) as t_bwd:
                    loss.backward()
                total_bwd_samples.append(t_bwd.elapsed_us)
                model.zero_grad(set_to_none=True)

            # Per-FAM totals = sum of internal timings.
            for idx, fam in enumerate(profiled_fams):
                internals = fam.last_timings
                per_fam_samples[idx].append(sum(internals.values()))
                for k in fam_internal_samples[idx]:
                    fam_internal_samples[idx][k].append(internals[k])
                fam.last_timings = {}

            # fams_total = sum across FAMs in this iteration.
            last_fam_sum = sum(per_fam_samples[i][-1] for i in range(len(profiled_fams)))
            fams_total_samples.append(last_fam_sum)
    finally:
        for h in hook_handles:
            h.remove()

    # Group encoder/decoder/head samples (sum across blocks per iteration).
    n = measure_iters
    encoder_samples = [
        block_samples["stem"][i]
        + sum(block_samples[f"enc_{j}"][i] for j in range(len(model.encoders)))
        + block_samples["bottleneck"][i]
        for i in range(n)
    ]
    decoder_samples = [
        sum(block_samples[f"dec_{j}"][i] for j in range(len(model.decoders))) for i in range(n)
    ]
    head_samples = list(block_samples["out"])
    other_samples = [
        total_fwd_samples[i]
        - fams_total_samples[i]
        - encoder_samples[i]
        - decoder_samples[i]
        - head_samples[i]
        for i in range(n)
    ]

    rows: list[dict] = []

    def _row(
        category: str,
        samples: list[float],
        *,
        module_path: str | None = None,
        spatial_hw: int | None = None,
        ch: int | None = None,
    ) -> None:
        s = _summarize(samples)
        rows.append(
            {
                "image_size": image_size,
                "batch_size": batch_size,
                "pass": "fwd",
                "category": category,
                "module_path": module_path,
                "spatial_hw": spatial_hw,
                "channels": ch,
                "n_iters": measure_iters,
                **s,
            }
        )

    _row("total_fwd", total_fwd_samples)
    _row("fams_total", fams_total_samples)
    for idx, fam_samples in enumerate(per_fam_samples):
        _row(
            "fam",
            fam_samples,
            module_path=f"fams.{idx}",
            spatial_hw=skip_hw[idx],
            ch=channels[idx],
        )
        for k in ("fft", "branches", "ifft", "final"):
            _row(
                k,
                fam_internal_samples[idx][k],
                module_path=f"fams.{idx}",
                spatial_hw=skip_hw[idx],
                ch=channels[idx],
            )
    _row("encoder", encoder_samples)
    _row("decoder", decoder_samples)
    _row("head", head_samples)
    _row("other", other_samples)

    if include_backward and total_bwd_samples:
        s = _summarize(total_bwd_samples)
        rows.append(
            {
                "image_size": image_size,
                "batch_size": batch_size,
                "pass": "bwd",
                "category": "total_bwd",
                "module_path": None,
                "spatial_hw": None,
                "channels": None,
                "n_iters": measure_iters,
                **s,
            }
        )

    return pl.DataFrame(rows)


def compute_summary(df: pl.DataFrame) -> dict:
    """Boil per-row timings down to percentage shares of FAM time.

    If multiple sweep configs are in `df`, picks the largest batch_size at the
    largest image_size as the canonical config for the breakdown (most realistic
    workload).
    """
    canonical = df.filter(
        (pl.col("image_size") == df["image_size"].max())
        & (pl.col("batch_size") == df["batch_size"].max())
    )
    if canonical.is_empty():
        raise ValueError(
            "No rows match max(image_size) AND max(batch_size); "
            "check that the sweep DataFrame is rectangular."
        )
    by_cat = {r["category"]: r["median_us"] for r in canonical.iter_rows(named=True)}
    total = by_cat["total_fwd"]
    fams = by_cat["fams_total"]
    # Sum fft / ifft / branches across all FAMs.
    transform = (
        canonical.filter(pl.col("category").is_in(["fft", "ifft"]))
        .select(pl.col("median_us").sum())
        .item()
    )
    branches = (
        canonical.filter(pl.col("category") == "branches").select(pl.col("median_us").sum()).item()
    )
    fams_pct = 100.0 * fams / total if total > 0 else 0.0
    transform_pct = 100.0 * transform / fams if fams > 0 else 0.0
    branches_pct = 100.0 * branches / fams if fams > 0 else 0.0
    return {
        "image_size": int(canonical["image_size"][0]),
        "batch_size": int(canonical["batch_size"][0]),
        "fams_pct_of_fwd": round(fams_pct, 2),
        "transform_pct_of_fams": round(transform_pct, 2),
        "branches_pct_of_fams": round(branches_pct, 2),
        "fams_total_us": round(fams, 2),
        "total_fwd_us": round(total, 2),
    }


def write_env_json(path: Path, device: torch.device) -> None:
    """Write a JSON snapshot of the runtime environment to `path`."""

    def _safe(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    # Call once and cache to avoid three subprocess invocations for git_dirty.
    git_porcelain = _safe(["git", "status", "--porcelain"])
    data = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "platform": platform.platform(),
        "git_sha": _safe(["git", "rev-parse", "HEAD"]),
        "git_dirty": (len(git_porcelain) > 0) if git_porcelain is not None else None,
        "hostname": socket.gethostname(),
    }
    path.write_text(json.dumps(data, indent=2))


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _maybe_capture_chrome_trace(
    cfg: ProfileConfig,
    out_dir: Path,
    device: torch.device,
) -> None:
    if not cfg.include_chrome_trace or not cfg.configs:
        return
    entry = cfg.configs[0]  # smallest config

    torch.manual_seed(0)
    model = FANet(
        in_channels=3,
        channels=cfg.model_channels,
        bottleneck=cfg.model_bottleneck,
        conv_kind=cfg.model_fam_conv_kind,
    ).to(device)
    model.train()
    x = torch.randn(entry.batch_size, 3, entry.image_size, entry.image_size, device=device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_dir / "chrome_trace.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spectrafan.profile")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = load_profile_config(args.config, overrides=args.override)
    device = _resolve_device(args.device)

    if args.output is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = Path("runs") / f"{ts}_profile"
    else:
        out_dir = args.output
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(
            f"[profile] error: output directory already exists: {out_dir}\n"
            "  use a different --output path or delete the existing directory.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    # Persist resolved config + env.
    copyfile(args.config, out_dir / "config.yaml")
    write_env_json(out_dir / "env.json", device=device)

    # Sweep.
    frames: list[pl.DataFrame] = []
    for entry in cfg.configs:
        print(
            f"[profile] image_size={entry.image_size} batch_size={entry.batch_size}",
            flush=True,
        )
        df = profile_one_config(
            image_size=entry.image_size,
            batch_size=entry.batch_size,
            channels=cfg.model_channels,
            bottleneck=cfg.model_bottleneck,
            fam_conv_kind=cfg.model_fam_conv_kind,
            warmup_iters=cfg.warmup_iters,
            measure_iters=cfg.measure_iters,
            device=device,
            seed=args.seed,
            include_backward=args.backward,
        )
        frames.append(df)

    timings = pl.concat(frames)
    timings.write_parquet(out_dir / "timings.parquet")

    summary = compute_summary(timings)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _maybe_capture_chrome_trace(cfg, out_dir, device)

    print(f"[profile] wrote {out_dir}", flush=True)
    print(
        f"[profile] FAM share of forward: {summary['fams_pct_of_fwd']}% "
        f"— transform {summary['transform_pct_of_fams']}% of FAM, "
        f"branches {summary['branches_pct_of_fams']}% of FAM",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
