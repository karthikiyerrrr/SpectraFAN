"""Tests for spectrafan.profile."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from spectrafan.fam import FAMComplex
from spectrafan.profile import (
    FAMProfiled,
    ProfileConfig,
    ProfileSweepEntry,
    Timer,
    load_profile_config,
)


def test_load_profile_config_reads_default_file() -> None:
    cfg = load_profile_config(Path("configs/profile.yaml"))

    assert isinstance(cfg, ProfileConfig)
    assert cfg.warmup_iters == 20
    assert cfg.measure_iters == 100
    assert cfg.include_chrome_trace is True
    assert cfg.configs == [
        ProfileSweepEntry(image_size=256, batch_size=4),
        ProfileSweepEntry(image_size=256, batch_size=16),
    ]
    assert cfg.model_channels == (64, 128, 256, 512)
    assert cfg.model_bottleneck == 1024
    assert cfg.model_fam_conv_kind == "depthwise"


def test_load_profile_config_applies_overrides() -> None:
    cfg = load_profile_config(
        Path("configs/profile.yaml"),
        overrides=["profile.warmup_iters=3", "profile.measure_iters=5"],
    )
    assert cfg.warmup_iters == 3
    assert cfg.measure_iters == 5


def test_timer_cpu_measures_positive_elapsed() -> None:
    device = torch.device("cpu")
    with Timer(device) as t:
        time.sleep(0.005)
    assert t.elapsed_us > 1000  # at least 1 ms


def test_timer_cpu_independent_instances_all_positive() -> None:
    device = torch.device("cpu")
    samples: list[float] = []
    for _ in range(3):
        with Timer(device) as t:
            time.sleep(0.001)
        samples.append(t.elapsed_us)
    assert len(samples) == 3
    assert all(s > 0 for s in samples)


def test_timer_cpu_does_not_record_elapsed_when_block_raises() -> None:
    """If the timed block raises, elapsed_us must stay at its initial 0.0."""
    device = torch.device("cpu")
    t = Timer(device)
    try:
        with t:
            time.sleep(0.001)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert t.elapsed_us == 0.0


def test_famprofiled_forward_matches_unwrapped_output() -> None:
    torch.manual_seed(0)
    fam = FAMComplex(channels=8, conv_kind="depthwise")
    wrapped = FAMProfiled(fam, device=torch.device("cpu"))

    x = torch.randn(2, 8, 16, 16)
    expected = fam(x)
    actual = wrapped(x)

    torch.testing.assert_close(actual, expected)


def test_famprofiled_records_internal_timings() -> None:
    torch.manual_seed(0)
    fam = FAMComplex(channels=8, conv_kind="depthwise")
    wrapped = FAMProfiled(fam, device=torch.device("cpu"))

    _ = wrapped(torch.randn(2, 8, 16, 16))

    assert set(wrapped.last_timings.keys()) == {"fft", "branches", "ifft", "final"}
    assert all(v > 0 for v in wrapped.last_timings.values())
