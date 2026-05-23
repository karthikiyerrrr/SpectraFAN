"""Tests for spectrafan.profile."""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import pytest
import torch

from spectrafan.fam import FAMComplex
from spectrafan.profile import (
    FAMProfiled,
    ProfileConfig,
    ProfileSweepEntry,
    Timer,
    compute_summary,
    load_profile_config,
    profile_one_config,
    write_env_json,
)

EXPECTED_CATEGORIES_FWD = {
    "total_fwd",
    "fams_total",
    "fam",
    "fft",
    "branches",
    "ifft",
    "final",
    "encoder",
    "decoder",
    "head",
    "other",
}


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


def test_profile_one_config_schema(tmp_path: Path) -> None:
    df = profile_one_config(
        image_size=32,
        batch_size=1,
        channels=(8, 16),
        bottleneck=32,
        fam_conv_kind="depthwise",
        warmup_iters=2,
        measure_iters=3,
        device=torch.device("cpu"),
        seed=0,
        include_backward=False,
    )

    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) >= {
        "image_size",
        "batch_size",
        "pass",
        "category",
        "module_path",
        "spatial_hw",
        "channels",
        "n_iters",
        "mean_us",
        "median_us",
        "p95_us",
        "iqr_us",
        "std_us",
    }
    # Closed vocabulary.
    assert set(df["category"].unique()) <= EXPECTED_CATEGORIES_FWD
    # All rows share the sweep config.
    assert df["image_size"].unique().to_list() == [32]
    assert df["batch_size"].unique().to_list() == [1]
    assert df["pass"].unique().to_list() == ["fwd"]


def test_profile_one_config_row_count() -> None:
    df = profile_one_config(
        image_size=32,
        batch_size=1,
        channels=(8, 16),  # 2 encoder scales => 2 FAMs
        bottleneck=32,
        fam_conv_kind="depthwise",
        warmup_iters=2,
        measure_iters=3,
        device=torch.device("cpu"),
        seed=0,
        include_backward=False,
    )

    # Formula for n_skips = len(channels):
    # total_fwd(1) + fams_total(1) + fam(n) + (fft+branches+ifft+final)*n
    # + encoder(1) + decoder(1) + head(1) + other(1)
    n_skips = 2
    expected_rows = 1 + 1 + n_skips + 4 * n_skips + 3 + 1
    assert df.height == expected_rows


def test_profile_one_config_sum_to_total_invariant() -> None:
    df = profile_one_config(
        image_size=32,
        batch_size=1,
        channels=(8, 16),
        bottleneck=32,
        fam_conv_kind="depthwise",
        warmup_iters=5,
        measure_iters=200,
        device=torch.device("cpu"),
        seed=0,
        include_backward=False,
    )

    by_cat = {row["category"]: row["median_us"] for row in df.iter_rows(named=True)}
    total = by_cat["total_fwd"]
    parts = (
        by_cat["fams_total"]
        + by_cat["encoder"]
        + by_cat["decoder"]
        + by_cat["head"]
        + by_cat["other"]
    )
    tol = max(0.01 * total, 50.0)
    assert abs(total - parts) <= tol, f"sum-to-total broke: total={total} parts={parts} tol={tol}"


def test_profile_one_config_per_fam_decomposition_invariant() -> None:
    df = profile_one_config(
        image_size=32,
        batch_size=1,
        channels=(8, 16),
        bottleneck=32,
        fam_conv_kind="depthwise",
        warmup_iters=5,
        measure_iters=200,
        device=torch.device("cpu"),
        seed=0,
        include_backward=False,
    )

    fam_rows = df.filter(pl.col("category") == "fam")
    for row in fam_rows.iter_rows(named=True):
        mp = row["module_path"]
        internals = df.filter(
            (pl.col("module_path") == mp)
            & pl.col("category").is_in(["fft", "branches", "ifft", "final"])
        )
        assert internals.height == 4
        parts = internals["median_us"].sum()
        tol = max(0.01 * row["median_us"], 50.0)
        assert abs(row["median_us"] - parts) <= tol, (
            f"FAM decomposition broke for {mp}: total={row['median_us']} parts={parts}"
        )


def test_compute_summary_picks_transform_when_fft_dominates(tmp_path: Path) -> None:
    df = pl.DataFrame(
        [
            {
                "image_size": 256,
                "batch_size": 4,
                "pass": "fwd",
                "category": "total_fwd",
                "module_path": None,
                "spatial_hw": None,
                "channels": None,
                "n_iters": 100,
                "mean_us": 1000.0,
                "median_us": 1000.0,
                "p95_us": 0,
                "iqr_us": 0,
                "std_us": 0,
            },
            {
                "image_size": 256,
                "batch_size": 4,
                "pass": "fwd",
                "category": "fams_total",
                "module_path": None,
                "spatial_hw": None,
                "channels": None,
                "n_iters": 100,
                "mean_us": 700.0,
                "median_us": 700.0,
                "p95_us": 0,
                "iqr_us": 0,
                "std_us": 0,
            },
            {
                "image_size": 256,
                "batch_size": 4,
                "pass": "fwd",
                "category": "fft",
                "module_path": "fams.0",
                "spatial_hw": 128,
                "channels": 64,
                "n_iters": 100,
                "mean_us": 300.0,
                "median_us": 300.0,
                "p95_us": 0,
                "iqr_us": 0,
                "std_us": 0,
            },
            {
                "image_size": 256,
                "batch_size": 4,
                "pass": "fwd",
                "category": "ifft",
                "module_path": "fams.0",
                "spatial_hw": 128,
                "channels": 64,
                "n_iters": 100,
                "mean_us": 200.0,
                "median_us": 200.0,
                "p95_us": 0,
                "iqr_us": 0,
                "std_us": 0,
            },
            {
                "image_size": 256,
                "batch_size": 4,
                "pass": "fwd",
                "category": "branches",
                "module_path": "fams.0",
                "spatial_hw": 128,
                "channels": 64,
                "n_iters": 100,
                "mean_us": 150.0,
                "median_us": 150.0,
                "p95_us": 0,
                "iqr_us": 0,
                "std_us": 0,
            },
            {
                "image_size": 256,
                "batch_size": 4,
                "pass": "fwd",
                "category": "final",
                "module_path": "fams.0",
                "spatial_hw": 128,
                "channels": 64,
                "n_iters": 100,
                "mean_us": 50.0,
                "median_us": 50.0,
                "p95_us": 0,
                "iqr_us": 0,
                "std_us": 0,
            },
        ]
    )

    s = compute_summary(df)

    # transform = (fft + ifft) / fams_total = 500 / 700 = 71.4%
    # branches  = 150 / 700 = 21.4%   -> 50 pp gap -> transform-bound.
    assert s["verdict"] == "transform"
    assert s["fams_pct_of_fwd"] == pytest.approx(70.0, abs=0.1)
    assert s["transform_pct_of_fams"] == pytest.approx(500 / 700 * 100, abs=0.1)


def test_write_env_json_includes_torch_and_device(tmp_path: Path) -> None:
    out = tmp_path / "env.json"
    write_env_json(out, device=torch.device("cpu"))

    data = json.loads(out.read_text())
    assert "torch_version" in data
    assert "cuda_available" in data
    assert data["device"] == "cpu"


def test_write_env_json_marks_dirty_false_in_clean_repo(tmp_path: Path, monkeypatch) -> None:
    """A clean git working tree should write git_dirty=False, not None."""
    import subprocess as _subproc

    real_check_output = _subproc.check_output

    def fake_check_output(cmd, stderr=None):
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return b""  # clean repo
        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return b"abcdef1234567890abcdef1234567890abcdef12\n"
        return real_check_output(cmd, stderr=stderr)

    monkeypatch.setattr(_subproc, "check_output", fake_check_output)

    out = tmp_path / "env.json"
    write_env_json(out, device=torch.device("cpu"))

    data = json.loads(out.read_text())
    assert data["git_dirty"] is False, f"expected False for clean repo, got {data['git_dirty']!r}"
    assert data["git_sha"] == "abcdef1234567890abcdef1234567890abcdef12"


def test_compute_summary_raises_on_empty_canonical_filter() -> None:
    """Non-rectangular sweep where max(image_size) & max(batch_size) don't intersect."""
    df = pl.DataFrame(
        [
            {
                "image_size": 256,
                "batch_size": 32,
                "pass": "fwd",
                "category": "total_fwd",
                "module_path": None,
                "spatial_hw": None,
                "channels": None,
                "n_iters": 100,
                "mean_us": 1.0,
                "median_us": 1.0,
                "p95_us": 0,
                "iqr_us": 0,
                "std_us": 0,
            },
            {
                "image_size": 512,
                "batch_size": 4,
                "pass": "fwd",
                "category": "total_fwd",
                "module_path": None,
                "spatial_hw": None,
                "channels": None,
                "n_iters": 100,
                "mean_us": 1.0,
                "median_us": 1.0,
                "p95_us": 0,
                "iqr_us": 0,
                "std_us": 0,
            },
        ]
    )
    # max image_size = 512, max batch_size = 32, but no row has both.
    with pytest.raises(ValueError, match="rectangular"):
        compute_summary(df)
