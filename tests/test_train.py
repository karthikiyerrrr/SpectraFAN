"""Tests for spectrafan.train."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch
from torch.utils.data import Dataset

from spectrafan.train import (
    AugConfig,
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    TrainConfig,
    fit,
)


class _SyntheticPairs(Dataset):
    """Tiny in-memory dataset: 8 random (3, 32, 32) images + binary masks."""

    def __init__(self, n: int = 8, size: int = 32) -> None:
        torch.manual_seed(0)
        self.images = torch.rand(n, 3, size, size)
        # Mask is a soft threshold on a low-frequency version of the image so the model
        # has something learnable, not pure noise.
        smoothed = torch.nn.functional.avg_pool2d(self.images.mean(1, keepdim=True), 8, 1, 4)
        smoothed = smoothed[..., :size, :size]  # crop back to (size, size) after padding
        self.masks = (smoothed > smoothed.median()).float()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx]


def _tiny_cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            channels=(8, 16, 32, 64),
            bottleneck=128,
            fam_conv_kind="depthwise",
        ),
        data=DataConfig(
            dataset="synthetic",
            image_size=32,
            batch_size=4,
            subset_size=None,
            val_subset_size=None,
            splits_dir=Path("data/splits/temimagenet_v1"),
        ),
        aug=AugConfig(),
        optim=OptimConfig(lr=1e-3),  # bumped from 1e-5 so 3 epochs is enough to move the loss
        train=TrainConfig(
            epochs=3,
            seed=0,
            device="cpu",
            run_root=tmp_path / "runs",
            loss_ce_weight=0.5,
            loss_dice_weight=0.5,
        ),
    )


def test_fit_one_epoch_decreases_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three epochs on a tiny synthetic Dataset; train loss must decrease."""
    ds = _SyntheticPairs()
    # Inject our synthetic dataset by monkey-patching the dataset factory.
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    run_dir = fit(cfg)

    metrics_path = run_dir / "metrics.parquet"
    assert metrics_path.is_file()
    df = pl.read_parquet(metrics_path)
    assert df.height == 3
    assert {"epoch", "lr", "train_loss", "val_loss", "train_iou", "val_iou"} <= set(df.columns)
    train_losses = df["train_loss"].to_list()
    assert train_losses[-1] < train_losses[0], f"train loss did not decrease: {train_losses}"

    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "config.yaml").is_file()


def test_fit_with_amp_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() with amp=True runs end-to-end on CPU (autocast is a no-op there) and produces finite metrics."""
    ds = _SyntheticPairs()
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    cfg.train.amp = True
    run_dir = fit(cfg)

    df = pl.read_parquet(run_dir / "metrics.parquet")
    assert df.height == cfg.train.epochs
    for col in ("train_loss", "val_loss", "train_iou", "val_iou"):
        values = df[col].to_list()
        assert all(v == v for v in values), f"{col} has NaN: {values}"


def _latest_run_dir(root: Path) -> Path:
    if not root.is_dir():
        pytest.skip(f"{root} does not exist; execute the smoke trial first")
    runs = sorted((p for p in root.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not runs:
        pytest.skip(f"no runs in {root}; execute the smoke trial first")
    return runs[-1]


@pytest.mark.slow
def test_smoke_trial_acceptance() -> None:
    """Validate the most recent run in runs/ meets the smoke trial gate.

    Pre-condition: ``uv run python -m spectrafan.train --config configs/smoke.yaml`` has
    been executed at least once. This test does NOT launch training.
    """
    run_dir = _latest_run_dir(Path("runs"))
    metrics_path = run_dir / "metrics.parquet"
    assert metrics_path.is_file(), f"no metrics.parquet under {run_dir}"

    df = pl.read_parquet(metrics_path)
    val_iou = df["val_iou"].to_list()

    assert df.height == 10, f"expected 10 epochs, got {df.height}"
    assert val_iou[2] > val_iou[1] > val_iou[0], (
        f"val_iou not strictly increasing over the first 3 epochs: {val_iou[:3]}"
    )
    assert val_iou[-1] > 0.5, f"final val_iou {val_iou[-1]:.4f} <= 0.5"

    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    # Round-trip the checkpoint to make sure it's loadable.
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt


def test_epoch_checkpoints_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With checkpoint_every=2 and 4 epochs, epoch_002.pt and epoch_004.pt are saved."""
    ds = _SyntheticPairs()
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    cfg.train.epochs = 4
    cfg.train.checkpoint_every = 2
    run_dir = fit(cfg)

    assert (run_dir / "epoch_002.pt").is_file()
    assert (run_dir / "epoch_004.pt").is_file()
    assert not (run_dir / "epoch_001.pt").exists()
    assert not (run_dir / "epoch_003.pt").exists()
    # best.pt + last.pt still saved every epoch
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "last.pt").is_file()


def test_env_json_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() writes env.json with expected keys."""
    import json as _json

    ds = _SyntheticPairs()
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    run_dir = fit(cfg)

    env_path = run_dir / "env.json"
    assert env_path.is_file()
    env = _json.loads(env_path.read_text())
    expected_keys = {
        "python_version",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "device",
        "platform",
        "git_sha",
        "git_dirty",
        "hostname",
        "amp_enabled",
    }
    assert expected_keys <= set(env), f"missing keys: {expected_keys - set(env)}"
    assert env["device"] == "cpu"
    assert env["amp_enabled"] is False


def test_set_global_seed_enables_determinism() -> None:
    """set_global_seed turns on torch.use_deterministic_algorithms (warn_only)."""
    from spectrafan.train import set_global_seed

    set_global_seed(0)
    assert torch.are_deterministic_algorithms_enabled()


def test_set_global_seed_deterministic_false_enables_benchmark() -> None:
    """set_global_seed(deterministic=False) turns OFF the deterministic flag
    and turns ON cudnn.benchmark (the speed-trading path used by sweep configs).
    Restore deterministic mode afterwards so it doesn't leak across tests."""
    from spectrafan.train import set_global_seed

    try:
        set_global_seed(0, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
    finally:
        set_global_seed(0, deterministic=True)


def test_full_repro_config_loads() -> None:
    """configs/full_repro.yaml loads cleanly and carries the locked recipe values."""
    from spectrafan.train import load_config

    cfg = load_config(Path("configs/full_repro.yaml"))
    assert cfg.train.epochs == 50
    assert cfg.train.amp is True
    assert cfg.train.checkpoint_every == 10
    # Inherited from default.yaml
    assert cfg.data.image_size == 256
    assert cfg.data.batch_size == 4
    assert cfg.optim.lr == 1.0e-5
    assert cfg.optim.decay == 0.99
    assert cfg.model.fam_conv_kind == "depthwise"
    assert cfg.model.bottleneck == 1024
    assert cfg.model.output_norm == "bn"


def test_resume_continues_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit two epochs, then resume from last.pt for two more; metrics.parquet has 4 rows total."""
    ds = _SyntheticPairs()
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    cfg.train.epochs = 2
    run_dir = fit(cfg)

    df_before = pl.read_parquet(run_dir / "metrics.parquet")
    assert df_before.height == 2

    # Resume with epochs bumped to 4 — should add 2 more rows (epochs 2 and 3).
    cfg.train.epochs = 4
    run_dir_2 = fit(cfg, resume_from=run_dir / "last.pt")
    assert run_dir_2 == run_dir, "resume must write back into the same run dir"

    df_after = pl.read_parquet(run_dir / "metrics.parquet")
    assert df_after.height == 4
    epochs = df_after["epoch"].to_list()
    assert epochs == [0, 1, 2, 3], f"unexpected epoch sequence after resume: {epochs}"


def test_model_config_name_defaults_to_fanet() -> None:
    """`ModelConfig.name` defaults to "fanet" so existing configs without the
    field continue to instantiate the original FANet class."""
    from spectrafan.train import ModelConfig

    cfg = ModelConfig()
    assert cfg.name == "fanet"


def test_dict_to_run_config_parses_model_name(tmp_path: Path) -> None:
    """The YAML loader picks up `model.name` from a config file."""
    from spectrafan.train import load_config

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(
        "model:\n  name: fanetmini\ndata:\n  splits_dir: data/splits/temimagenet_v1\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.model.name == "fanetmini"


def test_dict_to_run_config_defaults_model_name_when_omitted(tmp_path: Path) -> None:
    """A config without `model.name` falls back to the dataclass default."""
    from spectrafan.train import load_config

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("data:\n  splits_dir: data/splits/temimagenet_v1\n")
    cfg = load_config(cfg_path)
    assert cfg.model.name == "fanet"


def test_build_model_fanet_default() -> None:
    """build_model returns a FANet instance for the default name."""
    from spectrafan.train import ModelConfig, build_model
    from spectrafan.unet import FANet, FANetMini

    cfg = ModelConfig()  # name defaults to "fanet"
    model = build_model(cfg)
    assert isinstance(model, FANet)
    assert not isinstance(model, FANetMini), (
        "ModelConfig() with default name=fanet must NOT return a FANetMini"
    )


def test_build_model_fanetmini() -> None:
    """build_model returns a FANetMini instance when name='fanetmini'."""
    from spectrafan.train import ModelConfig, build_model
    from spectrafan.unet import FANetMini

    cfg = ModelConfig(name="fanetmini")
    model = build_model(cfg)
    assert isinstance(model, FANetMini)
    # FANetMini hardcodes its widths regardless of cfg.channels/bottleneck.
    assert tuple(fam.channels for fam in model.fams) == (32, 64, 128)


def test_build_model_unknown_name_raises() -> None:
    """build_model rejects unknown model.name values with a ValueError."""
    from spectrafan.train import ModelConfig, build_model

    cfg = ModelConfig(name="not-a-real-model")
    with pytest.raises(ValueError, match="unknown model.name"):
        build_model(cfg)


def test_build_model_fanetmini_ignores_channels_and_bottleneck() -> None:
    """When name='fanetmini', cfg.channels/bottleneck are deliberately
    ignored by the dispatch (the class hardcodes them)."""
    from spectrafan.train import ModelConfig, build_model

    cfg = ModelConfig(name="fanetmini", channels=(99, 99, 99, 99), bottleneck=99)
    model = build_model(cfg)
    assert tuple(fam.channels for fam in model.fams) == (32, 64, 128), (
        "FANetMini must hardcode (32, 64, 128); cfg.channels must be ignored"
    )


def test_optim_config_new_field_defaults() -> None:
    """The 4 new OptimConfig fields default to backward-compatible values."""
    from spectrafan.train import OptimConfig

    cfg = OptimConfig()
    assert cfg.betas == (0.9, 0.999)
    assert cfg.schedule == "exponential"
    assert cfg.warmup_epochs == 0
    assert cfg.min_lr == 0.0


def test_load_config_passes_new_optim_fields(tmp_path: Path) -> None:
    """load_config round-trips all 4 new optim fields from YAML into OptimConfig."""
    from spectrafan.train import load_config

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(
        "data:\n  splits_dir: data/splits/temimagenet_v1\n"
        "optim:\n"
        "  optimizer: adamw\n"
        "  lr: 1.0e-4\n"
        "  weight_decay: 1.0e-4\n"
        "  betas: [0.9, 0.95]\n"
        "  schedule: cosine\n"
        "  warmup_epochs: 5\n"
        "  min_lr: 1.0e-6\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.optim.optimizer == "adamw"
    assert cfg.optim.lr == 1.0e-4
    assert cfg.optim.weight_decay == 1.0e-4
    assert cfg.optim.betas == (0.9, 0.95)
    assert cfg.optim.schedule == "cosine"
    assert cfg.optim.warmup_epochs == 5
    assert cfg.optim.min_lr == 1.0e-6


def test_build_optimizer_rmsprop_uses_momentum() -> None:
    """build_optimizer returns RMSprop with the configured momentum."""
    import torch

    from spectrafan.train import OptimConfig, build_optimizer

    model = torch.nn.Linear(3, 1)
    cfg = OptimConfig(optimizer="rmsprop", lr=1.0e-5, momentum=0.99, weight_decay=1.0e-8)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.RMSprop)
    pg = opt.param_groups[0]
    assert pg["lr"] == 1.0e-5
    assert pg["momentum"] == 0.99
    assert pg["weight_decay"] == 1.0e-8


def test_build_optimizer_adamw_uses_betas() -> None:
    """build_optimizer returns AdamW with the configured betas."""
    import torch

    from spectrafan.train import OptimConfig, build_optimizer

    model = torch.nn.Linear(3, 1)
    cfg = OptimConfig(optimizer="adamw", lr=1.0e-4, betas=(0.9, 0.95), weight_decay=1.0e-4)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    pg = opt.param_groups[0]
    assert pg["lr"] == 1.0e-4
    assert pg["betas"] == (0.9, 0.95)
    assert pg["weight_decay"] == 1.0e-4


def test_build_optimizer_unknown_raises() -> None:
    """build_optimizer rejects unknown optimizer names with a ValueError."""
    import torch

    from spectrafan.train import OptimConfig, build_optimizer

    model = torch.nn.Linear(3, 1)
    cfg = OptimConfig(optimizer="sgd")
    with pytest.raises(ValueError, match="unknown optim.optimizer"):
        build_optimizer(model, cfg)


def test_build_scheduler_exponential_no_warmup() -> None:
    """schedule='exponential' with warmup_epochs=0 returns ExponentialLR directly."""
    import torch

    from spectrafan.train import OptimConfig, build_optimizer, build_scheduler

    model = torch.nn.Linear(3, 1)
    cfg = OptimConfig(optimizer="rmsprop", schedule="exponential", decay=0.95, warmup_epochs=0)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_epochs=10)
    assert isinstance(sched, torch.optim.lr_scheduler.ExponentialLR)
    assert sched.gamma == 0.95


def test_build_scheduler_cosine_no_warmup() -> None:
    """schedule='cosine' with warmup_epochs=0 returns CosineAnnealingLR with correct T_max."""
    import torch

    from spectrafan.train import OptimConfig, build_optimizer, build_scheduler

    model = torch.nn.Linear(3, 1)
    cfg = OptimConfig(optimizer="adamw", schedule="cosine", warmup_epochs=0, min_lr=1.0e-6)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_epochs=50)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 50
    assert sched.eta_min == 1.0e-6


def test_build_scheduler_cosine_with_warmup_uses_sequentialLR() -> None:  # noqa: N802
    """schedule='cosine' with warmup_epochs>0 wraps LinearLR + CosineAnnealingLR in SequentialLR."""
    import torch

    from spectrafan.train import OptimConfig, build_optimizer, build_scheduler

    model = torch.nn.Linear(3, 1)
    cfg = OptimConfig(optimizer="adamw", schedule="cosine", warmup_epochs=5, min_lr=1.0e-6)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_epochs=50)
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)
    inner = sched._schedulers
    assert isinstance(inner[0], torch.optim.lr_scheduler.LinearLR)
    assert isinstance(inner[1], torch.optim.lr_scheduler.CosineAnnealingLR)
    assert inner[1].T_max == 45  # 50 - 5 warmup
    assert sched._milestones == [5]


def test_build_scheduler_warmup_exceeds_epochs_raises() -> None:
    """warmup_epochs >= total_epochs is rejected with a ValueError."""
    import torch

    from spectrafan.train import OptimConfig, build_optimizer, build_scheduler

    model = torch.nn.Linear(3, 1)
    cfg = OptimConfig(optimizer="adamw", schedule="cosine", warmup_epochs=50)
    opt = build_optimizer(model, cfg)
    with pytest.raises(ValueError, match="warmup_epochs"):
        build_scheduler(opt, cfg, total_epochs=50)


def test_build_scheduler_unknown_raises() -> None:
    """build_scheduler rejects unknown schedule names with a ValueError."""
    import torch

    from spectrafan.train import OptimConfig, build_optimizer, build_scheduler

    model = torch.nn.Linear(3, 1)
    cfg = OptimConfig(optimizer="rmsprop", schedule="step")
    opt = build_optimizer(model, cfg)
    with pytest.raises(ValueError, match="unknown optim.schedule"):
        build_scheduler(opt, cfg, total_epochs=10)


def test_fit_smoke_adamw_cosine_warmup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() runs end-to-end with the AdamW+cosine+warmup bundle, produces a
    finite metrics.parquet, AND saves an AdamW-shaped optimizer state (proves
    fit() actually switched optimizers, not just survived the run)."""
    ds = _SyntheticPairs()
    import spectrafan.train as train_mod

    monkeypatch.setattr(train_mod, "build_datasets", lambda _cfg: (ds, ds))

    cfg = _tiny_cfg(tmp_path)
    cfg.optim.optimizer = "adamw"
    cfg.optim.lr = 1.0e-3  # match _tiny_cfg's tuned scale
    cfg.optim.weight_decay = 1.0e-4
    cfg.optim.betas = (0.9, 0.999)
    cfg.optim.schedule = "cosine"
    cfg.optim.warmup_epochs = 1
    cfg.optim.min_lr = 1.0e-6
    cfg.train.epochs = 3
    run_dir = fit(cfg)

    df = pl.read_parquet(run_dir / "metrics.parquet")
    assert df.height == 3
    for col in ("train_loss", "val_loss", "train_iou", "val_iou"):
        values = df[col].to_list()
        assert all(v == v for v in values), f"{col} has NaN: {values}"

    # Prove the optimizer was actually AdamW (not RMSprop).
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    state_per_param = ckpt["optimizer_state_dict"]["state"]
    assert state_per_param, "optimizer state is empty (no steps were taken?)"
    sample_state = next(iter(state_per_param.values()))
    assert "exp_avg" in sample_state and "exp_avg_sq" in sample_state, (
        f"expected AdamW state keys (exp_avg, exp_avg_sq), got {list(sample_state.keys())}"
    )


def test_fanetmini_sweep_configs_load() -> None:
    """All three sweep configs round-trip through load_config and produce the
    expected OptimConfig + DataConfig + loss values from the spec's sweep matrix."""
    from spectrafan.train import load_config

    cfg_b = load_config(Path("configs/fanetmini_sweep_B.yaml"))
    assert cfg_b.model.name == "fanetmini"
    assert cfg_b.data.batch_size == 16
    assert cfg_b.data.num_workers == 8
    assert cfg_b.optim.optimizer == "rmsprop"
    assert cfg_b.optim.lr == 4.0e-5
    assert cfg_b.optim.momentum == 0.9
    assert cfg_b.optim.schedule == "exponential"
    assert cfg_b.train.epochs == 50
    assert cfg_b.train.amp is True
    assert cfg_b.train.deterministic is False
    assert cfg_b.train.loss_ce_weight == 0.5
    assert cfg_b.train.loss_dice_weight == 0.5

    cfg_c = load_config(Path("configs/fanetmini_sweep_C.yaml"))
    assert cfg_c.model.name == "fanetmini"
    assert cfg_c.data.batch_size == 16
    assert cfg_c.data.num_workers == 8
    assert cfg_c.optim.optimizer == "adamw"
    assert cfg_c.optim.lr == 1.0e-4
    assert cfg_c.optim.weight_decay == 1.0e-4
    assert cfg_c.optim.betas == (0.9, 0.999)
    assert cfg_c.optim.schedule == "cosine"
    assert cfg_c.optim.warmup_epochs == 5
    assert cfg_c.optim.min_lr == 1.0e-6
    assert cfg_c.train.epochs == 50
    assert cfg_c.train.amp is True
    assert cfg_c.train.deterministic is False
    assert cfg_c.train.loss_ce_weight == 0.5
    assert cfg_c.train.loss_dice_weight == 0.5

    cfg_d = load_config(Path("configs/fanetmini_sweep_D.yaml"))
    assert cfg_d.model.name == "fanetmini"
    assert cfg_d.data.num_workers == 8  # inherited from C via single-hop extends
    assert cfg_d.optim.optimizer == "adamw"
    assert cfg_d.optim.schedule == "cosine"
    assert cfg_d.optim.warmup_epochs == 5
    assert cfg_d.train.amp is True  # inherited from C
    assert cfg_d.train.deterministic is False  # inherited from C
    assert cfg_d.train.loss_ce_weight == 0.3
    assert cfg_d.train.loss_dice_weight == 0.7
