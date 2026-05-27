"""Training entry point for FANet.

Composes TEMImageNetDataset + augmentation + BCEDiceLoss + RunningMetrics +
a configurable optimizer (RMSprop or AdamW) + LR scheduler into ``fit(cfg)``.
Writes per-epoch metrics to
``runs/<id>/metrics.parquet`` and checkpoints ``last.pt`` / ``best.pt``.

CLI:
    uv run python -m spectrafan.train --config configs/smoke.yaml \\
        [--override key.subkey=value ...]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from spectrafan.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    load_config,
    run_config_to_dict,
)
from spectrafan.data import TEMImageNetDataset
from spectrafan.losses import BCEDiceLoss
from spectrafan.metrics import RunningMetrics
from spectrafan.transforms import eval_transforms, train_transforms
from spectrafan.unet import FANet, FANetMini

# ---------------------------------------------------------------------------
# Device + seed + run dir
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs and (optionally) enable deterministic CUDA kernels.

    When `deterministic=False`, cudnn.benchmark is turned on and the global
    deterministic-algorithm flag is turned off — trades bitwise reproducibility
    for ~5-20x conv kernel speedup on shape/dtype combos where deterministic
    kernels are slow (a real bottleneck for FANetMini at BS>=16).
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Best-effort; not fully deterministic on MPS.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS_WORKSPACE_CONFIG is required by torch.use_deterministic_algorithms on CUDA;
        # without it, cuBLAS matmul ops raise RuntimeError that warn_only does NOT silence.
        # setdefault respects any pre-existing override from the user's shell.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only=True so ops without deterministic kernels (e.g. some interpolations)
        # don't crash training; the warning still surfaces the nondeterminism.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def _capture_env(device: torch.device, amp_enabled: bool) -> dict:
    """Snapshot the runtime environment for reproducibility audits.

    Failures fall through to None rather than aborting the run — env.json is
    best-effort metadata, not a hard precondition.
    """

    def _safe(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    git_sha = _safe(["git", "rev-parse", "HEAD"])
    git_status = _safe(["git", "status", "--porcelain"])
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(device),
        "platform": platform.platform(),
        "git_sha": git_sha,
        "git_dirty": bool(git_status) if git_status is not None else None,
        "hostname": socket.gethostname(),
        "amp_enabled": amp_enabled,
    }


def make_run_dir(cfg: RunConfig, config_stem: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(cfg.train.run_root) / f"{timestamp}_{config_stem}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config_to_dict(cfg)))
    return run_dir


# ---------------------------------------------------------------------------
# Dataset / loader construction (separated so tests can monkey-patch)
# ---------------------------------------------------------------------------


def build_datasets(cfg: RunConfig) -> tuple[Dataset, Dataset]:
    train_ds = TEMImageNetDataset(
        root=cfg.data.root,
        split="train",
        image_size=cfg.data.image_size,
        splits_dir=cfg.data.splits_dir,
        transforms=train_transforms(
            p_flip=cfg.aug.p_flip,
            max_rot_deg=cfg.aug.max_rot_deg,
            zoom_range=tuple(cfg.aug.zoom_range),
            noise_sigma=cfg.aug.noise_sigma,
        ),
        subset_size=cfg.data.subset_size,
        input_norm=cfg.data.input_norm,
        in_channels=cfg.data.in_channels,
    )
    val_ds = TEMImageNetDataset(
        root=cfg.data.root,
        split="val",
        image_size=cfg.data.image_size,
        splits_dir=cfg.data.splits_dir,
        transforms=eval_transforms(),
        subset_size=cfg.data.val_subset_size,
        input_norm=cfg.data.input_norm,
        in_channels=cfg.data.in_channels,
    )
    return train_ds, val_ds


def build_loaders(
    train_ds: Dataset, val_ds: Dataset, cfg: RunConfig, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    pin = device.type == "cuda"
    persistent = cfg.data.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Metrics writer
# ---------------------------------------------------------------------------


class MetricsParquetWriter:
    """Append-per-epoch parquet writer (rewrites the file each append).

    If the path already exists, prior rows are loaded so subsequent appends
    extend the historical record (used by --resume).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: list[dict] = []
        if path.is_file():
            existing = pl.read_parquet(path)
            self._rows = existing.to_dicts()

    def append(self, row: dict) -> None:
        self._rows.append(row)
        pl.DataFrame(self._rows).write_parquet(self.path)


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------


def _autocast_ctx(amp_enabled: bool, device: torch.device) -> contextlib.AbstractContextManager:
    """Return a bf16 autocast context for CUDA when AMP is on; no-op otherwise.

    bf16 has the same exponent range as fp32, so no GradScaler is needed.
    On MPS/CPU autocast is skipped — the device-portable recipe stays fp32 there.
    """
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


_TRAIN_PROFILE_ENV = "SPECTRAFAN_TRAIN_PROFILE"


def _train_profile_enabled() -> bool:
    """Opt-in via env var; on for epoch 1 only, zero overhead when off."""
    return os.environ.get(_TRAIN_PROFILE_ENV, "0") not in ("0", "", "false", "False")


def _sync(device: torch.device) -> None:
    """torch.cuda.synchronize() on CUDA, no-op otherwise. Required around
    timers because launches are async — without sync we'd time queue depth,
    not actual work."""
    if device.type == "cuda":
        torch.cuda.synchronize()


class _PhaseTimer:
    """Context manager that prints `[profile] <name>=<sec>s` on exit. No-op
    when enabled=False, so callers can wrap unconditionally."""

    def __init__(self, name: str, enabled: bool, device: torch.device) -> None:
        self.name = name
        self.enabled = enabled
        self.device = device
        self.t0 = 0.0

    def __enter__(self) -> _PhaseTimer:
        if self.enabled:
            _sync(self.device)
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.enabled:
            _sync(self.device)
            dt = time.perf_counter() - self.t0
            print(f"  [profile] {self.name}={dt:.2f}s", flush=True)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool = False,
    *,
    profile: bool = False,
) -> dict[str, float]:
    model.train()
    rm = RunningMetrics()
    loss_sum = 0.0
    n_batches = 0
    data_wait_s: list[float] = []
    fwd_bwd_s: list[float] = []
    step_s: list[float] = []
    t_prev = time.perf_counter()  # always defined; only read+used when profile is on
    for image, mask in loader:
        if profile:
            _sync(device)
            t_now = time.perf_counter()
            data_wait_s.append(t_now - t_prev)
            t_prev = t_now
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(amp_enabled, device):
            logits = model(image)
            loss = loss_fn(logits, mask)
        logits = logits.float()
        loss.backward()
        if profile:
            _sync(device)
            t_now = time.perf_counter()
            fwd_bwd_s.append(t_now - t_prev)
            t_prev = t_now
        optimizer.step()
        if profile:
            _sync(device)
            t_now = time.perf_counter()
            step_s.append(t_now - t_prev)
            t_prev = t_now
        loss_sum += loss.item()
        n_batches += 1
        rm.update(logits, mask)
    if profile:
        from statistics import median as _med

        for name, samples in (
            ("data_wait", data_wait_s),
            ("fwd_bwd", fwd_bwd_s),
            ("optim_step", step_s),
        ):
            if samples:
                print(
                    f"  [profile] train/{name}: "
                    f"median={_med(samples) * 1000:.1f}ms "
                    f"total={sum(samples):.1f}s n={len(samples)}",
                    flush=True,
                )
    metrics = rm.compute()
    return {"loss": loss_sum / max(n_batches, 1), **metrics}


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
) -> dict[str, float]:
    model.eval()
    rm = RunningMetrics()
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with _autocast_ctx(amp_enabled, device):
                logits = model(image)
                loss = loss_fn(logits, mask)
            logits = logits.float()
            loss_sum += loss.item()
            n_batches += 1
            rm.update(logits, mask)
    metrics = rm.compute()
    return {"loss": loss_sum / max(n_batches, 1), **metrics}


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_iou: float,
    cfg: RunConfig,
) -> None:
    import random as _random

    import numpy as _np

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_iou": val_iou,
            "config": run_config_to_dict(cfg),
            "rng": {
                "python": _random.getstate(),
                "numpy": _np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
        },
        path,
    )


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------


def _load_resume_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    """Restore model/optimizer/scheduler/RNG state. Returns the starting epoch (saved_epoch + 1)."""
    import random as _random

    import numpy as _np

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    rng = ckpt.get("rng", {})
    if "python" in rng:
        _random.setstate(rng["python"])
    if "numpy" in rng:
        _np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return int(ckpt["epoch"]) + 1


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------


def build_model(model_cfg: ModelConfig, data_cfg: DataConfig) -> torch.nn.Module:
    """Construct the model class chosen by `model_cfg.name`.

    For `fanetmini`, channels and bottleneck on the config are ignored
    (the class hardcodes them); only `output_norm` and `fam_conv_kind`
    flow through. `data_cfg.in_channels` threads the input channel count
    through both variants so 1-channel STEM inputs are supported.
    """
    if model_cfg.name == "fanet":
        return FANet(
            in_channels=data_cfg.in_channels,
            channels=tuple(model_cfg.channels),
            bottleneck=model_cfg.bottleneck,
            output_norm=model_cfg.output_norm,
            conv_kind=model_cfg.fam_conv_kind,
        )
    if model_cfg.name == "fanetmini":
        return FANetMini(
            in_channels=data_cfg.in_channels,
            output_norm=model_cfg.output_norm,
            conv_kind=model_cfg.fam_conv_kind,
        )
    raise ValueError(f"unknown model.name: {model_cfg.name!r} (expected 'fanet' or 'fanetmini')")


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """Construct the optimizer chosen by `cfg.optimizer`."""
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unknown optim.optimizer: {cfg.optimizer!r} (expected 'rmsprop' or 'adamw')")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimConfig,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construct the LR scheduler chosen by `cfg.schedule`, optionally wrapped in warmup.

    For `cfg.warmup_epochs > 0` the main scheduler is preceded by a LinearLR
    warmup via SequentialLR. The cosine schedule's `T_max` is reduced by
    `warmup_epochs` so the full cosine period still ends at `total_epochs`.
    """
    if cfg.warmup_epochs >= total_epochs:
        raise ValueError(
            f"optim.warmup_epochs ({cfg.warmup_epochs}) must be < train.epochs ({total_epochs})"
        )
    if cfg.schedule == "exponential":
        main = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.decay)
    elif cfg.schedule == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - cfg.warmup_epochs,
            eta_min=cfg.min_lr,
        )
    else:
        raise ValueError(
            f"unknown optim.schedule: {cfg.schedule!r} (expected 'exponential' or 'cosine')"
        )
    if cfg.warmup_epochs <= 0:
        return main
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, main], milestones=[cfg.warmup_epochs]
    )


# ---------------------------------------------------------------------------
# Top-level fit
# ---------------------------------------------------------------------------


def fit(cfg: RunConfig, config_stem: str = "run", resume_from: Path | None = None) -> Path:
    set_global_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)

    if resume_from is not None:
        run_dir = resume_from.parent
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume parent dir does not exist: {run_dir}")
    else:
        run_dir = make_run_dir(cfg, config_stem=config_stem)

    (run_dir / "env.json").write_text(json.dumps(_capture_env(device, cfg.train.amp), indent=2))

    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_loaders(train_ds, val_ds, cfg, device)

    model = build_model(cfg.model, cfg.data).to(device)

    loss_fn = BCEDiceLoss(
        ce_weight=cfg.train.loss.ce_weight, dice_weight=cfg.train.loss.dice_weight
    )
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.train.epochs)

    start_epoch = 0
    if resume_from is not None:
        start_epoch = _load_resume_state(resume_from, model, optimizer, scheduler)

    writer = MetricsParquetWriter(run_dir / "metrics.parquet")
    best_val_iou = max(
        (row.get("val_iou", float("-inf")) for row in writer._rows),
        default=float("-inf"),
    )

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.perf_counter()
        lr_this_epoch = optimizer.param_groups[0]["lr"]
        # Profile epoch 1 only (relative to the start of this fit call) when
        # SPECTRAFAN_TRAIN_PROFILE=1 — captures per-phase wall-clock breakdown
        # to diagnose what's actually slow.
        profile_this_epoch = _train_profile_enabled() and epoch == start_epoch
        with _PhaseTimer("train_loop", profile_this_epoch, device):
            train_stats = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                amp_enabled=cfg.train.amp,
                profile=profile_this_epoch,
            )
        with _PhaseTimer("val_loop", profile_this_epoch, device):
            val_stats = validate(
                model,
                val_loader,
                loss_fn,
                device,
                amp_enabled=cfg.train.amp,
            )
        with _PhaseTimer("scheduler_step", profile_this_epoch, device):
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": lr_this_epoch,
            "wall_sec": time.perf_counter() - t0,
            "train_loss": train_stats["loss"],
            "train_iou": train_stats["iou"],
            "train_dice": train_stats["dice"],
            "train_px_acc": train_stats["px_acc"],
            "val_loss": val_stats["loss"],
            "val_iou": val_stats["iou"],
            "val_dice": val_stats["dice"],
            "val_px_acc": val_stats["px_acc"],
        }
        with _PhaseTimer("metrics_parquet_write", profile_this_epoch, device):
            writer.append(row)
        print(
            f"epoch {epoch + 1:>3d}/{cfg.train.epochs}  "
            f"train_iou={train_stats['iou']:.4f}  val_iou={val_stats['iou']:.4f}  "
            f"loss={train_stats['loss']:.4f}/{val_stats['loss']:.4f}  "
            f"lr={lr_this_epoch:.2e}  wall={row['wall_sec']:.1f}s",
            flush=True,
        )
        with _PhaseTimer("save_last_pt", profile_this_epoch, device):
            _save_checkpoint(
                run_dir / "last.pt", model, optimizer, scheduler, epoch, val_stats["iou"], cfg
            )
        # Save an immortal snapshot every checkpoint_every epochs. 1-indexed in
        # the filename for human readability; 0-indexed in code.
        if (epoch + 1) % cfg.train.checkpoint_every == 0:
            with _PhaseTimer("save_epoch_pt", profile_this_epoch, device):
                _save_checkpoint(
                    run_dir / f"epoch_{epoch + 1:03d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_stats["iou"],
                    cfg,
                )
        if val_stats["iou"] > best_val_iou:
            with _PhaseTimer("save_best_pt", profile_this_epoch, device):
                _save_checkpoint(
                    run_dir / "best.pt", model, optimizer, scheduler, epoch, val_stats["iou"], cfg
                )
            best_val_iou = val_stats["iou"]

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(prog="python -m spectrafan.train")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY.SUBKEY=VALUE",
        help="repeatable; YAML-coerced (e.g. --override train.epochs=5)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to a checkpoint (e.g. runs/<id>/last.pt) to resume training from",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    run_dir = fit(cfg, config_stem=args.config.stem, resume_from=args.resume)
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    _main()
