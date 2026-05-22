"""Training entry point for FANet.

Composes TEMImageNetDataset + augmentation + BCEDiceLoss + RunningMetrics +
RMSprop + ExponentialLR into ``fit(cfg)``. Writes per-epoch metrics to
``runs/<id>/metrics.parquet`` and checkpoints ``last.pt`` / ``best.pt``.

CLI:
    uv run python -m spectrafan.train --config configs/smoke.yaml \\
        [--override key.subkey=value ...]
"""

from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from spectrafan.data import TEMImageNetDataset
from spectrafan.fam import ConvKind
from spectrafan.losses import BCEDiceLoss
from spectrafan.metrics import RunningMetrics
from spectrafan.transforms import eval_transforms, train_transforms
from spectrafan.unet import FANet, OutputNorm

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    channels: tuple[int, ...] = (64, 128, 256, 512)
    bottleneck: int = 1024
    fam_conv_kind: ConvKind = "depthwise"
    output_norm: OutputNorm = "bn"


@dataclass
class DataConfig:
    dataset: str = "temimagenet"
    root: Path = Path("data/raw/temimagenet")
    image_size: int = 512
    batch_size: int = 4
    subset_size: int | None = None
    val_subset_size: int | None = None
    splits_dir: Path = Path("data/splits/temimagenet_v1")
    num_workers: int = 2


@dataclass
class AugConfig:
    p_flip: float = 0.5
    max_rot_deg: float = 15.0
    zoom_range: tuple[float, float] = (0.9, 1.1)
    noise_sigma: float = 0.01


@dataclass
class OptimConfig:
    optimizer: str = "rmsprop"
    lr: float = 1.0e-5
    decay: float = 0.99
    weight_decay: float = 1.0e-8
    momentum: float = 0.999


@dataclass
class TrainConfig:
    epochs: int = 200
    seed: int = 0
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    run_root: Path = Path("runs")
    loss_ce_weight: float = 0.5
    loss_dice_weight: float = 0.5
    amp: bool = False
    checkpoint_every: int = 10


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Config loading (YAML + extends + --override)
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    """Parse `--override` values as YAML scalars (bool/int/float/str/list)."""
    return yaml.safe_load(value)


def _apply_override(d: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def load_config(path: Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML config with ``extends:`` support and ``key.subkey=value`` overrides."""
    raw = yaml.safe_load(path.read_text()) or {}
    if "extends" in raw:
        base_path = path.parent / raw.pop("extends")
        base = yaml.safe_load(base_path.read_text()) or {}
        raw = _deep_merge(base, raw)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must be key=value; got {override!r}")
        key, _, value = override.partition("=")
        _apply_override(raw, key.strip(), _coerce(value.strip()))
    return _dict_to_run_config(raw)


def _dict_to_run_config(d: dict) -> RunConfig:
    model = ModelConfig(
        channels=tuple(d.get("model", {}).get("channels", ModelConfig.channels)),
        bottleneck=d.get("model", {}).get("bottleneck", ModelConfig.bottleneck),
        fam_conv_kind=d.get("model", {}).get("fam_conv_kind", ModelConfig.fam_conv_kind),
        output_norm=d.get("model", {}).get("output_norm", ModelConfig.output_norm),
    )
    data_raw = d.get("data", {}) or {}
    data = DataConfig(
        dataset=data_raw.get("dataset", DataConfig.dataset),
        root=Path(data_raw.get("root", DataConfig.root)),
        image_size=data_raw.get("image_size", DataConfig.image_size),
        batch_size=data_raw.get("batch_size", DataConfig.batch_size),
        subset_size=data_raw.get("subset_size", None),
        val_subset_size=data_raw.get("val_subset_size", None),
        splits_dir=Path(data_raw.get("splits_dir", DataConfig.splits_dir)),
        num_workers=data_raw.get("num_workers", DataConfig.num_workers),
    )
    aug_raw = d.get("aug", {}) or {}
    aug = AugConfig(
        p_flip=aug_raw.get("p_flip", AugConfig.p_flip),
        max_rot_deg=aug_raw.get("max_rot_deg", AugConfig.max_rot_deg),
        zoom_range=tuple(aug_raw.get("zoom_range", AugConfig.zoom_range)),
        noise_sigma=aug_raw.get("noise_sigma", AugConfig.noise_sigma),
    )
    optim_raw = d.get("optim", {}) or {}
    optim_cfg = OptimConfig(
        optimizer=optim_raw.get("optimizer", OptimConfig.optimizer),
        lr=optim_raw.get("lr", OptimConfig.lr),
        decay=optim_raw.get("decay", OptimConfig.decay),
        weight_decay=optim_raw.get("weight_decay", OptimConfig.weight_decay),
        momentum=optim_raw.get("momentum", OptimConfig.momentum),
    )
    train_raw = d.get("train", {}) or {}
    loss_raw = train_raw.get("loss", {}) or {}
    train_cfg = TrainConfig(
        epochs=train_raw.get("epochs", TrainConfig.epochs),
        seed=train_raw.get("seed", TrainConfig.seed),
        device=train_raw.get("device", TrainConfig.device),
        run_root=Path(train_raw.get("run_root", TrainConfig.run_root)),
        loss_ce_weight=loss_raw.get("ce_weight", TrainConfig.loss_ce_weight),
        loss_dice_weight=loss_raw.get("dice_weight", TrainConfig.loss_dice_weight),
        amp=train_raw.get("amp", TrainConfig.amp),
        checkpoint_every=train_raw.get("checkpoint_every", TrainConfig.checkpoint_every),
    )
    return RunConfig(model=model, data=data, aug=aug, optim=optim_cfg, train=train_cfg)


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


def set_global_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort; not fully deterministic on MPS.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(cfg: RunConfig, config_stem: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = cfg.train.run_root / f"{timestamp}_{config_stem}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_run_config_to_dict(cfg)))
    return run_dir


def _run_config_to_dict(cfg: RunConfig) -> dict:
    d = asdict(cfg)
    # asdict turns tuples into tuples and Paths into Paths; YAML wants lists and strings.
    d["model"]["channels"] = list(d["model"]["channels"])
    d["aug"]["zoom_range"] = list(d["aug"]["zoom_range"])
    d["data"]["root"] = str(d["data"]["root"])
    d["data"]["splits_dir"] = str(d["data"]["splits_dir"])
    d["train"]["run_root"] = str(d["train"]["run_root"])
    return d


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
            zoom_range=cfg.aug.zoom_range,
            noise_sigma=cfg.aug.noise_sigma,
        ),
        subset_size=cfg.data.subset_size,
    )
    val_ds = TEMImageNetDataset(
        root=cfg.data.root,
        split="val",
        image_size=cfg.data.image_size,
        splits_dir=cfg.data.splits_dir,
        transforms=eval_transforms(),
        subset_size=cfg.data.val_subset_size,
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
    """Append-per-epoch parquet writer (rewrites the file each append)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: list[dict] = []

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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool = False,
) -> dict[str, float]:
    model.train()
    rm = RunningMetrics()
    loss_sum = 0.0
    n_batches = 0
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(amp_enabled, device):
            logits = model(image)
            loss = loss_fn(logits, mask)
        logits = logits.float()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        n_batches += 1
        rm.update(logits, mask)
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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_iou": val_iou,
            "config": _run_config_to_dict(cfg),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Top-level fit
# ---------------------------------------------------------------------------


def fit(cfg: RunConfig, config_stem: str = "run") -> Path:
    set_global_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    run_dir = make_run_dir(cfg, config_stem=config_stem)

    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_loaders(train_ds, val_ds, cfg, device)

    model = FANet(
        channels=tuple(cfg.model.channels),
        bottleneck=cfg.model.bottleneck,
        output_norm=cfg.model.output_norm,
        conv_kind=cfg.model.fam_conv_kind,
    ).to(device)

    loss_fn = BCEDiceLoss(
        ce_weight=cfg.train.loss_ce_weight, dice_weight=cfg.train.loss_dice_weight
    )
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )
    # Liu et al. 2026 Table on §3.2 lists "LR decay rate 0.99" — interpreted here as
    # ExponentialLR γ=0.99 per epoch. RMSprop's smoothing constant `alpha` is not
    # specified by the paper and is left at PyTorch's default (0.99 — coincidentally).
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.optim.decay)

    writer = MetricsParquetWriter(run_dir / "metrics.parquet")
    best_val_iou = float("-inf")

    for epoch in range(cfg.train.epochs):
        t0 = time.perf_counter()
        lr_this_epoch = optimizer.param_groups[0]["lr"]
        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, amp_enabled=cfg.train.amp,
        )
        val_stats = validate(
            model, val_loader, loss_fn, device, amp_enabled=cfg.train.amp,
        )
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
        writer.append(row)
        _save_checkpoint(
            run_dir / "last.pt", model, optimizer, scheduler, epoch, val_stats["iou"], cfg
        )
        if val_stats["iou"] > best_val_iou:
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
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    run_dir = fit(cfg, config_stem=args.config.stem)
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    _main()
