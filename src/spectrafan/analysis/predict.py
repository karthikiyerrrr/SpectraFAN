"""Generate inference artifacts for a finished training run.

CLI:
    uv run python -m spectrafan.analysis.predict --run runs/<id>
    uv run python -m spectrafan.analysis.predict --latest fanetmini

Writes <run>/predictions.npz (16 val + 16 test sample preds) and
<run>/test_metrics.json (IoU/Dice/pixel-acc over the full test split).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spectrafan.data import TEMImageNetDataset
from spectrafan.data.transforms import eval_transforms
from spectrafan.training.metrics import RunningMetrics
from spectrafan.training.train import build_model, load_config, resolve_device

SAMPLE_SIZE = 16


def find_latest_run(runs_root: Path, suffix: str) -> Path:
    """Most-recently-mtime'd `runs_root/*_<suffix>/` directory.

    Raises FileNotFoundError if no matching directory exists.
    """
    candidates = sorted(
        (p for p in runs_root.glob(f"*_{suffix}") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no {runs_root}/*_{suffix}/ dirs found")
    return candidates[-1]


def _collect_subset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run `model` over every item in `dataset`. Returns (images, masks, preds, ids)
    as numpy arrays of shape (N, 1, H, W) for the first three and (N,) for ids."""
    images, masks, preds, ids = [], [], [], []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            logits = model(x.unsqueeze(0).to(device))
            p = (torch.sigmoid(logits)[0, 0] > 0.5).to("cpu").to(torch.uint8).numpy()
            images.append(x[0:1].numpy().astype(np.float32))
            masks.append(y.to(torch.uint8).numpy())
            preds.append(p[np.newaxis, ...])
            ids.append(dataset._rows[idx]["stem"])
    return (
        np.stack(images),
        np.stack(masks),
        np.stack(preds),
        np.asarray(ids, dtype=np.str_),
    )


def _full_test_metrics(
    model: torch.nn.Module, cfg, device: torch.device
) -> tuple[dict[str, float], int]:
    """Run `model` over the full test split. Returns (metrics_dict, n_samples)."""
    test_ds = TEMImageNetDataset(
        root=cfg.data.root,
        split="test",
        image_size=cfg.data.image_size,
        splits_dir=cfg.data.splits_dir,
        transforms=eval_transforms(),
    )
    loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=0)
    rm = RunningMetrics()
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rm.update(model(xb), yb)
    return rm.compute(), len(test_ds)


def predict_run(run_dir: Path) -> None:
    """Emit predictions.npz + test_metrics.json into `run_dir`.

    Loads <run_dir>/config.yaml and <run_dir>/best.pt, dispatches the model via
    `build_model`, then runs inference on a 16-sample subset of val and test
    splits (saved as arrays) and on the full test split (saved as metrics).
    """
    cfg = load_config(run_dir / "config.yaml")
    device = resolve_device(cfg.train.device)

    best_path = run_dir / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"missing best.pt in run dir: {run_dir}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)

    model = build_model(cfg.model, cfg.data).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    arrays: dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        ds = TEMImageNetDataset(
            root=cfg.data.root,
            split=split,
            image_size=cfg.data.image_size,
            splits_dir=cfg.data.splits_dir,
            transforms=eval_transforms(),
            subset_size=SAMPLE_SIZE,
        )
        imgs, msks, prds, ids = _collect_subset(model, ds, device)
        arrays[f"{split}_images"] = imgs
        arrays[f"{split}_masks"] = msks
        arrays[f"{split}_preds"] = prds
        arrays[f"{split}_ids"] = ids
    np.savez_compressed(run_dir / "predictions.npz", **arrays)

    test_metrics, test_size = _full_test_metrics(model, cfg, device)
    metrics_blob = {
        "epoch": int(ckpt["epoch"]) + 1,
        "val_iou": float(ckpt["val_iou"]),
        "test_iou": float(test_metrics["iou"]),
        "test_dice": float(test_metrics["dice"]),
        "test_px_acc": float(test_metrics["px_acc"]),
        "test_size": int(test_size),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics_blob, indent=2))


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m spectrafan.analysis.predict")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, help="Path to a run directory.")
    group.add_argument(
        "--latest",
        metavar="SUFFIX",
        help="Resolve to the most recent runs/*_<SUFFIX>/ directory.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Parent dir scanned by --latest (default: runs).",
    )
    args = parser.parse_args()
    run_dir = args.run if args.run is not None else find_latest_run(args.runs_root, args.latest)
    predict_run(run_dir)
    print(f"predict complete: {run_dir}")


if __name__ == "__main__":
    _main()
