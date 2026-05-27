"""Segmentation metrics: IoU, Dice, pixel accuracy, and a num/den running aggregator.

The standalone functions take probabilities (post-sigmoid) so unit tests can
pass crafted binary tensors directly. ``RunningMetrics.update`` takes logits
and applies sigmoid internally; that's the API the train/validate loops use.

Aggregation across batches uses running sums of intersection / union / Dice
numerator / Dice denominator / correct / total, and computes ratios once at
``compute()``. Mean-of-batch-means is biased on the last partial batch.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _binarize(prob: Tensor, threshold: float) -> Tensor:
    return (prob >= threshold).float()


def iou(prob: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    """Mean IoU treating the whole input as one set. Empty union returns 1.0."""
    pred = _binarize(prob, threshold)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    if union.item() == 0.0:
        return torch.tensor(1.0, device=prob.device)
    return inter / union


def dice(prob: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    """Dice over the whole input. Empty pred + empty target returns 1.0."""
    pred = _binarize(prob, threshold)
    num = 2.0 * (pred * target).sum()
    den = pred.sum() + target.sum()
    if den.item() == 0.0:
        return torch.tensor(1.0, device=prob.device)
    return num / den


def pixel_accuracy(prob: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    pred = _binarize(prob, threshold)
    return (pred == target).float().mean()


class RunningMetrics:
    """Accumulate IoU / Dice / pixel-accuracy by summing num/den across batches.

    Sums are kept as zero-dim tensors on the input device and only synced to
    host scalars in ``compute()``. This avoids one ``.item()`` per batch per
    statistic, which serializes the GPU/MPS pipeline.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._inter: Tensor | None = None
        self._union: Tensor | None = None
        self._dice_num: Tensor | None = None
        self._dice_den: Tensor | None = None
        self._correct: Tensor | None = None
        self._total: int = 0

    def reset(self) -> None:
        self._inter = None
        self._union = None
        self._dice_num = None
        self._dice_den = None
        self._correct = None
        self._total = 0

    def update(self, logits: Tensor, target: Tensor) -> None:
        prob = torch.sigmoid(logits.detach())
        target = target.detach()
        pred = _binarize(prob, self.threshold)
        inter = (pred * target).sum()
        pred_plus_target = pred.sum() + target.sum()
        union = pred_plus_target - inter
        correct = (pred == target).sum()

        if self._inter is None:
            self._inter = inter
            self._union = union
            self._dice_num = 2.0 * inter
            self._dice_den = pred_plus_target
            self._correct = correct
        else:
            self._inter = self._inter + inter
            self._union = self._union + union
            self._dice_num = self._dice_num + 2.0 * inter
            self._dice_den = self._dice_den + pred_plus_target
            self._correct = self._correct + correct
        self._total += target.numel()

    def compute(self) -> dict[str, float]:
        if (
            self._inter is None
            or self._union is None
            or self._dice_num is None
            or self._dice_den is None
            or self._correct is None
        ):
            return {"iou": 1.0, "dice": 1.0, "px_acc": 0.0}
        inter = float(self._inter.item())
        union = float(self._union.item())
        dice_num = float(self._dice_num.item())
        dice_den = float(self._dice_den.item())
        correct = float(self._correct.item())
        iou_val = 1.0 if union == 0.0 else inter / union
        dice_val = 1.0 if dice_den == 0.0 else dice_num / dice_den
        acc_val = 0.0 if self._total == 0 else correct / self._total
        return {"iou": iou_val, "dice": dice_val, "px_acc": acc_val}
