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
        return torch.tensor(1.0)
    return inter / union


def dice(prob: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    """Dice over the whole input. Empty pred + empty target returns 1.0."""
    pred = _binarize(prob, threshold)
    num = 2.0 * (pred * target).sum()
    den = pred.sum() + target.sum()
    if den.item() == 0.0:
        return torch.tensor(1.0)
    return num / den


def pixel_accuracy(prob: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    pred = _binarize(prob, threshold)
    return (pred == target).float().mean()


class RunningMetrics:
    """Accumulate IoU / Dice / pixel-accuracy by summing num/den across batches."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._inter = 0.0
        self._union = 0.0
        self._dice_num = 0.0
        self._dice_den = 0.0
        self._correct = 0.0
        self._total = 0.0

    def update(self, logits: Tensor, target: Tensor) -> None:
        prob = torch.sigmoid(logits.detach())
        pred = _binarize(prob, self.threshold)
        inter = (pred * target).sum().item()
        self._inter += inter
        self._union += (pred.sum() + target.sum()).item() - inter
        self._dice_num += 2.0 * inter
        self._dice_den += (pred.sum() + target.sum()).item()
        self._correct += (pred == target).sum().item()
        self._total += float(target.numel())

    def compute(self) -> dict[str, float]:
        iou_val = 1.0 if self._union == 0.0 else self._inter / self._union
        dice_val = 1.0 if self._dice_den == 0.0 else self._dice_num / self._dice_den
        acc_val = 0.0 if self._total == 0.0 else self._correct / self._total
        return {"iou": iou_val, "dice": dice_val, "px_acc": acc_val}
