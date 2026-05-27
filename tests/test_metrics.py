"""Tests for spectrafan.metrics."""

from __future__ import annotations

import torch

from spectrafan.training.metrics import RunningMetrics, dice, iou, pixel_accuracy


def test_iou_hand_computed() -> None:
    """Intersection 4, union 12 -> IoU 4/12 = 1/3."""
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, :2, :] = 1.0  # 8 positive pixels
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :, :2] = 1.0  # 8 positive pixels; overlap is the 2x2 top-left = 4
    # intersection = 4, union = 8 + 8 - 4 = 12
    assert torch.isclose(iou(pred, target), torch.tensor(4.0 / 12.0), atol=1e-6)


def test_iou_empty_union_returns_one() -> None:
    """All-zero pred and all-zero target -> IoU = 1.0 (nothing to find, found nothing)."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    assert torch.isclose(iou(pred, target), torch.tensor(1.0))


def test_dice_hand_computed() -> None:
    """2 * intersection / (|pred| + |target|) = 2*4 / (8+8) = 0.5."""
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, :2, :] = 1.0
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :, :2] = 1.0
    assert torch.isclose(dice(pred, target), torch.tensor(0.5), atol=1e-6)


def test_pixel_accuracy_hand_computed() -> None:
    """8 correct of 16 pixels -> 0.5."""
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, :2, :] = 1.0
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :, :2] = 1.0
    # Disagreement: top-right 2x2 (pred=1, target=0) + bottom-left 2x2 (pred=0, target=1) = 8 wrong.
    # Agreement: top-left 2x2 (both 1) + bottom-right 2x2 (both 0) = 8 correct.
    assert torch.isclose(pixel_accuracy(pred, target), torch.tensor(8.0 / 16.0), atol=1e-6)


def test_running_metrics_aggregates_by_num_over_den_not_mean_of_means() -> None:
    """RunningMetrics.compute() on two batches matches the metric on the concatenated input."""
    torch.manual_seed(0)
    logits_a = torch.randn(3, 1, 8, 8)
    target_a = (torch.rand(3, 1, 8, 8) > 0.5).float()
    logits_b = torch.randn(2, 1, 8, 8)
    target_b = (torch.rand(2, 1, 8, 8) > 0.5).float()

    rm = RunningMetrics()
    rm.update(logits_a, target_a)
    rm.update(logits_b, target_b)
    got = rm.compute()

    cat_logits = torch.cat([logits_a, logits_b], dim=0)
    cat_target = torch.cat([target_a, target_b], dim=0)
    cat_prob = torch.sigmoid(cat_logits)
    expected_iou = iou(cat_prob, cat_target).item()
    expected_dice = dice(cat_prob, cat_target).item()
    expected_acc = pixel_accuracy(cat_prob, cat_target).item()

    assert abs(got["iou"] - expected_iou) < 1e-6
    assert abs(got["dice"] - expected_dice) < 1e-6
    assert abs(got["px_acc"] - expected_acc) < 1e-6
