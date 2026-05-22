"""Tests for spectrafan.losses."""

from __future__ import annotations

import torch

from spectrafan.losses import BCEDiceLoss


def test_bce_dice_zero_on_perfect_pred() -> None:
    """Saturated-correct logits yield near-zero combined loss."""
    torch.manual_seed(0)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    # +100 logit where target=1, -100 logit where target=0 -> sigmoid ~ 1 or ~ 0 respectively.
    logits = 100.0 * (2.0 * target - 1.0)
    loss = BCEDiceLoss()(logits, target)
    assert loss.item() < 1e-3


def test_bce_dice_high_on_inverted_pred() -> None:
    """Saturated-wrong logits yield large combined loss."""
    torch.manual_seed(0)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    logits = -100.0 * (2.0 * target - 1.0)
    loss = BCEDiceLoss()(logits, target)
    assert loss.item() > 1.0


def test_bce_dice_weighted_sum_matches_hand_computation() -> None:
    """Combined loss equals ce_weight * bce + dice_weight * dice_loss on a fixed input."""
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 4)
    target = (torch.rand(2, 1, 4, 4) > 0.5).float()

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    p = torch.sigmoid(logits)
    smooth = 1.0
    num = 2.0 * (p * target).sum() + smooth
    den = p.sum() + target.sum() + smooth
    dice_loss = 1.0 - num / den
    expected = 0.5 * bce + 0.5 * dice_loss

    loss = BCEDiceLoss(ce_weight=0.5, dice_weight=0.5, smooth=smooth)(logits, target)
    assert torch.isclose(loss, expected, atol=1e-6)
