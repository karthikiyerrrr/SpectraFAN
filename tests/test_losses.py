"""Tests for spectrafan.losses."""

from __future__ import annotations

import torch

from spectrafan.training.losses import BCEDiceLoss


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


def test_bce_dice_matches_hardcoded_expected_value() -> None:
    """Loss on a fixed 2x1x2x2 input matches a hand-computed scalar.

    Anchors the formula to an externally-derived number so an accidental
    reformulation (e.g. smooth in the wrong place) cannot pass the test by
    being self-consistent with the implementation.
    """
    # Construct a deterministic 2-batch input by hand so the expected loss is computable.
    # logits chosen so sigmoid(logits) is a clean set of values:
    #   row 0: 0, 0, 2, -2  -> sigmoid ~ 0.5, 0.5, 0.8808, 0.1192
    #   row 1: 0, 2, -2, 0  -> sigmoid ~ 0.5, 0.8808, 0.1192, 0.5
    logits = torch.tensor(
        [
            [[[0.0, 0.0], [2.0, -2.0]]],
            [[[0.0, 2.0], [-2.0, 0.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 1.0]]],
        ]
    )
    # Hand-computation (with smooth=1.0):
    #   BCE per element via softplus identity bce(z, t) = max(z, 0) - z*t + log(1 + exp(-|z|))
    #   element BCEs (8 elements):
    #     (z=0,  t=1) -> 0.6931472
    #     (z=0,  t=0) -> 0.6931472
    #     (z=2,  t=1) -> 0.1269280
    #     (z=-2, t=0) -> 0.1269280
    #     (z=0,  t=0) -> 0.6931472
    #     (z=2,  t=1) -> 0.1269280
    #     (z=-2, t=0) -> 0.1269280
    #     (z=0,  t=1) -> 0.6931472
    #   mean BCE = 3.2802528 / 8 = 0.4100316
    #
    #   probabilities p:  0.5, 0.5, 0.8808, 0.1192,  0.5, 0.8808, 0.1192, 0.5  (sum = 4.0)
    #   targets:          1,   0,   1,      0,        0,   1,      0,      1    (sum = 4.0)
    #   p * target sum  = 0.5 + 0 + 0.8808 + 0 + 0 + 0.8808 + 0 + 0.5 = 2.7616
    #   dice num = 2 * 2.7616 + 1 = 6.5232
    #   dice den = 4.0 + 4.0 + 1   = 9.0
    #   dice_loss = 1 - 6.5232 / 9.0 = 0.27520
    #
    #   total = 0.5 * 0.4100316 + 0.5 * 0.27520 = 0.34261...
    expected = 0.34261580

    loss = BCEDiceLoss(ce_weight=0.5, dice_weight=0.5, smooth=1.0)(logits, target)
    assert abs(loss.item() - expected) < 1e-4, f"loss={loss.item():.7f} expected={expected}"
