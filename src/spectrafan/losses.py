"""Combined BCE + Dice loss for binary segmentation.

BCE consumes logits via `binary_cross_entropy_with_logits` for numerical
stability. Dice is computed once over the whole batch (num/den summed over
B * 1 * H * W), which is the standard interpretation of `0.5 * CE + 0.5 * Dice`
in segmentation papers; per-image-Dice-then-average is a fallback if the
reproduction spec's accuracy disagrees with Table 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as nnf
from torch import Tensor, nn


class BCEDiceLoss(nn.Module):
    """0.5 * BCEWithLogitsLoss + 0.5 * SoftDiceLoss by default."""

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        bce = nnf.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        p = torch.sigmoid(logits)
        num = 2.0 * (p * target).sum() + self.smooth
        den = p.sum() + target.sum() + self.smooth
        dice_loss = 1.0 - num / den
        return self.ce_weight * bce + self.dice_weight * dice_loss
