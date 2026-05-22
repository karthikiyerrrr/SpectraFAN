"""Augmentation pipeline for (image, mask) pairs.

Geometric transforms (flip, rotation, zoom) apply jointly to image and mask via
``torchvision.transforms.v2`` + ``tv_tensors``. ``tv_tensors.Mask`` dispatches
to nearest-neighbor interpolation, so masks stay binary through rotation/zoom.

Additive Gaussian noise is applied to the image only.

The augmentation callable is a top-level class (not a local closure) so it is
picklable for DataLoader workers under macOS spawn.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2

Transform = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


class _TrainTransform:
    """Joint (image, mask) augmentation: flip / rotate / zoom / Gaussian noise."""

    def __init__(
        self,
        p_flip: float,
        max_rot_deg: float,
        zoom_range: tuple[float, float],
        noise_sigma: float,
    ) -> None:
        if noise_sigma < 0.0:
            raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")
        self.noise_sigma = noise_sigma
        self.geom = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=p_flip),
                v2.RandomVerticalFlip(p=p_flip),
                v2.RandomRotation(degrees=max_rot_deg),
                v2.RandomAffine(degrees=0, scale=zoom_range),
            ]
        )

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        img_tv = tv_tensors.Image(image)
        msk_tv = tv_tensors.Mask(mask)
        img_tv, msk_tv = self.geom(img_tv, msk_tv)
        if self.noise_sigma > 0.0:
            img_tv = img_tv + torch.randn_like(img_tv) * self.noise_sigma
        return img_tv.as_subclass(Tensor), msk_tv.as_subclass(Tensor).float()


def train_transforms(
    p_flip: float = 0.5,
    max_rot_deg: float = 15.0,
    zoom_range: tuple[float, float] = (0.9, 1.1),
    noise_sigma: float = 0.01,
) -> Transform:
    """Return a joint (image, mask) augmentation callable.

    Magnitudes are the project defaults; the paper specifies the menu but not
    the magnitudes (recorded in the spec).
    """
    return _TrainTransform(p_flip, max_rot_deg, zoom_range, noise_sigma)


def eval_transforms() -> Transform | None:
    """No-op for eval/test loaders."""
    return None
