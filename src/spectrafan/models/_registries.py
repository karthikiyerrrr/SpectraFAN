"""Registry instances for models and skip-transforms.

A leaf module so model/fam modules can import the instances to self-register
without importing the ``spectrafan.models`` package (which imports them back).
"""

from __future__ import annotations

from torch import nn

from spectrafan.registry import Registry

MODEL_REGISTRY: Registry[nn.Module] = Registry("model")
SKIP_TRANSFORM_REGISTRY: Registry[nn.Module] = Registry("skip_transform")
