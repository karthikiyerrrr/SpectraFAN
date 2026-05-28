"""Model architectures and the registries that make them pluggable."""

from __future__ import annotations

from torch import nn

from spectrafan.config import DataConfig, ModelConfig
from spectrafan.models import (
    atomsegnet as _atomsegnet,  # noqa: F401  (imports register architectures)
)
from spectrafan.models import (
    fam as _fam,  # noqa: F401  (belt-and-suspenders: unet imports fam, but state the dependency)
)
from spectrafan.models import (
    skip_transforms as _skip_transforms,  # noqa: F401  (imports register "identity")
)
from spectrafan.models import unet as _unet  # noqa: F401  (imports register architectures)
from spectrafan.models._registries import MODEL_REGISTRY, SKIP_TRANSFORM_REGISTRY
from spectrafan.models.atomsegnet import AtomSegNet
from spectrafan.models.fam import FAMComplex
from spectrafan.models.unet import FANet, FANetMini


def build_model(model_cfg: ModelConfig, data_cfg: DataConfig) -> nn.Module:
    """Construct the architecture named by ``model_cfg.name`` via the registry."""
    return MODEL_REGISTRY.build(model_cfg.name, model_cfg, data_cfg)


__all__ = [
    "AtomSegNet",
    "FAMComplex",
    "FANet",
    "FANetMini",
    "MODEL_REGISTRY",
    "SKIP_TRANSFORM_REGISTRY",
    "build_model",
]
