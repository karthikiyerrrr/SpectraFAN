"""SpectraFAN - frequency-domain attention for crystallographic image segmentation."""

from spectrafan.config import RunConfig, load_config
from spectrafan.models import (
    MODEL_REGISTRY,
    SKIP_TRANSFORM_REGISTRY,
    FAMComplex,
    FANet,
    FANetMini,
    build_model,
)
from spectrafan.training.train import fit

__version__ = "0.0.1"

__all__ = [
    "FAMComplex",
    "FANet",
    "FANetMini",
    "MODEL_REGISTRY",
    "RunConfig",
    "SKIP_TRANSFORM_REGISTRY",
    "__version__",
    "build_model",
    "fit",
    "load_config",
]
