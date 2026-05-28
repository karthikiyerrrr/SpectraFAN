"""Non-FAM skip transforms.

The ``identity`` transform is a channel-preserving pass-through used to build the
no-FAM baseline arm of a FAM ablation: the same backbone with the FAM removed from
every skip. Lives apart from fam.py because it is not a frequency module.
"""

from __future__ import annotations

from torch import nn

from spectrafan.models._registries import SKIP_TRANSFORM_REGISTRY
from spectrafan.models.fam import ConvKind


@SKIP_TRANSFORM_REGISTRY.register("identity")
def _build_identity(channels: int, conv_kind: ConvKind = "depthwise") -> nn.Module:
    """Pass-through skip. Accepts (channels, conv_kind) to match the registry's
    call signature; both are ignored."""
    return nn.Identity()
