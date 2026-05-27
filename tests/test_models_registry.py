"""Tests for model + skip-transform registries and build_model."""

from __future__ import annotations

import pytest
import torch

from spectrafan.config import DataConfig, ModelConfig
from spectrafan.models import MODEL_REGISTRY, SKIP_TRANSFORM_REGISTRY, build_model
from spectrafan.models.unet import FANet, FANetMini


def test_skip_transform_registered():
    assert "fam_complex" in SKIP_TRANSFORM_REGISTRY.keys()


def test_model_registry_has_both():
    assert set(MODEL_REGISTRY.keys()) == {"fanet", "fanetmini"}


def test_build_model_dispatches():
    data = DataConfig(in_channels=1)
    assert isinstance(build_model(ModelConfig(name="fanet"), data), FANet)
    assert isinstance(build_model(ModelConfig(name="fanetmini"), data), FANetMini)


def test_build_model_unknown_lists_keys():
    with pytest.raises(ValueError, match=r"registered:"):
        build_model(ModelConfig(name="nope"), DataConfig())


def test_fanet_uses_selected_skip_transform():
    from spectrafan.models.fam import FAMComplex

    m = FANetMini(in_channels=1, skip_transform="fam_complex")
    assert all(isinstance(f, FAMComplex) for f in m.fams)


def test_fanetmini_forward_shape():
    m = FANetMini(in_channels=1)
    out = m(torch.randn(1, 1, 64, 64))
    assert out.shape == (1, 1, 64, 64)
