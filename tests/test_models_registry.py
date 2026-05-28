"""Tests for model + skip-transform registries and build_model."""

from __future__ import annotations

import pytest
import torch

from spectrafan.config import DataConfig, ModelConfig
from spectrafan.models import MODEL_REGISTRY, SKIP_TRANSFORM_REGISTRY, build_model
from spectrafan.models.unet import FANet, FANetMini


def test_skip_transform_registered():
    assert "fam_complex" in SKIP_TRANSFORM_REGISTRY.keys()


def test_model_registry_has_all_four():
    assert set(MODEL_REGISTRY.keys()) == {"fanet", "fanetmini", "atomsegnet", "famsegnet"}


def test_build_model_dispatches_atomsegnet_and_famsegnet():
    from spectrafan.models.atomsegnet import AtomSegNet
    from spectrafan.models.fam import FAMComplex

    data = DataConfig(in_channels=3)
    atom = build_model(ModelConfig(name="atomsegnet"), data)
    fam = build_model(ModelConfig(name="famsegnet"), data)
    assert isinstance(atom, AtomSegNet) and isinstance(fam, AtomSegNet)
    # FAM presence is fixed by the model name, regardless of ModelConfig.skip_transform default.
    assert not any(isinstance(t, FAMComplex) for t in atom.skip_transforms)
    assert all(isinstance(t, FAMComplex) for t in fam.skip_transforms)


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


def test_identity_skip_transform_registered_and_passthrough():
    assert "identity" in SKIP_TRANSFORM_REGISTRY.keys()
    t = SKIP_TRANSFORM_REGISTRY.build("identity", channels=64, conv_kind="depthwise")
    x = torch.randn(2, 64, 16, 16)
    assert torch.equal(t(x), x)
