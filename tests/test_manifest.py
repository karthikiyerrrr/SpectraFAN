"""Tests for spectrafan.manifest (run.json reader/writer)."""

from __future__ import annotations

import pytest

from spectrafan.manifest import infer_stages, load_manifest, write_manifest


def _touch(run_dir, name, body="x"):
    (run_dir / name).write_text(body)


def test_fresh_write_creates_manifest(tmp_path):
    _touch(tmp_path, "metrics.parquet")
    m = write_manifest(
        tmp_path,
        "train",
        model="fanetmini",
        dataset="temimagenet",
        headline={"best_val_iou": 0.84, "best_epoch": 147},
    )
    assert (tmp_path / "run.json").is_file()
    assert m["schema_version"] == 1
    assert m["run_id"] == tmp_path.name
    assert m["stages"] == ["train"]
    assert m["model"] == "fanetmini"
    assert m["dataset"] == "temimagenet"
    assert m["headline"]["best_val_iou"] == 0.84
    assert "metrics.parquet" in m["artifacts"]
    assert "run.json" not in m["artifacts"]
    assert m["created"] and m["updated"]


def test_stage_union_is_idempotent(tmp_path):
    write_manifest(tmp_path, "train")
    m = write_manifest(tmp_path, "train")
    assert m["stages"] == ["train"]


def test_headline_merges_across_stages(tmp_path):
    write_manifest(tmp_path, "train", model="fanetmini", headline={"best_val_iou": 0.84})
    _touch(tmp_path, "test_metrics.json")
    m = write_manifest(tmp_path, "predict", headline={"test_iou": 0.845})
    assert m["stages"] == ["train", "predict"]
    assert m["headline"] == {"best_val_iou": 0.84, "test_iou": 0.845}
    assert m["model"] == "fanetmini"  # preserved when later call omits it


def test_stages_kept_in_canonical_order(tmp_path):
    write_manifest(tmp_path, "predict")
    m = write_manifest(tmp_path, "train")
    assert m["stages"] == ["train", "predict"]


def test_unknown_stage_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown stage"):
        write_manifest(tmp_path, "bogus")


def test_infer_stages_from_artifacts(tmp_path):
    _touch(tmp_path, "metrics.parquet")
    _touch(tmp_path, "fam_diagnosis.json")
    assert infer_stages(tmp_path) == ["train", "diagnose"]


def test_load_manifest_prefers_file(tmp_path):
    write_manifest(tmp_path, "train", model="fanetmini")
    loaded = load_manifest(tmp_path)
    assert loaded["model"] == "fanetmini"
    assert loaded["stages"] == ["train"]


def test_load_manifest_synthesizes_for_legacy_run(tmp_path):
    _touch(tmp_path, "test_metrics.json")
    _touch(tmp_path, "predictions.npz")
    loaded = load_manifest(tmp_path)
    assert loaded["stages"] == ["predict"]
    assert loaded["headline"] == {}
    assert loaded["model"] is None
    assert "test_metrics.json" in loaded["artifacts"]
