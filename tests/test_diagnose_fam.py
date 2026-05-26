"""Tests for spectrafan.diagnose_fam."""

from __future__ import annotations


def test_module_imports() -> None:
    """diagnose_fam imports cleanly and exposes diagnose_run."""
    from spectrafan import diagnose_fam

    assert hasattr(diagnose_fam, "diagnose_run")
