"""Which folds the l1-budget leg runs on each part. Host-only: the folds are stubbed.

fkg (~130 tokens) never engages the trimul T1 path, so main 6e42015fc's grid-dependent 512-token
Protenix-v2 fold on Blackhole passed this leg. A part with more than 8x8 cores now also folds the
size ladder's 512-token input at its native grid and at 8x8.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gate():
    path = REPO_ROOT / "scripts" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("tt_bio_release_gate_l1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _folds(gate, monkeypatch, phys):
    calls = []

    def fold(label, grid, cap, keep, data=gate.L1_BUDGET_DATA):
        calls.append((label, grid, data.name))
        return {"model": f"l1-budget:{label}", "target": data.stem, "md5": "x", "gate": True, "error": None}
    monkeypatch.setattr(gate, "_l1_budget_physical_grid", lambda: phys)
    monkeypatch.setattr(gate, "_l1_budget_fold", fold)
    gate.run_l1_budget_fold(keep=False)
    return calls


def test_blackhole_folds_the_512_token_target_at_native_and_8x8(gate, monkeypatch):
    wide = [c for c in _folds(gate, monkeypatch, (11, 10)) if c[2] == gate.L1_BUDGET_WIDE_DATA.name]
    assert wide == [("512-native", None, "cdk2x2_512.yaml"), ("512-8x8", (8, 8), "cdk2x2_512.yaml")]
    assert gate.L1_BUDGET_WIDE_DATA.exists()


def test_wormhole_skips_it_because_native_is_8x8(gate, monkeypatch):
    assert all(c[2] == gate.L1_BUDGET_DATA.name for c in _folds(gate, monkeypatch, (8, 8)))
