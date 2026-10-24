"""Which folds the l1-budget leg runs on each part. Host-only: the folds are stubbed.

fkg (~130 tokens) never engages the trimul T1 path, so the leg also folds the size ladder's
512-token input at the native grid and at every smaller real part grid on the same side of
tenstorrent._apply_grid_thresholds' small-grid line. A grid across that line takes Wormhole's chunk
sizes by design, so 8x8 forced on a Blackhole card is not a part any user runs.
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


def _wide(gate, monkeypatch, phys):
    return [c for c in _folds(gate, monkeypatch, phys) if c[2] == gate.L1_BUDGET_WIDE_DATA.name]


def test_p150a_folds_the_512_token_target_at_native_and_the_p300c_grid(gate, monkeypatch):
    assert _wide(gate, monkeypatch, (13, 10)) == [("512-native", None, "cdk2x2_512.yaml"),
                                                  ("512-11x10", (11, 10), "cdk2x2_512.yaml")]
    assert gate.L1_BUDGET_WIDE_DATA.exists()


def test_p300c_has_no_smaller_blackhole_grid_and_skips_it(gate, monkeypatch):
    assert _wide(gate, monkeypatch, (11, 10)) == []


def test_galaxy_folds_it_at_native_and_8x8(gate, monkeypatch):
    assert _wide(gate, monkeypatch, (8, 9)) == [("512-native", None, "cdk2x2_512.yaml"),
                                                ("512-8x8", (8, 8), "cdk2x2_512.yaml")]


def test_wormhole_skips_it_because_native_is_8x8(gate, monkeypatch):
    assert all(c[2] == gate.L1_BUDGET_DATA.name for c in _folds(gate, monkeypatch, (8, 8)))
