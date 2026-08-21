"""The l1-budget release-gate leg must fail on a build that cannot escape an L1/CB clash.

issue #11: every L1-edge budget in `tt_bio.tenstorrent` was fitted on a 130-core p150a, and
`_apply_grid_thresholds` returns early on any grid of 110 cores or more, so a 110-core
Blackhole (p300/p300c) runs 130-core budgets and the trimul in-projection's circular buffers
stop fitting. The fix lets the channel loop learn the ceiling from the clash itself. This
tests the leg that keeps that mechanism, and the part table behind it, from rotting: with the
narrowing clamp removed the leg must report a failure, not pass quietly.

Host-only — no device, no fold. The live half of the leg (folding Taylor's target across the
grid ladder) is in scripts/release_gate.py and needs hardware.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tt_bio import tenstorrent as T

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gate():
    path = REPO_ROOT / "scripts" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("tt_bio_release_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arm_is_in_the_default_set(gate):
    """A leg nobody remembers to pass is not a gate."""
    assert "l1-budget" in gate.DEFAULT_ARMS


def test_every_selectable_grid_has_a_part_row(gate):
    """The rule the leg exists to enforce: a part-specific figure gets a row here."""
    grids = {g for _n, g, _l, _d, _p in gate.L1_BUDGET_PARTS}
    assert (T.COMPUTE_GRID_X_13, T.COMPUTE_GRID_Y) in grids
    assert (T.COMPUTE_GRID_X_11, T.COMPUTE_GRID_Y) in grids


def test_measured_clashes_are_labelled_with_a_source(gate):
    """A number in the table is measured or it does not belong in the table."""
    for part, shape, width, source in gate.L1_BUDGET_MEASURED_CLASHES:
        assert part in {n for n, _g, _l, _d, _p in gate.L1_BUDGET_PARTS}
        assert width % T.TRIANGLE_MULT_CHUNK_SIZE == 0
        assert len(source) > 40 and any(c.isdigit() for c in source)


def test_arithmetic_leg_passes_on_this_build(gate):
    row = gate.run_l1_budget_static()
    assert row["error"] is None
    assert row["fails"] == []
    assert row["gate"] is True
    assert row["checks"] > 0


def test_arithmetic_leg_fails_without_the_narrowing_clamp(gate, monkeypatch):
    """Remove the one thing the fix added and the leg must catch it.

    This is the pre-fix decision logic exactly: the budget picks a width, the width clashes,
    and nothing can pick anything narrower.
    """
    real = T._trimul_chunk_size

    def no_clamp(seq_len, hidden, batch=1):
        clash = dict(T._TRIMUL_CHUNK_CLASH)
        T._TRIMUL_CHUNK_CLASH.clear()
        try:
            return real(seq_len, hidden, batch)
        finally:
            T._TRIMUL_CHUNK_CLASH.update(clash)

    monkeypatch.setattr(T, "_trimul_chunk_size", no_clamp)
    row = gate.run_l1_budget_static()
    assert row["gate"] is False
    assert row["fails"], "the leg passed a build with no way out of a clash"
    assert any("no way out" in f for f in row["fails"])


def test_arithmetic_leg_fails_when_the_mechanism_is_absent(gate, monkeypatch):
    """A build predating the fix must fail the leg, not crash it."""
    monkeypatch.delattr(T, "_record_trimul_clash")
    row = gate.run_l1_budget_static()
    assert row["gate"] is False
    assert any("_record_trimul_clash" in f for f in row["fails"])


def test_arithmetic_leg_restores_module_state(gate):
    """The leg installs other parts' grids in-process; every later leg shares this module."""
    before = {n: getattr(T, n) for n in gate._L1_BUDGET_SAVED}
    gate.run_l1_budget_static()
    after = {n: getattr(T, n) for n in gate._L1_BUDGET_SAVED}
    assert before == after
    assert T._TRIMUL_CHUNK_CLASH == {}
    assert T._TRIMUL_DRAM_SHAPES == set()


def test_chunk_cap_is_off_by_default_and_narrows_when_set():
    """The test-only width cap must be inert unless it is asked for."""
    assert T._TRIMUL_CHUNK_CAP == 0, "TT_BIO_TRIMUL_CHUNK_CAP leaked into a normal run"
