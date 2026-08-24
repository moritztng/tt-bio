"""The rf3-1024aa release-gate leg: wired, floored, and reading the number it claims to.

`scripts/release_gate.py`'s MODELS table scores every fold model on the same 117-residue
target. This leg is the one that folds 997 aa, and what it must not do is pass while the
harness under it stopped reporting a crystal distance -- an accuracy floor that cannot
fail is worse than no floor, because the release report says PASS.

Host-only: no card, no checkpoint. The live half (one device rollout of the 7EIP anchor)
is in scripts/release_gate.py.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gate():
    path = REPO_ROOT / "scripts" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("tt_bio_release_gate_rf3_1024aa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(*device_vs_xtal, metrics=True):
    """A stand-in accuracy_cell report carrying one device-vs-crystal reading per seed."""
    return {
        "vs_crystal": {
            "n_ca_compared": 966,
            "per_seed": [{"seed": i, "device_vs_xtal_A": v, "reference_vs_xtal_A": v + 0.04}
                         for i, v in enumerate(device_vs_xtal)],
        },
        "metrics": {"kabsch_rmsd": {"cross": {"mean": 0.6406}}} if metrics else None,
    }


def _row(gate, monkeypatch, tmp_path, report):
    """Run the leg with the cell stubbed out, so the test scores the leg's own reading."""
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)  # keeps the work dir out of the tree

    def fake_run_fold(cmd, timeout, **kw):
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(report))
        return 0, False

    monkeypatch.setattr(gate, "_run_fold", fake_run_fold)
    return gate.run_rf3_1024aa(keep=True)


def test_arm_is_in_the_default_set(gate):
    """A leg nobody remembers to pass is not a gate."""
    assert "rf3-1024aa" in gate.DEFAULT_ARMS


def test_the_reference_cache_it_reads_is_committed(gate):
    """The leg refuses to run without it, and the alternative is a multi-hour host
    reference nobody budgeted for -- so a rename of this path must fail here, not on a
    release host."""
    assert (gate.RF3_1024AA_REF_CACHE / f"seed{gate.RF3_1024AA_SEED}.npz").exists()


def test_the_measured_reading_is_the_committed_one(gate):
    """The floor's justification is a number in an artifact, not a memory of one."""
    rep = json.loads((REPO_ROOT / "perf" / "rf3" / "results" / "a0_7eip997.json").read_text())
    seed0 = next(r for r in rep["vs_crystal"]["per_seed"] if r["seed"] == gate.RF3_1024AA_SEED)
    assert seed0["device_vs_xtal_A"] == gate.RF3_1024AA_XTAL_MEASURED


def test_the_floor_keeps_about_2x_headroom(gate):
    """The discipline every MODELS floor uses: generous enough that a correct fold's seed
    spread cannot fail it, tight enough that a gross failure cannot clear it."""
    ratio = gate.RF3_1024AA_MAX_XTAL_A / gate.RF3_1024AA_XTAL_MEASURED
    assert 1.8 <= ratio <= 2.5


def test_the_measured_fold_passes(gate, monkeypatch, tmp_path):
    row = _row(gate, monkeypatch, tmp_path, _report(gate.RF3_1024AA_XTAL_MEASURED))
    assert row["gate"] and row["error"] is None
    assert row["xtal_a"] == gate.RF3_1024AA_XTAL_MEASURED
    assert row["n_ca"] == 966


def test_a_fold_past_the_floor_fails_and_says_by_how_much(gate, monkeypatch, tmp_path):
    row = _row(gate, monkeypatch, tmp_path, _report(gate.RF3_1024AA_MAX_XTAL_A + 0.5))
    assert not row["gate"]
    assert "from the crystal" in row["error"]


def test_the_worst_seed_governs(gate, monkeypatch, tmp_path):
    """Lengthening the seed list may only tighten this leg. A mean would let one good seed
    carry a bad one."""
    row = _row(gate, monkeypatch, tmp_path,
               _report(gate.RF3_1024AA_XTAL_MEASURED, gate.RF3_1024AA_MAX_XTAL_A + 2.0))
    assert not row["gate"]
    assert row["xtal_a"] == gate.RF3_1024AA_MAX_XTAL_A + 2.0


def test_a_report_with_no_crystal_block_fails(gate, monkeypatch, tmp_path):
    """The failure this test exists for: the cell stops emitting vs_crystal (a renamed
    ground_truth_ca.json, a changed key) and the leg has nothing to compare. It must fail,
    not pass on an absent number."""
    row = _row(gate, monkeypatch, tmp_path, {"metrics": None})
    assert not row["gate"] and "vs_crystal" in row["error"]


def test_a_pending_reference_still_gates_the_crystal(gate, monkeypatch, tmp_path):
    """X is evidence, the crystal is the floor: no X means no X column, not no verdict."""
    row = _row(gate, monkeypatch, tmp_path,
               _report(gate.RF3_1024AA_XTAL_MEASURED, metrics=False))
    assert row["gate"] and row["x_a"] is None
