"""The AF2-IG device-trunk leg's classifier, and the two mutations that keep it honest.

The leg is registered as GAP-evidenced: its residual is a real, root-caused bfloat16 floor, so
the gate cannot ask it for a PASS. What the gate can ask is whether the residual is still the
SAME floor, and that question is the whole leg. A classifier that answers "yes" too easily is a
gate that cannot fail, so these tests come in two halves.

The card-bound half is committed as artifacts rather than re-run here: an un-mutated device run
on a second card (bit-identical to the committed record on all 94 taps and all 6 scalars, qb1
cards 3 and 0, pass 13) and the two `tap_gate.py --device --mutate` runs. Every one of the
classifier's three substantive conditions fires independently on both mutations, which is the
property that matters -- an allowlist keyed on tap names alone would swallow a mutation that
only worsens the taps already failing.

The card-free half drives the decision rule directly, one condition per test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "af2_port"))

import device_floor as df  # noqa: E402

ARTIFACTS = REPO / "scripts" / "af2_port" / "parity_artifacts" / "laczc128_b80"
COMMITTED = REPO / "docs" / "implementation-parity-data" / "af2ig-trunk-device.json"


def _committed() -> dict:
    if not COMMITTED.exists():
        pytest.skip("committed device floor record absent")
    return json.loads(COMMITTED.read_text())


def _report(taps, scalars, **kw):
    """A minimal report in tap_gate's shape: (tap, verdict, envelope_ratio) triples."""
    rows = [{"tap": t, "verdict": v, "envelope_ratio": r} for t, v, r in taps]
    return {"rows": rows, "taps_scored": kw.pop("taps_scored", len(rows)),
            "scalars": [{"scalar": s, "verdict": v, "delta": d} for s, v, d in scalars],
            **kw}


FLOOR = _report([("a", "FAIL", 4.0), ("b", "FAIL", 12.0), ("c", "PASS", None)],
                [("plddt", "FAIL", 0.0028), ("i_ptm", "PASS", 0.0026)])


def test_the_same_residual_is_a_gap():
    verdict, detail = df.af2ig_device_floor_verdict(FLOOR, FLOOR)
    assert verdict == "GAP"
    assert "committed bf16 floor" in detail


def test_a_new_failing_tap_fails():
    live = _report([("a", "FAIL", 4.0), ("b", "FAIL", 12.0), ("c", "FAIL", 1.7)],
                   [("plddt", "FAIL", 0.0028), ("i_ptm", "PASS", 0.0026)])
    verdict, detail = df.af2ig_device_floor_verdict(live, FLOOR)
    assert verdict == "FAIL" and "new failing tap c" in detail


def test_a_widening_ratio_on_an_already_failing_tap_fails():
    """The condition that makes the name allowlist honest: a mutation is allowed to be caught
    here and nowhere else."""
    ok = _report([("a", "FAIL", 4.0 * df.DEFAULT_TOL), ("b", "FAIL", 12.0), ("c", "PASS", None)],
                 [("plddt", "FAIL", 0.0028), ("i_ptm", "PASS", 0.0026)])
    assert df.af2ig_device_floor_verdict(ok, FLOOR)[0] == "GAP"
    bad = _report([("a", "FAIL", 4.0 * df.DEFAULT_TOL * 1.001), ("b", "FAIL", 12.0),
                   ("c", "PASS", None)],
                  [("plddt", "FAIL", 0.0028), ("i_ptm", "PASS", 0.0026)])
    verdict, detail = df.af2ig_device_floor_verdict(bad, FLOOR)
    assert verdict == "FAIL" and "ratio" in detail


def test_a_scalar_drift_fails():
    live = _report([("a", "FAIL", 4.0), ("b", "FAIL", 12.0), ("c", "PASS", None)],
                   [("plddt", "FAIL", 0.0028 * 2), ("i_ptm", "PASS", 0.0026)])
    assert df.af2ig_device_floor_verdict(live, FLOOR)[0] == "FAIL"
    live = _report([("a", "FAIL", 4.0), ("b", "FAIL", 12.0), ("c", "PASS", None)],
                   [("plddt", "FAIL", 0.0028), ("i_ptm", "FAIL", 0.0026)])
    assert "new failing scalar i_ptm" in df.af2ig_device_floor_verdict(live, FLOOR)[1]


def test_an_unadjudicated_miss_cannot_be_excused():
    """`--no-envelope` (or a scorer that never reached the float32 arm) leaves no ratio to
    bound, so there is nothing to compare and the floor claim is unsupported."""
    live = _report([("a", "FAIL", None), ("b", "FAIL", 12.0), ("c", "PASS", None)],
                   [("plddt", "FAIL", 0.0028), ("i_ptm", "PASS", 0.0026)])
    assert df.af2ig_device_floor_verdict(live, FLOOR)[0] == "FAIL"


def test_a_vanished_tap_fails():
    live = dict(FLOOR, not_implemented=["structure_module#3/traj"])
    assert df.af2ig_device_floor_verdict(live, FLOOR)[0] == "FAIL"
    live = _report([("a", "FAIL", 4.0), ("b", "FAIL", 12.0)],
                   [("plddt", "FAIL", 0.0028), ("i_ptm", "PASS", 0.0026)], taps_scored=2)
    assert df.af2ig_device_floor_verdict(live, dict(FLOOR, taps_scored=3))[0] == "FAIL"


def test_the_reports_own_verdict_field_is_never_read():
    """`tap_gate.py --mutate` inverts its own verdict field, so a classifier keyed on it would
    call every mutation a pass and every clean run a failure."""
    assert df.af2ig_device_floor_verdict(dict(FLOOR, verdict="FAIL"), FLOOR)[0] == "GAP"
    off = _report([("a", "FAIL", 40.0), ("b", "FAIL", 12.0), ("c", "PASS", None)],
                  [("plddt", "FAIL", 0.0028), ("i_ptm", "PASS", 0.0026)], verdict="PASS")
    assert df.af2ig_device_floor_verdict(off, FLOOR)[0] == "FAIL"


def test_an_absent_prerequisite_stays_a_gap_and_an_error_stays_an_error():
    assert df.af2ig_device_floor_verdict({"verdict": "GAP", "error": "no card"}, FLOOR)[0] == "GAP"
    assert df.af2ig_device_floor_verdict({"error": "boom"}, FLOOR)[0] == "ERROR"
    assert df.af2ig_device_floor_verdict({"rows": []}, FLOOR)[0] == "NO-DATA"


@pytest.mark.parametrize("name", ["device_mutate_extra_const", "device_mutate_block_order"])
def test_the_real_mutations_are_caught_by_every_condition(name):
    """Not just FAIL: each of the three substantive conditions has to fire on its own, or the
    control only proves the allowlist works and says nothing about the ratio bounds."""
    path = ARTIFACTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    committed, live = _committed(), json.loads(path.read_text())
    assert df.af2ig_device_floor_verdict(live, committed)[0] == "FAIL"

    floor_taps, floor_scalars = df._failing_taps(committed), df._failing_scalars(committed)
    live_taps, live_scalars = df._failing_taps(live), df._failing_scalars(live)
    assert set(live_taps) - set(floor_taps), "no new failing tap: the name condition is silent"
    worst_ratio = max(live_taps[t] / floor_taps[t] for t in floor_taps
                      if live_taps.get(t) is not None)
    assert worst_ratio > df.DEFAULT_TOL, "the ratio condition alone would not catch this"
    worst_scalar = max(live_scalars[s] / floor_scalars[s] for s in floor_scalars
                       if live_scalars.get(s) is not None)
    assert worst_scalar > df.DEFAULT_TOL, "the scalar condition alone would not catch this"


def test_a_second_card_reproduces_the_committed_floor():
    path = ARTIFACTS / "device_trunk_complex_card0.json"
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    committed, live = _committed(), json.loads(path.read_text())
    assert df.af2ig_device_floor_verdict(live, committed)[0] == "GAP"
    # Bit-identical, which is why DEFAULT_TOL's floor of 1.10 is what binds rather than a
    # measured cross-card spread.
    rows = {r["tap"]: r for r in live["rows"]}
    assert all(rows[r["tap"]].get("pcc") == r.get("pcc") for r in committed["rows"])
