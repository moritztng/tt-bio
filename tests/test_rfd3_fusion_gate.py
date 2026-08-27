"""The rfd3-fusion release-gate arm must fail when either fusion lever stops firing.

Both RFD3 diffusion fusion levers are size-conditioned and both decline silently: region 1
(`RFD3_SOFTMAX_PV_FUSED`) needs the fused softmax kernel to engage at the gathered key width,
region 2 (`RFD3_FC1_SPLIT_SILU`) needs the pair Transition's third L1 resident to leave the chunk
height alone. Neither raises when it declines, so a regression that quietly stops one of them
costs seconds and nothing says so. The arm censuses two fixtures and asserts opposite verdicts:
must-serve at R4 (685 tokens), must-decline at 40 tokens.

Host-only. The live half -- two censused designs on the device -- is in scripts/release_gate.py.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gate():
    path = REPO_ROOT / "scripts" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("tt_bio_release_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PV_DECLINED_AT_4 = 108   # measured, not predicted -- see _rfd3_fusion_expected's docstring


def _rows(pv_served, pv_declined, fc1_served, fc1_declined,
          pv_rejects=None, fc1_rejects=None, resolved="True"):
    return {
        "RFD3_SOFTMAX_PV_FUSED": {
            "resolved": resolved, "served": pv_served, "declined": pv_declined,
            "rejects": pv_rejects if pv_rejects is not None else {}},
        "RFD3_FC1_SPLIT_SILU": {
            "resolved": resolved, "served": fc1_served, "declined": fc1_declined,
            "rejects": fc1_rejects if fc1_rejects is not None else {}},
    }


def _healthy_r4(gate, steps=4):
    e = gate._rfd3_fusion_expected(steps)
    return _rows(e["pv_served"], PV_DECLINED_AT_4, e["fc1_served"], e["fc1_declined"],
                 pv_rejects={gate.RFD3_FUSION_PV_DECLINE: PV_DECLINED_AT_4},
                 fc1_rejects={gate.RFD3_FUSION_FC1_DECLINE: e["fc1_declined"],
                              "rows=64 w=704 hidden=256 no-pinned-config": 1,
                              "rows=45 w=704 hidden=256 no-pinned-config": 1})


def test_arm_is_in_the_default_set(gate):
    """A leg nobody remembers to pass is not a gate."""
    assert "rfd3-fusion" in gate.DEFAULT_ARMS


def test_both_fixtures_exist_and_are_tracked(gate):
    """An arm whose fixture is somebody's untracked scratch file passes once, on one host."""
    for spec in (gate.RFD3_FUSION_R4_SPEC, gate.RFD3_FUSION_SMALL_SPEC):
        assert spec.exists(), spec
        rc = subprocess.run(["git", "ls-files", "--error-unmatch", str(spec)],
                            cwd=REPO_ROOT, capture_output=True).returncode
        assert rc == 0, f"{spec} is not git-tracked"


def test_both_levers_are_registered_in_the_census(gate):
    """The arm reads the shared census, so the levers have to be in its table."""
    flags = gate.lever_census_flags()
    for flag in gate.RFD3_FUSION_FLAGS:
        assert flag in flags


def test_expected_census_matches_the_three_folds_that_measured_it(gate):
    """The formula, against the 3-step, 4-step and 200-step folds that measured it."""
    assert gate._rfd3_fusion_expected(3) == {"pv_served": 18, "fc1_served": 110,
                                             "fc1_declined": 10}
    assert gate._rfd3_fusion_expected(4) == {"pv_served": 27, "fc1_served": 154,
                                             "fc1_declined": 14}
    e = gate._rfd3_fusion_expected(200)
    assert (e["pv_served"], e["fc1_served"], e["fc1_declined"]) == (1791, 8778, 798)


def test_a_healthy_r4_census_reports_nothing(gate):
    assert gate._rfd3_fusion_findings(_healthy_r4(gate), 4, serve=True) == []


def test_a_dark_lever_at_r4_fails_the_arm(gate):
    """The negative control: the exact regression the arm exists to catch.

    Zero served with the flag still resolving True is a lever that stopped firing without
    raising -- the only symptom is the clock, which no gate arm reads.
    """
    rows = _healthy_r4(gate)
    rows["RFD3_SOFTMAX_PV_FUSED"]["served"] = 0
    rows["RFD3_SOFTMAX_PV_FUSED"]["declined"] = 135
    findings = gate._rfd3_fusion_findings(rows, 4, serve=True)
    assert findings and any("dark" in f for f in findings)


def test_a_partly_dark_lever_fails_too(gate):
    """`served > 0` would pass this. One call of twenty-seven is the same defect."""
    rows = _healthy_r4(gate)
    rows["RFD3_SOFTMAX_PV_FUSED"]["served"] = 1
    assert gate._rfd3_fusion_findings(rows, 4, serve=True)


def test_declining_at_the_gathered_key_width_fails(gate):
    """The width assertion: that site is the whole win, so it must never decline at R4."""
    rows = _healthy_r4(gate)
    rows["RFD3_SOFTMAX_PV_FUSED"]["rejects"] = {
        gate.RFD3_FUSION_PV_DECLINE: 216,
        "value residency does not fit L1 " + gate.RFD3_FUSION_PV_SERVED_KEY: 27}
    findings = gate._rfd3_fusion_findings(rows, 4, serve=True)
    assert findings and any("gathered key width" in f for f in findings)


def test_a_new_region_2_decline_clause_fails(gate):
    """A clause that is not the gated one and not a first-call row is a changed guard."""
    rows = _healthy_r4(gate)
    rows["RFD3_FC1_SPLIT_SILU"]["rejects"] = dict(
        rows["RFD3_FC1_SPLIT_SILU"]["rejects"], **{"w=704 no-chunked-site": 2})
    findings = gate._rfd3_fusion_findings(rows, 4, serve=True)
    assert findings and any("unexpected R4 decline clause" in f for f in findings)


def test_the_first_call_rows_alone_do_not_fail(gate):
    """`no-pinned-config` is one call per chunk shape that still took the split."""
    assert gate._rfd3_fusion_findings(_healthy_r4(gate), 4, serve=True) == []


def test_losing_the_chunk_height_clause_fails(gate):
    """Region 2's gated clause going missing means the guard stopped being reached."""
    rows = _healthy_r4(gate)
    del rows["RFD3_FC1_SPLIT_SILU"]["rejects"][gate.RFD3_FUSION_FC1_DECLINE]
    assert gate._rfd3_fusion_findings(rows, 4, serve=True)


def test_the_flag_being_off_is_reported_as_such(gate):
    """A count read with the default off measures the harness, not the guard."""
    rows = _healthy_r4(gate)
    rows["RFD3_FC1_SPLIT_SILU"]["resolved"] = "False"
    findings = gate._rfd3_fusion_findings(rows, 4, serve=True)
    assert findings and any("not 'True'" in f for f in findings)


def test_an_unregistered_lever_is_reported_as_such(gate):
    """No census row is not a pass. It is the instrument missing."""
    rows = _healthy_r4(gate)
    del rows["RFD3_SOFTMAX_PV_FUSED"]
    assert gate._rfd3_fusion_findings(rows, 4, serve=True)


def test_the_small_leg_wants_zero_served(gate):
    zero = _rows(0, 3, 0, 4)
    assert gate._rfd3_fusion_findings(zero, 4, serve=False) == []


def test_the_small_leg_fails_if_a_lever_starts_firing(gate):
    """Firing outside the shapes the levers were proven bit-exact on is an accuracy defect."""
    rows = _rows(0, 3, 6, 4)
    findings = gate._rfd3_fusion_findings(rows, 4, serve=False)
    assert findings and any("wrong structure" in f for f in findings)
