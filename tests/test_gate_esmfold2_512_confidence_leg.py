"""The release gate's ESMFold2-Fast L=512 confidence leg.

Kabsch and the distance-matrix metric grade where the atoms went. On 2026-09-02 (05327d6ce)
ESMFold2-Fast placed cdk2x2_512 inside the seed floor while reading mean plDDT 0.021 below the
reference, and the esmfold2 rule, PASS-if-scored, could not see it. The leg's `max_plddt_delta`
grades that gap. The cases below are its negative controls, built from the two real records.
"""
import ast
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "full_parity_gate.py"
FIXTURE = (REPO / "docs/implementation-parity-data/ref-fixtures/esmfold2-fast/cdk2x2_512/"
           "nomsa_10cycle_100step_1sample_fp32_prod")
BAR = 0.005

# Production lever set on acbce1322, qb2 card 0, 2026-10-15.
TODAY = {"protein": "cdk2x2_512", "plddt_mean": {"ref_seed_spread": 0.00422,
                                                 "mean_abs_delta": 0.00140}}
# The record the gap was found in, 05327d6ce, 2026-09-02.
SEP02 = {"protein": "cdk2x2_512", "plddt_mean": {"ref_seed_spread": 0.00430,
                                                 "mean_abs_delta": 0.02096}}


def _rule():
    """The confidence check alone, without running the gate's module-level setup."""
    body = [n for n in ast.parse(GATE.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_esmfold2_confidence_gap"]
    assert body, "the gate no longer defines _esmfold2_confidence_gap"
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GATE), "exec"), ns)
    return ns["_esmfold2_confidence_gap"]


def test_todays_record_passes():
    assert _rule()([TODAY], BAR) == ""


def test_the_sep02_gap_is_reported():
    gap = _rule()([SEP02], BAR)
    assert "cdk2x2_512" in gap and "0.0210" in gap, gap


def test_a_record_without_a_plddt_block_is_not_a_pass():
    rec = copy.deepcopy(TODAY)
    rec.pop("plddt_mean")
    assert _rule()([rec], BAR)


def test_a_leg_without_a_bar_is_ungraded():
    """The other esmfold2 legs keep PASS-if-scored."""
    assert _rule()([SEP02], None) == ""


def test_the_bar_sits_above_the_reference_seed_spread_and_below_the_old_gap():
    assert TODAY["plddt_mean"]["ref_seed_spread"] < BAR < SEP02["plddt_mean"]["mean_abs_delta"]


def test_the_leg_is_registered_against_its_committed_fixture():
    src = GATE.read_text()
    assert 'Leg("esmfold2-fast-512-confidence"' in src, "the leg is not in LEGS"
    leg = src.split('Leg("esmfold2-fast-512-confidence"', 1)[1].split("\n    Leg(", 1)[0]
    assert f"max_plddt_delta={BAR}" in leg
    assert '"cdk2x2_512"' in leg and '"--loops", "10"' in leg and '"--steps", "100"' in leg
    assert "esmfold2-fast/cdk2x2_512/nomsa_10cycle_100step_1sample_fp32_prod" in leg
    meta = json.loads((FIXTURE / "meta.json").read_text())
    assert meta["settings"]["recycling_cycles"] == 10 and meta["settings"]["requested_steps"] == 100
    assert meta["envelope"]["settings"]["target_id"] == "cdk2x2_512"
    for s in (0, 1):
        assert (FIXTURE / "ref_fp32" / f"seed{s}.npz").exists(), f"seed{s} reference not committed"
