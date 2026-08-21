"""The batch-position gate leg's verdict, checked against the numbers it exists for.

A gate arm that cannot fail is decoration. These cases pin the three ways this one has
to behave: the real 0.6.4 defect must fail it, the real post-fix run must pass it, and a
run where every fold collapsed to one constant must fail it too.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from release_gate import BATCH_POSITION_N, batch_position_failures

# What tt-bio 0.6.4 actually returned for three byte-identical CDK2 + ligand targets.
PRE_FIX = [
    {"pos": 1, "coords": "a" * 16, "affinity_pred_value": 0.648724,
     "affinity_probability_binary": 0.250316},
    {"pos": 2, "coords": "b" * 16, "affinity_pred_value": 0.722511,
     "affinity_probability_binary": 0.261044},
    {"pos": 3, "coords": "c" * 16, "affinity_pred_value": 0.687149,
     "affinity_probability_binary": 0.255877},
    {"pos": 4, "coords": "d" * 16, "affinity_pred_value": 0.697898,
     "affinity_probability_binary": 0.265459},
]

# perf/boltz2-affinity-batchpos/probe_fixed_n3x.json, the fix branch's own run.
POST_FIX = [
    {"pos": 1, "coords": "0310c9d3871eb24e", "affinity_pred_value": 0.652816,
     "affinity_probability_binary": 0.250316},
    {"pos": 2, "coords": "0310c9d3871eb24e", "affinity_pred_value": 0.652816,
     "affinity_probability_binary": 0.250316},
    {"pos": 3, "coords": "0310c9d3871eb24e", "affinity_pred_value": 0.652816,
     "affinity_probability_binary": 0.250316},
    {"pos": 4, "coords": "93feee2af0ffe8f3", "affinity_pred_value": 0.697898,
     "affinity_probability_binary": 0.265459},
]


def test_pre_fix_run_fails_the_leg():
    fails = batch_position_failures(PRE_FIX, BATCH_POSITION_N)
    assert fails, "the 0.6.4 defect must fail this leg"
    assert any("affinity_pred_value depends on batch position" in f for f in fails)
    assert any("coords depends on batch position" in f for f in fails)


def test_post_fix_run_passes_the_leg():
    assert batch_position_failures(POST_FIX, BATCH_POSITION_N) == []


def test_collapsed_run_fails_on_the_control():
    """Every target identical, control included: agreement for the wrong reason."""
    collapsed = [dict(POST_FIX[0]) for _ in range(BATCH_POSITION_N + 1)]
    fails = batch_position_failures(collapsed, BATCH_POSITION_N)
    assert fails and all("not discriminating" in f for f in fails)


def test_missing_control_fails():
    fails = batch_position_failures(POST_FIX[:BATCH_POSITION_N], BATCH_POSITION_N)
    assert any("agreement is unproven" in f for f in fails)


def test_short_run_fails():
    fails = batch_position_failures(POST_FIX[:1], BATCH_POSITION_N)
    assert fails and "only 1/3" in fails[0]
