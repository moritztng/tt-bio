"""Lock the two reducer defects found on 2026-09-17, both of which were silent.

A criterion that is computed, printed and never evaluated is worse than no criterion, and a rate
column that divides a whole-fold numerator by a per-call denominator reads 1,590,402 TFLOP/s
without anything noticing. Each test below fails on the pre-fix reducer.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "sweep2"


def _reduce(run, tmp):
    out = run / "budget.json"
    subprocess.run([sys.executable, str(HERE / "reduce.py"), str(run), "--out", str(out)],
                   check=True, capture_output=True)
    return json.loads(out.read_text())


def _fixture(tmp, mutate=None):
    """A copy of the archived session, optionally mutated. Reduction is pure CPU."""
    src = json.loads((RUN / "replay.json").read_text())
    if mutate:
        mutate(src)
    run = tmp / ("run%d" % len(list(tmp.iterdir())))
    run.mkdir()
    (run / "replay.json").write_text(json.dumps(src))
    return run


def test_per_key_rate_is_whole_fold_over_whole_fold(tmp_path):
    """TFLOP/s per key must be flops_per_call / s_per_call, not fold FLOPs / per-call seconds."""
    b = _reduce(_fixture(tmp_path), tmp_path)
    assert b["keys"], "no priced keys"
    for p in b["keys"]:
        if not p["TFLOPs"]:
            continue
        # the identity: the per-key rate is its own FLOPs over its own time, so it can never
        # exceed the same-session dense cube, whatever the call count is.
        assert math.isclose(p["TFLOPs"],
                            b["roofs"]["cube8192_TFLOPs"] * p["pct_of_cube"] / 100, rel_tol=1e-12)
        assert p["pct_of_cube"] <= 100.0, p["key"]


def test_cube_criterion_fires_when_an_arm_beats_the_cube(tmp_path):
    def impossible(d):
        for r in d["rows"]:
            if r["key"] == "linear|out=1x512x3072|K=768":
                r["s_per_call_qualified"] /= 40.0
                for m in r["marks"]:
                    m["s_per_call"] /= 40.0
    b = _reduce(_fixture(tmp_path, impossible), tmp_path)
    assert "no_arm_exceeds_the_cube" in b["refutation"]["fired"]
    assert b["refutation"]["no_arm_exceeds_the_cube"] is False


def test_dram_criterion_fires_and_tolerates_the_measured_residual(tmp_path):
    """The roof's own +0.44 % calibration residual must not fire; a real overshoot must."""
    b = _reduce(_fixture(tmp_path), tmp_path)
    assert b["refutation"]["no_arm_exceeds_the_dram_roof"] is True
    assert 0.0 < b["refutation"]["dram_roof_residual_pct"] < 1.0

    def impossible(d):
        for r in d["rows"]:
            if r["key"] == "add_|out=1x512x512x128|K=None":
                r["s_per_call_qualified"] /= 2.0
                for m in r["marks"]:
                    m["s_per_call"] /= 2.0
    b = _reduce(_fixture(tmp_path, impossible), tmp_path)
    assert "no_arm_exceeds_the_dram_roof" in b["refutation"]["fired"]


def test_clean_session_fires_nothing(tmp_path):
    b = _reduce(_fixture(tmp_path), tmp_path)
    assert b["refutation"]["fired"] == [], b["refutation"]["fired"]
    assert b["totals"]["measured_keys_fold_s"] <= b["totals"]["bare_fold_s"]
