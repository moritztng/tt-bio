"""The verdict line is generated, and the rule it generates by.

A cap is the largest size below the FIRST failure, not the largest that happened to fold. The
L1 clash class is not monotone in residue count -- OpenDDE folds 544, throws at 576 and folds
608 -- so a generator that reported the largest passing rung would publish a size that throws.
That is the test below with a gap in it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

GEN = Path(__file__).resolve().parent.parent / "perf" / "whceil" / "findings.py"


def _write(tmp_path, model, rows):
    p = tmp_path / f"ladder_{model}.jsonl"
    with p.open("w") as fh:
        for rung, verdict, extra in rows:
            r = {"model": model, "rung": rung, "verdict": verdict, "wall_s": 1.0,
                 "device": 0, "rc": 0, **extra}
            fh.write(json.dumps(r) + "\n")
    return p


def _run(tmp_path):
    out = subprocess.run([sys.executable, str(GEN), str(tmp_path)],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()


def test_a_model_that_folds_the_bar_and_fails_above_it_passes(tmp_path):
    _write(tmp_path, "m", [("cdk2x2_1024_d8192", "PASS", {}),
                           ("cdk2x2_1088_d8192", "OOM_DRAM",
                            {"request_bytes": 2424307712, "wall_kind": "FRAGMENTATION"})])
    line, = _run(tmp_path)
    assert line.startswith("MODEL m: PASS")
    assert "1024 aa" in line and "1088 aa" in line
    assert "FRAGMENTATION" in line and "2424307712" in line


def test_the_cap_is_below_the_first_failure_even_when_a_larger_rung_folded(tmp_path):
    """The non-monotone case. 1152 folds, but 1088 threw, so the cap is 1024."""
    _write(tmp_path, "m", [("cdk2x2_1024_d8192", "PASS", {}),
                           ("cdk2x2_1088_d8192", "CLASH_L1", {"wall_kind": "L1_CB_CLASH"}),
                           ("cdk2x2_1152_d8192", "PASS", {})])
    line, = _run(tmp_path)
    assert "largest below the first failure 1024 aa" in line, line
    assert "1152" not in line.split("first failure")[0]


def test_no_failure_anywhere_is_a_floor_and_says_so(tmp_path):
    _write(tmp_path, "m", [("cdk2x2_1024_d8192", "PASS", {}),
                           ("cdk2x2_1408_d8192", "PASS", {})])
    line, = _run(tmp_path)
    assert line.startswith("MODEL m: PARTIAL")
    assert "floor and not a ceiling" in line
    # The campaign's DONE_CHECK wants a stated cause next to any non-PASS verdict, and a line
    # that only says "nothing failed" does not give one. This is that word.
    assert "limit" in line


def test_a_model_that_cannot_fold_the_bar_fails(tmp_path):
    _write(tmp_path, "m", [("cdk2x2_1024_d8192", "OOM_DRAM",
                            {"request_bytes": 1, "wall_kind": "CUMULATIVE_RESIDENCY"})])
    line, = _run(tmp_path)
    assert line.startswith("MODEL m: FAIL")


def test_an_unclassified_error_is_not_counted_as_a_size_failure(tmp_path):
    """A guard refusal and a missing shared library are not measurements of anything. Without
    this the round-0 rows would have set every OF3-family cap to 'fails at 1024'."""
    _write(tmp_path, "m", [("cdk2x2_1024_d8192", "ERROR", {}),
                           ("cdk2x2_1024_d8192", "PASS", {}),
                           ("cdk2x2_1088_d8192", "PASS", {})])
    line, = _run(tmp_path)
    assert line.startswith("MODEL m: PARTIAL"), line


def test_depths_are_separate_measurements(tmp_path):
    _write(tmp_path, "m", [("cdk2x2_1024_d8192", "PASS", {}),
                           ("cdk2x2_1024_d14190", "OOM_DRAM",
                            {"request_bytes": 2, "wall_kind": "FRAGMENTATION"})])
    lines = _run(tmp_path)
    assert len(lines) == 2, lines
    assert any("depth 8192" in ln and "PARTIAL" in ln for ln in lines)
    assert any("depth 14190" in ln and "FAIL" in ln for ln in lines)


def test_an_affinity_rung_is_not_reported_at_a_depth_it_never_read(tmp_path):
    """Nesso-1 scores a pair with no alignment. Falling back to the structure fixtures' 35 rows
    would put a depth on a measurement that never had one."""
    p = tmp_path / "ladder_nesso1.jsonl"
    p.write_text(json.dumps({"model": "nesso1", "command": "affinity", "rung": "nesso1_1024",
                             "verdict": "PASS", "wall_s": 1.0, "device": 0, "rc": 0}) + "\n")
    line, = _run(tmp_path)
    assert "reads no alignment" in line
    assert "depth 35" not in line


def test_partial_says_the_bar_is_cleared_so_it_does_not_read_as_a_problem(tmp_path):
    """PARTIAL is about the ceiling, never about the bar. Next to a model that folds 1536 the
    bare word would read as something being wrong with the model."""
    _write(tmp_path, "m", [("cdk2x2_1536_d8192", "PASS", {})])
    line, = _run(tmp_path)
    assert line.startswith("MODEL m: PARTIAL")
    assert "clears the 1024 bar" in line


def test_fail_says_the_bar_is_not_cleared(tmp_path):
    _write(tmp_path, "m", [("cdk2x2_1024_d8192", "OOM_DRAM",
                            {"request_bytes": 1, "wall_kind": "FRAGMENTATION"})])
    line, = _run(tmp_path)
    assert "does NOT clear the 1024 bar" in line
