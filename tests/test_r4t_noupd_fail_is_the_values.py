"""The D38 no-update FAIL belongs to the measured values, not to the harness (D43).

`perf/r4t/results/noupd_P_70fcdd387.json` is the production-lever arm exactly as committed. Grading it
through the real CLI against the committed floor fails on three cells. These arms change only those
cells, or only a cell that passes, so they show the verdict moves with the values and nothing else:
the doc lines up with the floor by name, carries every term, and grades PASS once the three cells
read the reference, and FAIL again under each of the broken arms.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P_DOC = ROOT / "perf" / "r4t" / "results" / "noupd_P_70fcdd387.json"
FLOOR = ROOT / "perf" / "r4t" / "floor" / "up_bf16.json"
GRADER = ROOT / "perf" / "r4t" / "grade_parity.py"
FAILING = {(0, "bb_ca"), (7, "bb_ca"), (8, "insertion_prediction")}


def _doc() -> dict:
    return json.loads(P_DOC.read_text())


def _set(doc: dict, i: int, term: str, value: float) -> None:
    row = doc["ours"][i]
    row["total"] += value - row[term]
    row[term] = value


def _repaired() -> dict:
    doc = _doc()
    for i, term in FAILING:
        _set(doc, i, term, doc["ref"][i][term])
    return doc


def _grade(tmp_path, doc: dict) -> subprocess.CompletedProcess:
    p = tmp_path / "p.json"
    p.write_text(json.dumps(doc))
    return subprocess.run([sys.executable, str(GRADER), "--parity", str(p), "--floor", str(FLOOR)],
                          capture_output=True, text=True)


def test_the_committed_arm_fails_on_exactly_the_three_reported_cells(tmp_path):
    r = _grade(tmp_path, _doc())
    assert r.returncode == 1, r.stdout + r.stderr
    flagged = [l for l in r.stdout.splitlines() if l.startswith("  batch ")]
    got = {(int(l.split()[1]), l.split("'")[1]) for l in flagged}
    assert got == FAILING, r.stdout


def test_the_committed_reference_column_is_the_floors(tmp_path):
    floor = {row["batch"]: row["ref"] for row in json.loads(FLOOR.read_text())["rows"]}
    doc = _doc()
    for pair, ref in zip(doc["provenance"]["pairs"], doc["ref"]):
        want = floor[pair["batch"].removesuffix(".npz")]
        assert set(ref) == set(want)
        for k in ref:
            assert abs(ref[k] - want[k]) <= 1e-6 * max(1.0, abs(want[k])), (pair["batch"], k)


def test_reading_the_reference_in_the_three_cells_passes(tmp_path):
    r = _grade(tmp_path, _repaired())
    assert r.returncode == 0, r.stdout + r.stderr
    assert "LOSS-PARITY: PASS" in r.stdout


def test_ours_equal_to_ref_passes(tmp_path):
    doc = _doc()
    doc["ours"] = copy.deepcopy(doc["ref"])
    assert _grade(tmp_path, doc).returncode == 0


def test_repaired_arm_fails_again_with_one_passing_term_ten_percent_high(tmp_path):
    doc = _repaired()
    for i, row in enumerate(doc["ours"]):
        _set(doc, i, "local_latents", row["local_latents"] * 1.10)
    r = _grade(tmp_path, doc)
    assert r.returncode == 1 and "LOSS-PARITY: FAIL" in r.stdout, r.stdout + r.stderr


def test_repaired_arm_fails_again_with_one_term_dropped(tmp_path):
    doc = _repaired()
    for row in doc["ours"]:
        row.pop("sequence_prediction")
    r = _grade(tmp_path, doc)
    assert r.returncode == 1 and "LOSS-PARITY: FAIL" in r.stdout, r.stdout + r.stderr


def test_repaired_arm_fails_again_when_one_cell_moves_just_past_its_floor(tmp_path):
    """Tight arm: batch07 bb_ca at 1.01x its floored allowance fails, 0.99x passes."""
    floor = {row["batch"]: row for row in json.loads(FLOOR.read_text())["rows"]}["batch07"]
    ref = floor["ref"]["bb_ca"]
    allow = max(1e-2 * abs(ref), 1.5 * abs(floor["up_bf16"]["bb_ca"] - ref))
    for k, rc in ((0.99, 0), (1.01, 1)):
        doc = _repaired()
        _set(doc, 7, "bb_ca", ref + k * allow)
        r = _grade(tmp_path, doc)
        assert r.returncode == rc, (k, r.stdout + r.stderr)


def test_a_wrong_batch_name_is_refused_not_graded(tmp_path):
    doc = _repaired()
    doc["provenance"]["pairs"][3]["batch"] = "batch42.npz"
    assert _grade(tmp_path, doc).returncode == 2
