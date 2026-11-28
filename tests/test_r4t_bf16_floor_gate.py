"""The committed bf16 floor column, and the grader that reads it (D37).

The floor is data, measured once on pc CPU with `perf/r4t/bf16_floor.py` and committed with a
digest. A grader that reads data can be loosened by editing the data, so this file pins the
committed file's shape and digest, grades two deliberately broken arms through the real CLI
before any arm is read as a PASS, and shows the refusals that keep a floor from being lined up
with the wrong batch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tt_bio.train.parity import LOSS_PARITY_MIN_BATCHES, floor_digest

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "perf" / "r4t" / "floor" / "up_bf16.json"
GRADER = ROOT / "perf" / "r4t" / "grade_parity.py"
TERMS = {"bb_ca", "local_latents", "insertion_prediction", "sequence_prediction",
         "signed_volume", "total"}


def _floor_doc() -> dict:
    return json.loads(FLOOR.read_text())


def test_the_committed_floor_matches_its_digest_and_covers_the_graded_batches():
    doc = _floor_doc()
    rows = doc["rows"]
    assert doc["digest"] == floor_digest(rows)
    assert len(rows) >= LOSS_PARITY_MIN_BATCHES
    assert {r["batch"] for r in rows} == {f"batch0{i}" for i in range(len(rows))}
    for r in rows:
        assert set(r["up_bf16"]) == set(r["ref"]) == TERMS
    assert doc["provenance"]["arm"] == "bf16"


def _parity_doc(scale_term=None, scale=1.0, drop_term=None) -> dict:
    """ours = the fp32 reference, optionally broken on one term across every batch."""
    rows = _floor_doc()["rows"]
    ours, ref, pairs = [], [], []
    for r in rows:
        a, b = dict(r["ref"]), dict(r["ref"])
        if scale_term:
            a[scale_term] = a[scale_term] * scale
            a["total"] = sum(v for k, v in a.items() if k != "total")
        if drop_term:
            a.pop(drop_term)
        ours.append(a)
        ref.append(b)
        pairs.append({"batch": r["batch"] + ".npz", "src": "synthetic", "winner": 0,
                      "upstream_winner": 0})
    return {"dtype": "bfloat16", "ours": ours, "ref": ref, "provenance": {"pairs": pairs}}


def _grade(tmp_path, doc, floor=FLOOR, name="p.json") -> subprocess.CompletedProcess:
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    cmd = [sys.executable, str(GRADER), "--parity", str(p)]
    if floor is not None:
        cmd += ["--floor", str(floor)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_an_arm_equal_to_the_reference_passes(tmp_path):
    r = _grade(tmp_path, _parity_doc())
    assert r.returncode == 0, r.stdout + r.stderr
    assert "LOSS-PARITY: PASS" in r.stdout and "floor=up_bf16x1.5" in r.stdout


def test_one_term_ten_percent_high_on_every_batch_fails(tmp_path):
    """The first broken arm D37 names. 10 % is above 1.5x upstream's own bf16 miss on every term."""
    r = _grade(tmp_path, _parity_doc(scale_term="local_latents", scale=1.10))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "LOSS-PARITY: FAIL" in r.stdout
    assert r.stdout.count("term 'local_latents'") >= LOSS_PARITY_MIN_BATCHES


def test_a_dropped_term_fails_even_though_every_remaining_term_matches(tmp_path):
    """The second broken arm. A relative tolerance alone would never see this."""
    r = _grade(tmp_path, _parity_doc(drop_term="signed_volume"))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "terms differ" in r.stdout


def test_a_term_inside_upstreams_own_bf16_miss_passes_where_the_plain_bar_fails(tmp_path):
    """bb_ca on batch01: upstream's own bf16 is 20.9 % off, so 10 % is not a finding."""
    doc = _parity_doc()
    rows = _floor_doc()["rows"]
    i = [r["batch"] for r in rows].index("batch01")
    doc["ours"][i]["bb_ca"] *= 1.10
    assert _grade(tmp_path, doc).returncode == 0
    assert _grade(tmp_path, doc, floor=None).returncode == 1


def test_grader_refuses_an_edited_floor_file(tmp_path):
    doc = _floor_doc()
    doc["rows"][0]["up_bf16"]["bb_ca"] *= 2.0
    bad = tmp_path / "floor.json"
    bad.write_text(json.dumps(doc))
    r = _grade(tmp_path, _parity_doc(), floor=bad)
    assert r.returncode == 2 and "REFUSED" in r.stdout


def test_grader_refuses_a_doc_without_batch_names_and_an_unknown_batch(tmp_path):
    doc = _parity_doc()
    doc["provenance"] = {}
    assert _grade(tmp_path, doc).returncode == 2
    doc = _parity_doc()
    doc["provenance"]["pairs"][2]["batch"] = "batch99.npz"
    r = _grade(tmp_path, doc)
    assert r.returncode == 2 and "no upstream bf16 floor" in r.stdout


def test_grader_refuses_a_floor_measured_against_a_different_reference(tmp_path):
    """The floor only scales a deviation from the SAME fp32 reference it was measured against."""
    doc = _parity_doc()
    doc["ref"][4]["local_latents"] *= 1.01
    doc["ours"][4]["local_latents"] *= 1.01
    r = _grade(tmp_path, doc)
    assert r.returncode == 2 and "measured against" in r.stdout


@pytest.mark.parametrize("n", [LOSS_PARITY_MIN_BATCHES - 1])
def test_a_short_arm_still_fails_on_n(tmp_path, n):
    doc = _parity_doc()
    for k in ("ours", "ref"):
        doc[k] = doc[k][:n]
    doc["provenance"]["pairs"] = doc["provenance"]["pairs"][:n]
    r = _grade(tmp_path, doc)
    assert r.returncode == 1 and "the bar needs >=" in r.stdout
