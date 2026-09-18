"""An arm's verdict must outlive the process that printed it, and must not cross trees.

Device-free and torch-free. On 2026-09-18 tt-quietbox2 hard-reset twice under campaign load,
at 11:03:23Z and at 19:05:50Z. The second one killed a release gate 47 minutes in with eight
arms already PASS -- fold models, rf3-1024aa, rfd3-fusion, boltzgen, rfd3, pxdesign,
opendde-abag, nesso1 -- and nothing on disk said so, so the only way to restart was from the
top. Neither boot logged a shutdown sequence, so there is no signal to trap and no exit code
to read.

The journal makes an arm's verdict a file. The risk that creates is the opposite one: a green
composed out of arms scored on different trees. Half the tests here are that negative control.

Run: python3 tests/test_gate_arm_journal.py, or via pytest.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import gate_journal as gj


KEY = {"commit": "abc1234", "dirty": False, "host": "tt-quietbox2", "card_type": "p300c",
       "fast": False, "diffusion_trace": False, "package": "/opt/venv/lib/tt_bio"}


def _journal(tmp):
    return Path(tmp) / "journal.jsonl"


def test_verdict_of_reads_the_three_headline_shapes():
    assert gj.verdict_of("GATE PASS — nesso1 scalars cleared the reference floor") == "PASS"
    assert gj.verdict_of("GATE FAIL — size-ladder drift vs the recorded baseline") == "FAIL"
    assert gj.verdict_of("GATE BLOCKED — every fold leg never opened a device") == "BLOCKED"
    assert gj.verdict_of("model            worst vs ref   PASS") is None
    assert gj.verdict_of("") is None


def test_a_recorded_pass_is_resumable_and_survives_a_torn_last_line():
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        gj.append(j, KEY, "nesso1", "PASS", "GATE PASS — nesso1 scalars cleared the floor")
        # A process killed mid-write leaves a partial line. It must cost that record and
        # nothing else: this is the exact case the journal exists for.
        with j.open("a") as fh:
            fh.write('{"ts": 1.0, "arm": "capac')
        assert set(gj.resumable(j, KEY)) == {"nesso1"}


def test_a_fail_is_never_resumed_and_a_later_fail_overrides_an_earlier_pass():
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        gj.append(j, KEY, "capacity", "FAIL", "GATE FAIL — capacity regression")
        assert gj.resumable(j, KEY) == {}
        gj.append(j, KEY, "nesso1", "PASS", "GATE PASS — nesso1 cleared")
        gj.append(j, KEY, "nesso1", "FAIL", "GATE FAIL — nesso1 drifted run to run")
        assert gj.resumable(j, KEY) == {}, "the LATEST verdict decides, not the best one"


@pytest.mark.parametrize("field,value", [
    ("commit", "deadbee"),
    ("host", "tt-quietbox"),
    ("card_type", "p150a"),
    ("fast", True),
    ("diffusion_trace", True),
    ("package", "/somewhere/else/tt_bio"),
])
def test_a_pass_does_not_cross_a_key_field(field, value):
    """The negative control. Every one of these changes what the arm would score, so a
    verdict recorded under one must not discharge the other."""
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        gj.append(j, KEY, "nesso1", "PASS", "GATE PASS — nesso1 cleared")
        assert gj.resumable(j, dict(KEY, **{field: value})) == {}, field
        assert set(gj.resumable(j, KEY)) == {"nesso1"}, "control: the same key still resumes"


def test_a_dirty_tree_is_never_resumable_in_either_direction():
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        dirty = dict(KEY, dirty=True)
        gj.append(j, dirty, "nesso1", "PASS", "GATE PASS — nesso1 cleared")
        assert gj.resumable(j, dirty) == {}, "two dirty runs at one commit are not one tree"
        assert gj.resumable(j, KEY) == {}, "a dirty record must not discharge a clean run"


def test_members_are_recorded_so_a_narrow_run_cannot_discharge_a_wide_one():
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        gj.append(j, KEY, "fold-models", "PASS", "GATE PASS — all models cleared",
                  members=["boltz2"])
        rec = gj.resumable(j, KEY)["fold-models"]
        assert rec["members"] == ["boltz2"]
        # The consumer of that field is release_gate._resume_plan; see its own test below.


def test_ingest_refuses_a_headline_it_cannot_map_rather_than_dropping_it():
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        log = Path(tmp) / "gate.log"
        log.write_text("GATE PASS — nesso1 scalars cleared the reference floor\n"
                       "GATE PASS — something nobody has written yet\n")
        with pytest.raises(ValueError, match="stale"):
            gj.ingest(log, KEY, j)


def test_ingest_backfills_a_log_with_its_line_as_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        log = Path(tmp) / "gate.log"
        log.write_text("noise\n"
                       "GATE PASS — rf3 at 997 aa cleared the crystal floor\n"
                       "more noise\n"
                       "GATE FAIL — capacity regression at the largest supported input\n")
        got = gj.ingest(log, KEY, j)
        assert got == [("rf3-1024aa", "PASS"), ("capacity", "FAIL")]
        recs = gj.read(j)
        assert recs[0]["source"].endswith("gate.log:2")
        assert set(gj.resumable(j, KEY)) == {"rf3-1024aa"}


def test_every_ingest_marker_is_still_literally_in_the_gate():
    """The table matches headline TEXT, so the only thing that keeps it honest is that the
    text still exists. A reworded headline has to fail here, not silently stop back-filling
    the arm it describes."""
    src = Path(REPO, "scripts", "release_gate.py").read_text()
    # The gate wraps its headlines across source lines, so compare against the source with
    # the f-string continuations joined back up.
    flat = " ".join(src.split())
    # A BLOCKED headline is composed by _gate_line as "GATE BLOCKED — {what} never opened a
    # device", so only the `what` half is a literal in the source. Check the two parts.
    BLOCKED = " never opened a device"
    assert BLOCKED in flat, "_gate_line no longer composes the BLOCKED headline this way"
    missing = []
    for arm, markers in gj.INGEST_MARKERS.items():
        for m in markers:
            probe = m[:-len(BLOCKED)] if m.endswith(BLOCKED) else m
            if " ".join(probe.split()) not in flat:
                missing.append((arm, m))
    assert not missing, f"ingest markers no longer in release_gate.py: {missing}"


def test_every_ingest_marker_names_an_arm_the_gate_actually_has():
    import release_gate as rg
    known = set(rg.DEFAULT_ARMS) | {"size-ladder", "fold-models", "esmc"}
    assert set(gj.INGEST_MARKERS) == known, (
        "the ingest table and the gate's arms have drifted apart; a missing arm makes a "
        "back-filled journal silently under-report what a log contains")


# --- release_gate's side of it -------------------------------------------------------------

def test_resume_plan_drops_only_the_arms_a_matching_pass_covers():
    import release_gate as rg
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        models = ["boltz2", "nesso1", "capacity", "size-ladder"]
        gj.append(j, KEY, "nesso1", "PASS", "GATE PASS — nesso1 cleared", members=["nesso1"])
        resumed, remaining = rg._resume_plan(j, KEY, models)
        assert set(resumed) == {"nesso1"}
        assert remaining == ["boltz2", "capacity", "size-ladder"]


def test_resume_plan_will_not_let_one_fold_model_discharge_the_whole_fold_leg():
    """The fold leg scores every model in MODELS under ONE headline, so its record has to
    carry which ones. A `--model boltz2` run must not discharge the full five."""
    import release_gate as rg
    with tempfile.TemporaryDirectory() as tmp:
        j = _journal(tmp)
        gj.append(j, KEY, "fold-models", "PASS", "GATE PASS — all models cleared",
                  members=["boltz2"])
        wide = [m for m in rg.MODELS]
        resumed, remaining = rg._resume_plan(j, KEY, wide)
        assert resumed == {}, "a one-model record cannot stand in for the whole fold leg"
        assert remaining == wide
        # Control: the same record DOES discharge a run that only asks for boltz2.
        resumed, remaining = rg._resume_plan(j, KEY, ["boltz2"])
        assert set(resumed) == {"fold-models"} and remaining == []


def test_every_gate_leg_routes_its_headline_through_the_journal():
    """A leg that still calls bare print() for its verdict is invisible to a resume, and the
    failure is silent: the arm just gets re-folded forever."""
    import re
    src = Path(REPO, "scripts", "release_gate.py").read_text()
    main = src[src.index("\ndef main()"):]
    stray = [ln.strip() for ln in main.splitlines()
             if re.search(r'print\((f?")GATE (PASS|FAIL|BLOCKED)', ln)]
    assert not stray, f"these verdicts bypass _headline(): {stray}"
    armed = set(re.findall(r'_headline\("([a-z0-9-]+)"', main))
    assert armed == set(gj.INGEST_MARKERS), (
        f"legs journalled: {sorted(armed)}; ingest table: {sorted(gj.INGEST_MARKERS)}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
