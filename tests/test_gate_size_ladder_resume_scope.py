"""A size-ladder record must say which of the nine ladders it scored.

Found 2026-09-19 while sizing a per-model split of the arm after qb2's second hard-reset of the
day. `--model size-ladder` is one --model value standing for nine per-model ladders, and
`--size-ladder-models boltz2` scores one of them under the same "GATE PASS - lever census and
scaling exponents" headline. Before this fix the journal recorded `members=["size-ladder"]` either
way, so a `--resume` discharged the full nine-model arm -- ~2h45m of device time on a p300c -- on
the strength of an 8 min debug run. Same defect as the `ingest()` members bug fixed the same day:
the record did not describe what was scored.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import release_gate as rg  # noqa: E402

FULL = list(rg.SIZE_LADDER_MODELS)


def _key(**over):
    k = {"commit": "abc1234", "dirty": False, "host": "h", "card_type": "p300c",
         "fast": False, "diffusion_trace": False, "package": "/p/tt_bio"}
    k.update(over)
    return k


def _journal(tmp_path, members):
    import gate_journal as gj
    p = tmp_path / "journal.jsonl"
    gj.append(p, _key(), "size-ladder", "PASS",
              "GATE PASS - lever census and scaling exponents", members=members)
    return p


def test_full_run_members_name_every_ladder():
    m = rg._arm_members("size-ladder", ["size-ladder"])
    assert m == sorted(f"size-ladder:{x}" for x in FULL)
    assert len(m) == 9


def test_subset_run_members_name_only_what_it_scored():
    assert rg._arm_members("size-ladder", ["size-ladder"], ["boltz2"]) == ["size-ladder:boltz2"]


def test_arm_not_requested_covers_nothing():
    """Negative control: the members must come from the request, not from the constant."""
    assert rg._arm_members("size-ladder", ["boltz2"]) == []


def test_one_model_record_does_not_discharge_the_nine_model_arm(tmp_path):
    """Forward-looking: a subset record under the NEW naming must not cover the full arm.
    This one also passes against the pre-fix code, because pre-fix code never wrote such a
    record. The test below is the one that fails on HEAD."""
    j = _journal(tmp_path, ["size-ladder:boltz2"])
    resumed, remaining = rg._resume_plan(j, _key(), ["size-ladder"])
    assert resumed == {}
    assert remaining == ["size-ladder"]


def test_nine_model_record_does_discharge_it(tmp_path):
    """Paired with the test above: the fix must still let a real full run resume, or it has
    only replaced a false green with a resume that never fires."""
    j = _journal(tmp_path, [f"size-ladder:{x}" for x in FULL])
    resumed, remaining = rg._resume_plan(j, _key(), ["size-ladder"])
    assert set(resumed) == {"size-ladder"}
    assert remaining == []


def test_a_record_that_cannot_name_its_ladders_refuses(tmp_path):
    """THE DEFECT. `members=["size-ladder"]` is what the pre-fix code wrote for BOTH a full run
    and a `--size-ladder-models boltz2` run, so it cannot prove which ladders were walked and
    must not resume. Verified to fail against HEAD (4 of these 8 do), which is the point:
    pre-fix, this record discharged all nine ladders. Refusing costs a re-run; the other
    direction ships an unrun arm inside a green gate."""
    j = _journal(tmp_path, ["size-ladder"])
    resumed, _ = rg._resume_plan(j, _key(), ["size-ladder"])
    assert resumed == {}


def test_other_arms_are_unchanged(tmp_path):
    """Negative control on the blast radius: the two arms that already had members, and a
    plain self-named arm, must resume exactly as before."""
    import gate_journal as gj
    j = tmp_path / "j.jsonl"
    gj.append(j, _key(), "nesso1", "PASS", "GATE PASS - nesso1 scalars cleared the reference floor")
    gj.append(j, _key(), "fold-models", "PASS",
              "GATE PASS - all models cleared parse + ground-truth floor + geometry",
              members=list(rg.MODELS))
    resumed, remaining = rg._resume_plan(j, _key(), ["nesso1", *rg.MODELS])
    assert set(resumed) == {"nesso1", "fold-models"}
    assert remaining == []


def test_a_different_tree_still_refuses(tmp_path):
    """Negative control on the key: namespaced members must not weaken the commit check."""
    j = _journal(tmp_path, [f"size-ladder:{x}" for x in FULL])
    resumed, _ = rg._resume_plan(j, _key(commit="deadbee"), ["size-ladder"])
    assert resumed == {}
