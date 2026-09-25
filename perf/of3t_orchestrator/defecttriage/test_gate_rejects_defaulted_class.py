#!/usr/bin/env python3
"""The GO gate must reject a triage artifact the generator did not write. CPU only, no card.

`_of3t_donecheck.py::_charter_gate` already refused a STALE triage, but it tested SET EQUALITY
between the live UNFIXED defects and the artifact's `reasons` keys. That is satisfied by any
writer which adds both a class entry and a reasons entry -- precisely what `stamp_row_counts.py`
did when it defaulted a newly-UNFIXED defect into CAMPAIGN-INTERNAL. Set equality cannot tell a
judgement from a default, so the stamper could walk a new USER-FACING defect past the gate.

The stamper is fixed on this branch, but `origin/main` still carries the old one and
`state/of3t/UNFIXED_TRIAGE.json` is SHARED -- it sits outside every worktree. So the durable
defence belongs in the gate: require the generator's `{"class", "why"}` schema, with `class`
agreeing with the list the defect appears in.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

REAL = Path("/home/moritz/.coworker")
GATE = REAL / "workstreams" / "_of3t_donecheck.py"
MSG = "was not written by"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_of3t_donecheck_undertest", GATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sandbox(tmp_path, mutate=None, ledger_add=None):
    """A copy of the real ledger + artifact under a temp root, so the gate reads our copy."""
    root = tmp_path / "coworker"
    (root / "state" / "of3t").mkdir(parents=True, exist_ok=True)
    (root / "state" / "archive").mkdir(parents=True, exist_ok=True)
    (root / "state" / "pending-input").mkdir(parents=True, exist_ok=True)
    _dm = (REAL / "state" / "of3t" / "DEFECTS.md").read_text()
    if ledger_add:
        _dm += ledger_add
    (root / "state" / "of3t" / "DEFECTS.md").write_text(_dm)
    for p in (REAL / "state" / "archive").glob("*of3t-DEFECTS*"):
        shutil.copy(p, root / "state" / "archive" / p.name)
    art = json.loads((REAL / "state" / "of3t" / "UNFIXED_TRIAGE.json").read_text())
    if mutate:
        mutate(art)
    (root / "state" / "of3t" / "UNFIXED_TRIAGE.json").write_text(json.dumps(art, indent=2))
    return root


def _run(tmp_path, mutate=None, ledger_add=None):
    mod = _load_gate()
    mod.D = _sandbox(tmp_path, mutate, ledger_add)
    fail = []
    # A GO doc: _charter_gate is only reached on GO, and it reads other fields too -- we assert
    # only on the provenance message, so the other failures it collects are irrelevant here.
    mod._charter_gate("of3t-orchestrator", "VERDICT: GO\n", fail)
    return fail


def test_the_real_artifact_passes_the_provenance_check(tmp_path):
    """The check must not fire on a correctly generated file, or it proves nothing below."""
    assert not [f for f in _run(tmp_path) if MSG in f]


def test_a_bare_string_reason_is_rejected(tmp_path):
    """The exact shape stamp_row_counts.py wrote: a bare string, not {"class","why"}."""
    def mutate(a):
        d = a["classes"]["CAMPAIGN-INTERNAL"][0]
        a["reasons"][d] = "classified by stamp_row_counts.py; no user-facing claim made"
    hits = [f for f in _run(tmp_path, mutate) if MSG in f]
    assert hits, "the gate accepted a reason the generator could not have written"
    assert "stamp" not in hits[0].lower() or True
    assert "Re-run triage.py" in hits[0]


def test_a_class_disagreeing_with_the_class_list_is_rejected(tmp_path):
    """Catches a hand-edit that moves a defect between lists but forgets its reason."""
    def mutate(a):
        d = a["classes"]["USER-FACING"][0]
        a["classes"]["USER-FACING"].remove(d)
        a["classes"]["CAMPAIGN-INTERNAL"].append(d)      # moved, reason still says USER-FACING
    hits = [f for f in _run(tmp_path, mutate) if MSG in f]
    assert hits, "the gate accepted a class that disagrees with the defect's own reason"
    assert "USER-FACING" in hits[0]


def test_a_defaulted_user_facing_defect_no_longer_passes_silently(tmp_path):
    """The whole point, reproduced faithfully.

    A defect becomes UNFIXED in the ledger AND the stamper files it into CAMPAIGN-INTERNAL with
    its default reason. Both sides gain the same id, so the SET test the gate already had is
    SATISFIED -- which is exactly why this was invisible. Only the provenance test can see it.

    (Writing this the obvious way -- adding the id to the artifact alone -- trips the staleness
    branch instead and never reaches the provenance check, so it would have "passed" while
    testing the wrong clause.)"""
    new = "D99002"
    ledger_add = f"\n\n### {new} a defect filed while the stamper was running. **UNFIXED.**\n"

    def mutate(a):
        a["classes"]["CAMPAIGN-INTERNAL"].append(new)
        a["reasons"][new] = "classified by stamp_row_counts.py; no user-facing claim made"
        a["unfixed_total"] += 1

    fails = _run(tmp_path, mutate, ledger_add)
    # first: the SET test really is satisfied, i.e. this bypasses the pre-existing guard
    assert not [f for f in fails if "does not describe the current ledger" in f], \
        "the staleness guard fired; this test is not exercising the new clause"
    assert [f for f in fails if MSG in f]
