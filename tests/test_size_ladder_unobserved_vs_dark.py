"""An unobserved rung must not be reported as a lever that went dark.

`frac` is `served / total if total else 0.0`, so a lever the census saw ZERO calls of reads exactly
the same 0.0 as one that declined every call. Boltz-2 hit this on 2026-09-20: seven levers at
768 aa and 896 aa read served 0 / declined 0 with rc=0 and the same 11x10 grid, and every one was
reported as "went dark" while 512 aa and 1024 aa served 72 / 560 / 3 of the same levers. Seven
independent regressions at two adjacent rungs is not what happened; one unmeasured fold is.

The verdict must not move -- both cases are findings and both fail the arm. Only the sentence
changes, because the old one sends the reader after the wrong defect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_gate as G  # noqa: E402


def _row(served, declined, rejects=None, resolved="True", how="wrap"):
    total = served + declined
    return {"served": served, "declined": declined, "rejects": rejects, "resolved": resolved,
            "how": how, "frac": (served / total) if total else 0.0}


def test_unobserved_rung_is_named_as_unmeasured_not_dark():
    base = {"ADALN_S_HOIST": _row(72, 0)}
    cur = {"ADALN_S_HOIST": _row(0, 0)}
    f = G._size_ladder_compare_levers(base, cur, "boltz2/768")
    assert len(f) == 1, f
    assert "observed NO calls" in f[0], f[0]
    assert "not measured" in f[0], f[0]
    assert "went dark" not in f[0], f[0]
    # the baseline's own counts belong in the sentence: they are what says the rung used to be
    # measured, which is the whole reason the zero is suspicious
    assert "72" in f[0], f[0]


def test_a_lever_that_really_declines_still_reads_as_dark_on_its_clause():
    base = {"K2": _row(560, 0)}
    cur = {"K2": _row(0, 560, rejects={"fill_preconditions:padded_mask": 560})}
    f = G._size_ladder_compare_levers(base, cur, "boltz2/768")
    assert len(f) == 1, f
    assert "went dark" in f[0], f[0]
    assert "fill_preconditions:padded_mask" in f[0], f[0]
    assert "observed NO calls" not in f[0], f[0]


def test_both_cases_are_still_findings_so_the_verdict_does_not_move():
    # NEGATIVE CONTROL for the change: the point of the patch is the wording, so prove the
    # arm still fails in the unobserved case. If this ever passes empty, the patch has
    # quietly turned a red arm green, which is the opposite of the intent.
    base = {"A": _row(72, 0), "B": _row(560, 0)}
    unobserved = {"A": _row(0, 0), "B": _row(0, 0)}
    declining = {"A": _row(0, 72, rejects={"c": 72}), "B": _row(0, 560, rejects={"c": 560})}
    assert len(G._size_ladder_compare_levers(base, unobserved, "m/768")) == 2
    assert len(G._size_ladder_compare_levers(base, declining, "m/768")) == 2


def test_a_reproducing_rung_reports_nothing_either_way():
    base = {"A": _row(72, 0)}
    assert G._size_ladder_compare_levers(base, {"A": _row(72, 0)}, "m/512") == []
