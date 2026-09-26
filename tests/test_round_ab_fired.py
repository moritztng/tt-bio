"""`round_ab.check_fired` — the guard that decides whether a finished A/B may be reported.

It runs after all 28 rounds, so a FALSE raise discards a completed card session and a MISSED
raise reports a 1.00x as a measurement when nothing fired. Both directions are pinned here, with
no device.
"""
import collections
import sys

import pytest

sys.path.insert(0, "perf/bcx_bwbytes")
sys.path.insert(0, ".")


def rows(*arms, start=3):
    return [{"round": start + i, "levers_on": a} for i, a in enumerate(arms)]


def C(**kw):
    return collections.Counter(kw)


def test_a_normal_run_passes_and_reports_what_it_served():
    import round_ab
    served, off = round_ab.check_fired(C(**{"tree:on": 96, "reblock_back:on": 24}),
                                       rows(False, True, False, True), precision=False)
    assert (served, off) == (120, 0)


def test_on_rounds_that_served_nothing_raise():
    """The gates are set at 256 and a BC2 round is 224, so this is the expected outcome of
    running the two routed levers at BC2's size. It must be loud, not a 1.00x."""
    import round_ab
    with pytest.raises(RuntimeError, match="served nothing"):
        round_ab.check_fired(C(), rows(False, True, False, True), precision=False)


def test_an_off_round_that_served_anything_raises():
    import round_ab
    with pytest.raises(RuntimeError, match="does not separate"):
        round_ab.check_fired(C(**{"tree:on": 96, "tree:off": 1}),
                             rows(False, True), precision=False)


def test_precision_arm_with_the_same_dtype_on_both_arms_raises():
    """A dtype lever that leaves the dtype alone is wired and inert, and its call count reads
    identically either way -- which is precisely how such a lever passes unnoticed."""
    import round_ab
    with pytest.raises(RuntimeError, match="did not change the softmax backward"):
        round_ab.check_fired(C(**{"softmax_bw:on:float32": 40, "softmax_bw:off:float32": 40}),
                             rows(False, True), precision=True)


def test_precision_arm_that_moved_the_dtype_passes():
    import round_ab
    served, off = round_ab.check_fired(
        C(**{"softmax_bw:on:bfloat16": 40, "softmax_bw:off:float32": 40}),
        rows(False, True), precision=True)
    assert served == 40 and off == 0


def test_a_missing_arm_marker_does_not_suppress_the_served_nothing_raise():
    """The control for the predicate. `levers_on` is None when an arm marker goes missing. Under
    truthiness such a round is not an ON round, so if every ON marker were dropped the guard
    would quietly decline to fire while `analyse` reported an arm with no rounds in it. Here one
    real ON round survives and the raise must still happen."""
    import round_ab
    with pytest.raises(RuntimeError, match="served nothing"):
        round_ab.check_fired(C(), rows(False, None, False, True), precision=False)


def test_all_markers_missing_is_not_read_as_an_on_arm():
    """And the other direction: with NO round marked ON there is nothing to have served, so the
    guard must stay quiet rather than blame the levers for a meter problem."""
    import round_ab
    served, off = round_ab.check_fired(C(), rows(None, None), precision=False)
    assert (served, off) == (0, 0)
