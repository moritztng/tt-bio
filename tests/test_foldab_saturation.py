"""The A/A floor cannot see a saturated box, so the harness needs a second detector.

Built from a real session: 2026-09-15, merged tree, 512 aa, 12 folds. A sibling worker's 200-step
fold sat on qb2 at 253 % CPU throughout. Both shipped slots came out at 23.5 s against a 15.217 s
quiet-box median, so the A/A floor read 0.4 % and the session would have published ratio ~1.000 as
INTERPRETABLE, refuting a win that is real. The numbers below are that session's.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf" / "roof_transition_chunk_bh"))
from foldab import leg_verdict  # noqa: E402

QUIET_512 = 15.217   # foldab_lo_c1.json, ship_median_s on a quiet box
TOL = 1.10


def test_saturated_session_is_blocked_even_though_its_floor_is_tight():
    # the case that actually happened: floor 0.4 %, box 1.54x slow
    assert leg_verdict(0.4, 23.5, QUIET_512, TOL) == "BLOCKED-ON-SATURATION"


def test_the_floor_alone_would_have_passed_that_session():
    # negative control: without a baseline there is nothing left but the floor, and it says fine.
    # If this ever returns anything but INTERPRETABLE the test above proves nothing.
    assert leg_verdict(0.4, 23.5, None, TOL) == "INTERPRETABLE"


def test_a_quiet_session_still_reads():
    assert leg_verdict(0.194, 15.217, QUIET_512, TOL) == "INTERPRETABLE"


def test_tolerance_is_a_band_not_an_equality():
    assert leg_verdict(0.2, QUIET_512 * 1.09, QUIET_512, TOL) == "INTERPRETABLE"
    assert leg_verdict(0.2, QUIET_512 * 1.11, QUIET_512, TOL) == "BLOCKED-ON-SATURATION"


def test_jitter_still_wins_over_saturation():
    # a loose floor is reported as contention even on a box that is fast in absolute terms
    assert leg_verdict(3.0, 15.0, QUIET_512, TOL) == "BLOCKED-ON-CONTENTION"
