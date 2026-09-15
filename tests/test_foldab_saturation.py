"""A/A floor and absolute wall are different detectors, and the floor's margin can be thin.

Real session, 2026-09-15, merged tree, 512 aa, 12 folds, `out/foldab_remerge_saturated_512.json`.
A sibling worker's 200-step 1024 aa fold sat on qb2 at 253 % CPU for part of it and the loadavg
span was 0.71 to 16.1. What came out:

    ship_median_s 23.514   quiet-box median 15.217   -> 1.545x
    aa_floor_pct  1.503    bar 1.0                   -> BLOCKED-ON-CONTENTION
    ratio         1.0009   (suppressed, correctly)

So the floor DID fire here, by 0.5 of a percentage point. The absolute wall missed by 45 %. Both
detectors agreed on this session; they disagree in how much room they had, and that is the argument
for carrying both. The floor measures whether the box CHANGED between the two shipped slots, so a
box that is uniformly and steadily slow moves it very little, while the wall sees that directly.
Neither is a superset of the other: a box jittering around a correct mean trips the floor and not
the wall.

The cases below are constructed to separate them, not replayed from the session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf" / "roof_transition_chunk_bh"))
from foldab import leg_verdict  # noqa: E402

QUIET_512 = 15.217   # foldab_lo_c1.json, ship_median_s on a quiet box
TOL = 1.10


def test_the_real_session_blocks():
    assert leg_verdict(1.503, 23.514, QUIET_512, TOL) == "BLOCKED-ON-CONTENTION"


def test_steady_saturation_blocks_where_the_floor_alone_would_not():
    # the case the second detector exists for: same 1.545x wall, but the box held that speed
    # steadily, so the two shipped slots agree and the floor reads well inside its bar
    assert leg_verdict(0.4, 23.514, QUIET_512, TOL) == "BLOCKED-ON-SATURATION"


def test_without_a_baseline_that_session_would_read_clean():
    # negative control. If this does not come back INTERPRETABLE the test above proves nothing,
    # because it would be the floor doing the work and not the new check.
    assert leg_verdict(0.4, 23.514, None, TOL) == "INTERPRETABLE"


def test_jitter_around_a_correct_mean_is_the_floor_s_case_and_not_the_wall_s():
    assert leg_verdict(3.0, QUIET_512, QUIET_512, TOL) == "BLOCKED-ON-CONTENTION"


def test_a_quiet_session_still_reads():
    assert leg_verdict(0.194, 15.217, QUIET_512, TOL) == "INTERPRETABLE"


def test_tolerance_is_a_band_not_an_equality():
    assert leg_verdict(0.2, QUIET_512 * 1.09, QUIET_512, TOL) == "INTERPRETABLE"
    assert leg_verdict(0.2, QUIET_512 * 1.11, QUIET_512, TOL) == "BLOCKED-ON-SATURATION"
