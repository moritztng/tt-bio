"""`round_ab.analyse()` on synthetic events, because it only runs at the END of a 28-round run.

A defect here costs the whole card session: the rounds are measured, the process exits, and the
summary that was the point of the run never appears. None of this needs a device -- `analyse` is
pure arithmetic over the event list -- so it is pinned here instead.
"""
import sys

import pytest

sys.path.insert(0, "perf/bcx_bwbytes")
sys.path.insert(0, ".")


def _events(arms_by_round, sg_by_round, evo_by_round, extra=0.3):
    """One round per entry: a start, an arm marker, a sequence_gradients span, device spans."""
    ev, t = [], 1000.0
    for r in sorted(arms_by_round):
        ev.append({"kind": "round_start", "phase": "round", "t0": t, "round": r})
        ev.append({"kind": "arm", "phase": "round", "t0": t, "round": r,
                   "levers_on": arms_by_round[r]})
        s0, s1 = t + 0.1, t + 0.1 + sg_by_round[r]
        ev.append({"kind": "sequence_gradients", "phase": "sequence_gradients",
                   "t0": s0, "t1": s1, "round": r})
        ev.append({"kind": "device", "phase": "evo:backward", "t0": s0 + 0.01,
                   "t1": s0 + 0.01 + evo_by_round[r], "dt": evo_by_round[r], "round": r})
        ev.append({"kind": "device", "phase": "extra:backward", "t0": s0 + 0.02,
                   "t1": s0 + 0.02 + extra, "dt": extra, "round": r})
        t = s1 + 0.2
    ev.append({"kind": "round_stop", "phase": "round", "t0": t, "round": max(arms_by_round) + 1})
    return ev, [(t0, 1350) for t0 in (1000.5, 1001.0)]


def _six_rounds(on_sg, off_sg, on_evo, off_evo):
    arms, sg, evo = {}, {}, {}
    for i, r in enumerate(range(3, 9)):
        on = r % 2 == 0
        arms[r] = on
        sg[r] = (on_sg if on else off_sg)[i // 2]
        evo[r] = (on_evo if on else off_evo)[i // 2]
    return _events(arms, sg, evo)


def test_ratios_and_separation_on_a_clean_ab():
    import round_ab
    ev, clk = _six_rounds([10.0, 10.2, 10.1], [11.0, 11.2, 11.1],
                          [7.0, 7.1, 7.05], [8.0, 8.1, 8.05])
    rows, summary = round_ab.analyse(ev, clk)
    assert summary["on"]["n"] == 3 and summary["off"]["n"] == 3
    assert summary["ratio_sg_off_over_on"] > 1.0          # OFF slower => lever helps
    assert summary["ratio_device_evo_off_over_on"] > 1.0
    assert summary["device_evo_separated"] is True        # every ON below every OFF
    assert summary["separated"] is True


def test_a_round_with_no_arm_marker_does_not_skew_separation():
    """The control. `levers_on` is None when a round has no arm event -- a dropped marker, or a
    round the meter ended early. The medians select with `is True` / `is False`, so such a round
    lands in NEITHER. `device_evo_separated` must use the same predicate, or a None round joins
    the OFF side by truthiness and silently decides the flag the run is reporting."""
    import round_ab
    ev, clk = _six_rounds([10.0, 10.2, 10.1], [11.0, 11.2, 11.1],
                          [7.0, 7.1, 7.05], [8.0, 8.1, 8.05])
    # drop round 5's arm marker (an OFF round) and give it a very low device time
    ev = [e for e in ev if not (e["kind"] == "arm" and e["round"] == 5)]
    for e in ev:
        if e["kind"] == "device" and e["round"] == 5 and e["phase"] == "evo:backward":
            e["dt"] = 0.5
    rows, summary = round_ab.analyse(ev, clk)
    assert summary["off"]["n"] == 2, "a marker-less round must not count as OFF"
    assert summary["device_evo_separated"] is True, (
        "the 0.5 s marker-less round must not be read as an OFF round and break separation")
