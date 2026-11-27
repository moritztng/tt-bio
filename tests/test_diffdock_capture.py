"""The device harness's host-side half: the clock summary and the warm-up order.

The capture and the replay need a card. What does not is the rule that a timing without a
live clock is not a measurement, and the rule that the body runs once eagerly before the
capture opens -- both are decisions this row is accountable for, so both are pinned here.
"""

from __future__ import annotations

import pytest

from tt_bio.diffdock.capture import capture_loop, clock_summary


def test_a_timing_with_no_clock_says_so_instead_of_reporting_a_number():
    s = clock_summary([])
    assert s["aiclk_mhz"] is None and s["aiclk_samples"] == 0
    assert "not comparable" in s["aiclk_note"]


def test_the_clock_is_the_median_of_live_samples_and_keeps_the_spread():
    s = clock_summary([1350, 1349, 1351, 1200])
    assert s["aiclk_mhz"] == 1349
    assert (s["aiclk_min"], s["aiclk_max"], s["aiclk_samples"]) == (1200, 1351, 4)


def test_the_body_runs_once_eagerly_before_the_capture_opens():
    """The warm-up rule, as a test rather than as a comment."""
    calls = []

    class FakeDevice:
        pass

    def body(t):
        calls.append(("body", t, _capturing[0]))
        return t

    _capturing = [False]

    def fake_capture_trace(dev, run):
        _capturing[0] = True
        try:
            return 7, run()
        finally:
            _capturing[0] = False

    import tt_bio.tenstorrent as T
    real = getattr(T, "capture_trace")
    T.capture_trace = fake_capture_trace
    try:
        tid, warm = capture_loop(FakeDevice(), body, 19)
    finally:
        T.capture_trace = real

    assert tid == 7 and warm == 0
    assert calls[0] == ("body", 0, False), "the first pass must be outside the capture"
    assert len(calls) == 20 and all(c[2] for c in calls[1:]), "19 captured steps after it"
