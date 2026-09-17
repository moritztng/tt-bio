#!/usr/bin/env python3
"""Controls for wedge_telemetry.py."""
import pytest

import wedge_telemetry as T


def test_the_two_states_separate_on_slv_rd_with_no_overlap():
    d = T.analyse()["discriminator"]
    assert d["wedged_slv_rd_range"] == [0, 0]
    assert d["folding_slv_rd_range"][0] > 0
    assert d["folding_slv_rd_range"][0] > d["wedged_slv_rd_range"][1]


def test_the_wedged_poll_rate_is_essentially_constant_and_folding_is_not():
    d = T.analyse()["discriminator"]
    assert d["wedged_mst_rd_spread_pct"] < 1.0, "a poll loop must be near-constant"
    assert d["folding_mst_rd_spread_pct"] > 50.0, "real work must vary"


def test_power_separates_too_but_less_sharply_than_the_counters():
    d = T.analyse()["discriminator"]
    lo_f, hi_f = d["folding_power_range_W"]
    lo_w, hi_w = d["wedged_power_range_W"]
    assert hi_w < hi_f, "wedged power must be below the folding peak"
    assert lo_w <= lo_f, "power alone overlaps: it is the weaker signal"


def test_every_wedged_sample_really_has_zero_slv_rd():
    assert all(r[4] == 0 for r in T.WEDGED)
    assert all(r[4] > 0 for r in T.FOLDING)


def test_the_second_wedge_is_recorded_as_NOT_scorable():
    s = T.analyse()["second_wedge_not_scorable"]
    assert s["aiclk"] == 0 and s["power_W"] == 0.0
    assert "NOT available" in s["note"]


def test_the_mixed_day_parsing_trap_is_documented_with_its_tell():
    t = T.analyse()["parsing_trap"]
    assert "38.9 hours" in t and "absolute epochs" in t and "1789520281" in t


def test_it_does_not_claim_a_validated_detector():
    lims = " ".join(T.analyse()["limits"])
    assert "not a validated detector" in lims
    assert "false positive" in lims


def test_stats_are_computed_from_the_rows_not_asserted():
    f = T._stats(T.FOLDING)
    assert f["slv_rd_max"] == max(r[4] for r in T.FOLDING)
    assert f["power_min_W"] == min(r[2] for r in T.FOLDING)


def test_a_synthetic_folding_trace_is_not_classified_as_wedged():
    """Negative control: vary slv_rd and the wedge signature must disappear."""
    rows = [(f"00:0{i}:00", 1350, 50.0, 1000000 + i * 90000, 500000 + i * 40000) for i in range(6)]
    s = T._stats(rows)
    assert s["slv_rd_min"] > 0 and s["mst_rd_spread_pct"] > 1.0


def test_a_synthetic_poll_loop_reproduces_the_wedge_signature():
    rows = [(f"00:0{i}:00", 800, 23.0, 1073000 + (i % 2), 0) for i in range(6)]
    s = T._stats(rows)
    assert s["slv_rd_max"] == 0 and s["mst_rd_spread_pct"] < 1.0
