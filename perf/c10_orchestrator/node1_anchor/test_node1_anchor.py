#!/usr/bin/env python3
"""Controls for node1_anchor.py."""
import pytest

import node1_anchor as N


def test_the_board_pairs_match_the_serials():
    r = N.analyse()
    for serial, nodes in r["board_pairs"].items():
        assert {N.NODES[n]["board_serial"] for n in nodes} == {serial}
    assert r["board_pairs"]["0000046131934103"] == [0, 1]
    assert r["board_pairs"]["000004613193410D"] == [2, 3]


def test_every_node_has_a_distinct_asic():
    asics = [d["asic"] for d in N.NODES.values()]
    assert len(set(asics)) == len(asics)


def test_the_gap_is_the_difference_of_the_two_quoted_folds():
    q = N.analyse()["the_open_question"]
    assert q["gap_s"] == pytest.approx(N.BASELINE_NODE0["fold_s"] - N.HISTORICAL_NODE1["fold_s"])
    assert q["gap_pct"] == pytest.approx(2.2, abs=0.05)


def test_the_firmware_confound_is_gone():
    c = N.analyse()["the_confound_is_gone"]
    assert c["match"] is True
    assert c["node0_firmware_recorded"] == c["node1_firmware_now"] == "19.15.0.0"
    assert "CLEAN ONE-VARIABLE" in c["reading"]


def test_a_firmware_mismatch_would_say_the_control_is_not_clean(monkeypatch):
    """Negative control: the cleanliness is derived, not asserted."""
    nodes = {k: dict(v) for k, v in N.NODES.items()}
    nodes[1]["fw_now"] = "19.11.0.0"
    monkeypatch.setattr(N, "NODES", nodes)
    c = N.analyse()["the_confound_is_gone"]
    assert c["match"] is False
    assert "NOT clean" in c["reading"]


def test_the_historical_node1_fold_really_carries_the_older_firmware():
    assert N.HISTORICAL_NODE1["fw"] == "19.11.0.0"
    assert N.HISTORICAL_NODE1["node"] == 1
    assert N.BASELINE_NODE0["node"] == 0


def test_node0_is_recorded_as_quarantined_and_its_firmware_is_quoted_not_read():
    r = N.analyse()
    assert "QUARANTINED" in N.NODES[0]["state"]
    assert N.NODES[0]["fw_now"] is None, "node 0 sysfs returns ERR; its firmware must be quoted"
    assert any("quoted from c10-fixed-cost's record" in l for l in r["limits"])


def test_all_numbers_of_record_are_listed_as_node0():
    r = N.analyse()
    joined = " ".join(r["numbers_of_record_are_all_node0"])
    for n in ("14.8813", "3.9830", "14665.0", "31.5522", "9.6801"):
        assert n in joined


def test_both_interpretations_are_given_and_neither_is_preferred():
    i = N.analyse()["interpretations"]
    assert len(i) >= 3
    assert "equivalent" in i[0] and "differ" in i[1]


def test_the_board_pair_is_not_claimed_as_an_equivalence():
    lims = " ".join(N.analyse()["limits"]).lower()
    assert "not mean they are interchangeable" in lims
    assert "reset granularity" in lims
