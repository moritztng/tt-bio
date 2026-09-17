#!/usr/bin/env python3
"""Controls for frontier.py."""
import json
import statistics as st

import pytest

import frontier as Fr


# --- the budget is exact arithmetic on two measured terms --------------------------------------
def test_the_frontier_identity_holds():
    r = Fr.analyse()
    need = r["requirement"]["must_remove_s"]
    assert r["measured"]["F_s"] + r["measured"]["W_s_at_pin"] - need == pytest.approx(10.0, abs=1e-9)
    assert r["requirement"]["W_alone"]["cut_Mcyc"] == pytest.approx(need * Fr.PIN, rel=1e-12)


def test_the_fold_reproduces_c10_fixed_costs_own_reading():
    assert Fr.analyse()["measured"]["fold_s"] == pytest.approx(14.846, abs=0.002)


def test_F_alone_cannot_reach_the_target_and_says_so():
    q = Fr.analyse()["requirement"]["F_alone"]
    assert q["possible"] is False and q["required_F_s"] < 0


def test_a_fold_with_a_huge_F_could_reach_the_target_on_F_alone(monkeypatch):
    """Negative control: the impossibility is a fact about THIS fold, not a hardcoded verdict."""
    monkeypatch.setattr(Fr, "F_S", 9.0)
    monkeypatch.setattr(Fr, "W_MCYC", 7900.0)
    q = Fr.analyse()["requirement"]["F_alone"]
    assert q["possible"] is True and q["required_F_s"] > 0


# --- the lever total ---------------------------------------------------------------------------
def test_lever_total_is_the_sum_of_its_rows_and_is_far_short():
    r = Fr.analyse()
    assert r["levers_total"]["fold_s"] == pytest.approx(sum(v for _, v, _ in Fr.LEVERS), rel=1e-12)
    assert r["levers_total"]["pct_of_requirement"] < 25.0


def test_the_measured_zero_rows_are_still_recorded_as_zero():
    """The trace measured to zero and the grid lever was withdrawn. If either ever reappears with a
    number, it must come from a measurement and this list has to be rewritten."""
    d = {k: v for k, v, _ in Fr.LEVERS}
    assert d["ttnn trace of the diffusion loop"] == 0.0
    assert d["per-class grid sizing"] == 0.0


# --- the matmul axis ---------------------------------------------------------------------------
def test_matmul_achieved_rate_is_derived_from_the_artifact_not_hardcoded(tmp_path, monkeypatch):
    d = json.loads(Fr.FLOOR.read_text())
    d["matmul_TFLOP"] = d["matmul_TFLOP"] * 2
    p = tmp_path / "f.json"
    p.write_text(json.dumps(d))
    monkeypatch.setattr(Fr, "FLOOR", p)
    assert Fr.analyse()["matmul_rate_axis"]["achieved_TFLOPs"] == pytest.approx(
        2 * 19.878, rel=1e-3)


def test_the_matmul_axis_is_the_only_one_big_enough():
    r = Fr.analyse()
    need = r["requirement"]["must_remove_s"]
    m = r["matmul_rate_axis"]
    assert min(m["saving_if_at_cube_s"].values()) > need, "closing the rate gap must overshoot"
    assert r["levers_total"]["fold_s"] < need, "no other axis reaches it"
    assert 0 < m["pct_of_the_gap_to_close"] < 100


def test_the_rate_needed_for_the_target_is_consistent_with_the_saving():
    m = Fr.analyse()["matmul_rate_axis"]
    need = Fr.analyse()["requirement"]["must_remove_s"]
    assert m["matmul_TFLOP_per_fold"] / m["rate_needed_for_target_TFLOPs"] == pytest.approx(
        m["modelled_seconds"] - need, rel=1e-9)


# --- the fourth clock artifact ------------------------------------------------------------------
def test_the_cube_cluster_is_tight_and_the_outlier_is_not_in_it():
    c = Fr.analyse()["matmul_rate_axis"]["cube_cluster_TFLOPs"]
    assert c["n"] >= 5
    assert c["spread_pct"] < 15.0, "the cluster must be tight for the outlier to mean anything"
    assert Fr.CUBE_OUTLIER < c["min"] * 0.8


def test_the_outlier_implies_a_clock_at_the_800_MHz_floor():
    a = Fr.analyse()["fourth_clock_artifact"]
    assert abs(a["implied_clock_vs_max_MHz"] - Fr.FLOOR_MHZ) < 40.0, (
        "the outlier no longer lands on the clock floor; the artifact reading has to be rewritten")


def test_an_outlier_inside_the_cluster_would_imply_burst_clock_and_no_artifact(monkeypatch):
    monkeypatch.setattr(Fr, "CUBE_OUTLIER", 113.0)
    a = Fr.analyse()["fourth_clock_artifact"]
    assert a["implied_clock_vs_max_MHz"] > 1250.0


def test_implied_clock_is_computed_per_cluster_member_not_asserted():
    a = Fr.analyse()["fourth_clock_artifact"]
    for src, v in Fr.CUBE_CLUSTER.items():
        assert a["implied_clock_MHz"][src] == pytest.approx(Fr.PIN * Fr.CUBE_OUTLIER / v, rel=1e-12)
    assert all(700 < x < 900 for x in a["implied_clock_MHz"].values())
