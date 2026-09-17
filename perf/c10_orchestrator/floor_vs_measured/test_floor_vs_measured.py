#!/usr/bin/env python3
"""Controls for floor_vs_measured.py."""
import json

import pytest

import floor_vs_measured as M


def test_it_reads_the_committed_artifact_rather_than_restating_it(tmp_path, monkeypatch):
    """Mutate the floor JSON and the conclusion must move. A module that hardcoded these numbers
    would silently keep reporting the old ones after the artifact was regenerated."""
    d = json.loads(M.FLOOR_JSON.read_text())
    d["floor_s"] = 9.0
    p = tmp_path / "fake.json"
    p.write_text(json.dumps(d))
    monkeypatch.setattr(M, "FLOOR_JSON", p)
    r = M.analyse()
    assert r["prize"]["floor_s"] == 9.0
    assert r["prize"]["prize_survives"] is True, "a 9 s floor must revive the prize"


def test_the_prize_identity_holds_in_the_committed_file():
    d = M.load()
    assert d["roofs"]["cell_of_record_s"] - d["floor_s"] == pytest.approx(d["prize_s"], abs=1e-6)
    assert M.analyse()["prize"]["identity_holds"] is True


def test_the_floors_two_halves_sum_to_the_floor():
    """If they do not, the traffic/arithmetic split is not a partition of the floor and the whole
    comparison against F and W is meaningless."""
    d = M.load()
    assert d["s_set_by_traffic"] + d["s_set_by_arithmetic"] == pytest.approx(d["floor_s"], rel=1e-9)


def test_measured_fold_reproduces_c10_fixed_costs_own_reading():
    assert M.analyse()["measured"]["fold_s_at_pin"] == pytest.approx(14.846, abs=0.002)


def test_a_synthetic_fold_exactly_at_its_floor_reads_100_percent_on_both_halves(monkeypatch,
                                                                                tmp_path):
    d = json.loads(M.FLOOR_JSON.read_text())
    p = tmp_path / "exact.json"
    d["s_set_by_traffic"] = M.F_512_S
    d["s_set_by_arithmetic"] = M.W_512_MCYC / M.PIN_MHZ
    d["floor_s"] = d["s_set_by_traffic"] + d["s_set_by_arithmetic"]
    p.write_text(json.dumps(d))
    monkeypatch.setattr(M, "FLOOR_JSON", p)
    t = M.analyse()["two_terms"]
    assert t["clock_immune"]["measured_pct_of_modelled"] == pytest.approx(100.0, abs=1e-9)
    assert t["clock_scaled"]["measured_pct_of_modelled"] == pytest.approx(100.0, abs=1e-9)


def test_tension_arithmetic_is_exact():
    r = M.analyse()["tension_with_size_scaling"]
    wr = r["work_ratio_measured"]
    for _, v in r["by_flop_model"].items():
        assert v["required_rate_ratio_512_over_298"] * wr == pytest.approx(v["flop_ratio"], rel=1e-12)
        assert v["298_achieves_pct_of_512_rate"] == pytest.approx(100.0 / v["required_rate_ratio_512_over_298"], rel=1e-12)


# --- staleness guards -------------------------------------------------------------------------
def test_the_fold_at_a_pinned_clock_is_still_BELOW_the_floor_the_prize_was_measured_against():
    """This is the retirement. If the fold ever reads above floor_s again the prize is back and
    this directory's central claim has to be rewritten."""
    p = M.analyse()["prize"]
    assert p["prize_survives"] is False
    assert p["measured_minus_floor_s"] < 0


def test_the_shape_rates_are_still_the_ones_measured_on_the_wrong_machine():
    """The provenance caveat says the floor's arithmetic half came from pc's 130-core firmware. If
    the artifact is ever regenerated on qb2 that caveat is wrong and must be rewritten."""
    assert M.load()["roofs"]["shape_rate_host"] == "pc"


def test_both_halves_still_agree_within_the_bounds_the_readme_quotes():
    t = M.analyse()["two_terms"]
    assert 85.0 < t["clock_immune"]["measured_pct_of_modelled"] < 110.0
    assert 95.0 < t["clock_scaled"]["measured_pct_of_modelled"] < 105.0


def test_the_tension_still_points_at_under_fill_rather_than_away_from_it():
    """Under an N^2 FLOP model the required rate falloff at 298 aa must stay large enough to be a
    live alternative explanation for size_scaling's finding."""
    v = M.analyse()["tension_with_size_scaling"]["by_flop_model"]["FLOPs ~ N^2"]
    assert v["298_achieves_pct_of_512_rate"] < 70.0


# --- negative controls ------------------------------------------------------------------------
def test_a_much_larger_F_breaks_the_clock_immune_agreement(monkeypatch):
    monkeypatch.setattr(M, "F_512_S", 6.5)
    t = M.analyse()["two_terms"]["clock_immune"]
    assert t["measured_pct_of_modelled"] > 110.0


def test_a_much_larger_W_breaks_the_clock_scaled_agreement(monkeypatch):
    monkeypatch.setattr(M, "W_512_MCYC", 19000.0)
    t = M.analyse()["two_terms"]["clock_scaled"]
    assert t["measured_pct_of_modelled"] > 105.0


def test_a_work_ratio_equal_to_the_flop_ratio_would_dissolve_the_tension(monkeypatch):
    monkeypatch.setattr(M, "WORK_RATIO", M.TOKEN_RATIO ** 2)
    v = M.analyse()["tension_with_size_scaling"]["by_flop_model"]["FLOPs ~ N^2"]
    assert v["298_achieves_pct_of_512_rate"] == pytest.approx(100.0, rel=1e-9)
