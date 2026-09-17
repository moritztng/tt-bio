#!/usr/bin/env python3
"""Controls for arithmetic_free_traffic.py."""
import json

import pytest

import arithmetic_free_traffic as A


def _fake(tmp_path, monkeypatch, by_op, buckets=None):
    c, f = A.load()
    c = dict(c, by_op=by_op)
    if buckets is not None:
        f = dict(f, buckets=buckets)
    pc, pf = tmp_path / "c.json", tmp_path / "f.json"
    pc.write_text(json.dumps(c))
    pf.write_text(json.dumps(f))
    monkeypatch.setattr(A, "CENSUS", pc)
    monkeypatch.setattr(A, "FLOOR", pf)


# --- known answer -----------------------------------------------------------------------------
def test_recovers_a_planted_zero_arithmetic_share_exactly(tmp_path, monkeypatch):
    by = {
        "compute": {"calls": 100, "B": 6e12, "s_traffic": 6.0, "s_arith": 9.0},
        "mover":   {"calls": 300, "B": 4e12, "s_traffic": 4.0, "s_arith": 0.0},
        "booking": {"calls": 600, "B": 0.0,  "s_traffic": 0.0, "s_arith": 0.0},
    }
    _fake(tmp_path, monkeypatch, by)
    r = A.analyse()
    m = r["zero_arithmetic"]["byte_moving"]
    assert m["calls"] == 300 and m["TB"] == pytest.approx(4.0)
    assert m["pct_of_calls"] == pytest.approx(30.0)
    assert m["pct_of_bytes"] == pytest.approx(40.0)
    assert r["zero_arithmetic"]["bookkeeping_zero_byte"]["calls"] == 600


def test_the_two_zero_arith_groups_partition_the_zero_arith_ops():
    c, _ = A.load()
    by = c["by_op"]
    zero = [v for v in by.values() if v["s_arith"] == 0]
    r = A.analyse()["zero_arithmetic"]
    assert (r["byte_moving"]["calls"] + r["bookkeeping_zero_byte"]["calls"]
            == pytest.approx(sum(v["calls"] for v in zero)))


def test_bookkeeping_ops_really_move_zero_bytes():
    assert A.analyse()["zero_arithmetic"]["bookkeeping_zero_byte"]["TB"] == 0.0


def test_it_reads_the_artifacts_rather_than_restating_them(tmp_path, monkeypatch):
    c, _ = A.load()
    by = {k: dict(v, s_arith=1.0) for k, v in c["by_op"].items()}   # nothing is zero-arithmetic now
    _fake(tmp_path, monkeypatch, by)
    r = A.analyse()
    assert r["zero_arithmetic"]["byte_moving"]["calls"] == 0


# --- the instruments agree on graph facts and disagree on seconds -----------------------------
def test_the_two_instruments_agree_on_the_fold_totals():
    a = A.analyse()["instrument_agreement"]
    assert a["fold_calls"]["census"] == a["fold_calls"]["floor"]
    assert a["fold_TB"]["census"] == pytest.approx(a["fold_TB"]["floor"], rel=1e-5), \
        "byte totals agree to 3 ppm, not exactly; README says so"


def test_zero_arith_call_counts_agree_within_one_percent():
    a = A.analyse()["instrument_agreement"]["zero_arith_calls"]
    assert abs(a["census"] - a["floor"]) / a["floor"] < 0.01


def test_the_byte_attribution_disagreement_is_real_and_must_stay_stated():
    """If the two instruments are ever reconciled, the README's 'composition is safe, any single
    number is not' framing is wrong and has to be rewritten."""
    a = A.analyse()["instrument_agreement"]["zero_arith_TB"]
    assert abs(a["census"] - a["floor"]) / a["floor"] > 0.15


def test_the_bracket_is_ordered_and_contains_the_measured_clock_immune_term():
    c = A.analyse()["cost_bracket_s"]
    assert c["low"] < c["high"]
    assert c["F_inside_bracket"] is True


# --- staleness guards -------------------------------------------------------------------------
def test_half_the_bytes_still_do_almost_none_of_the_arithmetic():
    r = A.analyse()
    assert r["zero_arithmetic"]["byte_moving"]["pct_of_bytes"] > 40.0
    assert r["fold_totals"]["nonmatmul_pct_of_FLOP"] < 0.1


def test_the_candidate_is_still_concentrated_in_three_op_classes():
    h = A.analyse()["headline_candidate"]
    assert len(h["ops"]) == 3
    assert h["pct_of_fold_bytes"] > 25.0


def test_the_prize_is_still_a_third_of_the_traffic_and_not_all_of_it():
    """Guards against the realistic prize quietly being promoted to the full traffic figure."""
    h = A.analyse()["headline_candidate"]
    for prize, traffic in zip(h["realistic_prize_s_bracket"], h["traffic_s_bracket"]):
        assert prize == pytest.approx(traffic / 3.0, rel=1e-9)
    assert A.FUSION_RETURN == pytest.approx(1 / 3)


# --- negative controls ------------------------------------------------------------------------
def test_giving_the_movers_arithmetic_empties_the_class(tmp_path, monkeypatch):
    c, _ = A.load()
    by = {k: (dict(v, s_arith=2.0) if k in ("ttnn.multiply_", "ttnn.add_", "ttnn.layer_norm") else v)
          for k, v in c["by_op"].items()}
    _fake(tmp_path, monkeypatch, by)
    r = A.analyse()
    assert r["zero_arithmetic"]["byte_moving"]["pct_of_bytes"] < 25.0
    assert not any(o["op"] in ("ttnn.multiply_", "ttnn.add_")
                   for o in r["concentration"][:3])


def test_a_fold_whose_movers_carry_real_flops_breaks_the_headline(tmp_path, monkeypatch):
    c, f = A.load()
    _fake(tmp_path, monkeypatch, c["by_op"])
    monkeypatch.setattr(A, "F_MEASURED_S", 0.5)
    assert A.analyse()["cost_bracket_s"]["F_inside_bracket"] is False
