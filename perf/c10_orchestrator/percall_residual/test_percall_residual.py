#!/usr/bin/env python3
"""Controls for percall_residual.py."""
import json

import pytest

import percall_residual as P


def test_the_lever_arm_is_wide_enough_to_tell_the_two_hypotheses_apart():
    r = P.analyse()
    assert r["lever_arm"]["range_x"] > 500, "a narrow FLOP range cannot distinguish rate from fixed"


def test_the_residual_is_flat_against_a_1152x_flop_range():
    """The finding. A rate deficit would make residual track FLOPs; it must not."""
    r = P.analyse()
    assert r["residual_us"]["excluding_tiny_shapes"]["range_x"] < 10
    assert r["lever_arm"]["range_x"] / r["residual_us"]["excluding_tiny_shapes"]["range_x"] > 100


def test_a_synthetic_rate_deficit_is_NOT_read_as_flat(tmp_path, monkeypatch):
    """Negative control: build a census whose model time is a constant multiple of roofline time,
    i.e. a pure rate deficit. The residual must then track FLOPs across the range."""
    census = json.loads(P.CENSUS.read_text())
    floor = json.loads(P.FLOOR.read_text())
    cube, bw = floor["roofs"]["dense_cube_TFLOPs"], floor["roofs"]["stream_GBps"] / 1e3
    shapes = {}
    for k, v in census["top_shapes"].items():
        if not k.startswith(("ttnn.linear", "ttnn.matmul")):
            continue
        m, kk, n = P._terms(k)
        flops = 2 * m * kk * n
        byts = (m * kk + kk * n + m * n) * P.BYTES_PER_ELEM
        roof_s = flops / (min(cube, flops / byts * bw) * 1e12)
        shapes[k] = dict(v, s_floor=roof_s * 4.0 * v["calls"])     # 4x slower, everywhere
    p = tmp_path / "c.json"
    p.write_text(json.dumps(dict(census, top_shapes=shapes)))
    monkeypatch.setattr(P, "CENSUS", p)
    r = P.analyse()
    synthetic = r["residual_us"]["excluding_tiny_shapes"]["range_x"]
    # Under a pure rate deficit the residual is a constant multiple of roofline TIME, so its range
    # is the roofline-time range over the same shapes -- 36x here, not the 1152x FLOP range, because
    # the DRAM-bound shapes compress it. The point is that it is an order of magnitude above what
    # the real data shows.
    real = 4.3
    assert synthetic > 8 * real, (
        f"a pure rate deficit must spread the residual far more than the real {real}x; got "
        f"{synthetic:.1f}x")


def test_a_synthetic_fixed_cost_IS_read_as_flat(tmp_path, monkeypatch):
    census = json.loads(P.CENSUS.read_text())
    floor = json.loads(P.FLOOR.read_text())
    cube, bw = floor["roofs"]["dense_cube_TFLOPs"], floor["roofs"]["stream_GBps"] / 1e3
    shapes = {}
    for k, v in census["top_shapes"].items():
        if not k.startswith(("ttnn.linear", "ttnn.matmul")):
            continue
        m, kk, n = P._terms(k)
        flops = 2 * m * kk * n
        byts = (m * kk + kk * n + m * n) * P.BYTES_PER_ELEM
        roof_s = flops / (min(cube, flops / byts * bw) * 1e12)
        shapes[k] = dict(v, s_floor=(roof_s + 20e-6) * v["calls"])   # exactly 20 us each
    p = tmp_path / "c.json"
    p.write_text(json.dumps(dict(census, top_shapes=shapes)))
    monkeypatch.setattr(P, "CENSUS", p)
    b = P.analyse()["residual_us"]["excluding_tiny_shapes"]
    assert b["median"] == pytest.approx(20.0, abs=0.01)
    assert b["sd"] < 0.01


def test_totals_decompose_the_modelled_time_exactly():
    t = P.analyse()["total"]
    assert t["roofline_s"] + t["residual_s"] == pytest.approx(t["modelled_s"], rel=1e-9)


def test_the_launch_floor_does_not_explain_the_residual():
    l = P.analyse()["launch_does_not_explain_it"]
    assert l["us_per_program"] == pytest.approx(P.LAUNCH_FLOOR_S / P.LAUNCH_PROGRAMS * 1e6)
    assert l["residual_over_launch_x"] > 5


# --- the alternative explanation must be stated, and no real per-call cost claimed --------------
def test_the_instrument_alternative_is_stated_and_no_qb2_claim_is_made():
    r = P.analyse()
    alt = r["ALTERNATIVE_the_instrument_added_it"]
    assert "ON PC" in alt and "does NOT claim" in alt
    assert "130-core" in alt


def test_it_makes_a_falsifiable_two_branch_prediction():
    p = P.analyse()["PREDICTED_for_the_measuring_row"]
    assert len(p) >= 3
    joined = " ".join(p).lower()
    assert "if the per-call cost is real" in joined and "if it was pc's instrument" in joined
    assert "3x" in joined


def test_tiny_shapes_are_excluded_from_the_headline_but_kept_in_the_table():
    r = P.analyse()
    tiny = [s for s in r["shapes"] if s["flops_per_call"] < P.SMALL_FLOP_CUTOFF]
    assert tiny, "the tiny shapes must still appear"
    assert r["residual_us"]["all"]["n"] > r["residual_us"]["excluding_tiny_shapes"]["n"]


def test_an_empty_census_says_no_data_rather_than_dividing_by_zero(tmp_path, monkeypatch):
    census = json.loads(P.CENSUS.read_text())
    census["top_shapes"] = {k: v for k, v in census["top_shapes"].items()
                            if not k.startswith(("ttnn.linear", "ttnn.matmul"))}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(census))
    monkeypatch.setattr(P, "CENSUS", p)
    assert P.analyse()["verdict"] == "NO DATA"
