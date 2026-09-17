#!/usr/bin/env python3
"""Controls for matmul_ceiling.py."""
import json

import pytest

import matmul_ceiling as M


def test_shape_terms_parses_a_known_signature():
    assert M._shape_terms("ttnn.linear|out=1x512x1536|in=1x512x768,768x1536") == (512, 768, 1536)
    assert M._shape_terms("ttnn.linear|out=1x16x512x128|in=1x16x512x512,512x128") == (
        16 * 512, 512, 128)


def test_a_known_answer_square_matmul_gets_its_exact_roofline():
    """A 4096^3 bf16 matmul: FLOPs and bytes are exact, so AI is exact."""
    m = k = n = 4096
    flops = 2 * m * k * n
    byts = (m * k + k * n + m * n) * M.BYTES_PER_ELEM
    assert flops == 137438953472
    assert byts == 100663296
    assert flops / byts == pytest.approx(1365.33, rel=1e-4)


def test_ceiling_is_the_min_of_cube_and_bandwidth_times_intensity():
    r = M.analyse()
    cube = r["roofs"]["dense_cube_TFLOPs"]
    bw = r["roofs"]["stream_GBps"] / 1e3
    for s in r["shapes"]:
        assert s["ceiling_cold_TFLOPs"] == pytest.approx(min(cube, s["ai_cold"] * bw), rel=1e-12)
        assert s["ceiling_cold_TFLOPs"] <= cube + 1e-9


def test_bound_label_agrees_with_the_machine_balance():
    r = M.analyse()
    bal = r["roofs"]["machine_balance_flop_per_byte"]
    for s in r["shapes"]:
        assert s["bound_cold"] == ("compute" if s["ai_cold"] > bal else "dram")


def test_weight_residency_is_not_what_decides_this():
    c = M.analyse()["class"]["structural_ceiling"]
    lo, hi = c["cold"]["ceiling_TFLOPs"], c["warm"]["ceiling_TFLOPs"]
    assert abs(hi - lo) / lo < 0.03, "cold and warm must bracket tightly, else the claim is weaker"


# --- the finding ------------------------------------------------------------------------------
def test_bandwidth_does_not_explain_todays_rate():
    c = M.analyse()["class"]
    assert c["structural_ceiling"]["cold"]["ceiling_TFLOPs"] > 3 * c["today_TFLOPs"]
    assert c["gap_to_structural_x"] > 3.0


def test_the_target_sits_well_inside_the_structural_ceiling():
    c = M.analyse()["class"]
    assert c["needed_for_target_TFLOPs"] < c["structural_ceiling"]["cold"]["ceiling_TFLOPs"]
    assert c["target_pct_of_structural"] < 60.0


def test_the_verdict_refuses_to_call_the_ceiling_achievable():
    r = M.analyse()
    assert "not excluded by bandwidth" in " ".join(r["limits"]).lower()
    assert "NOT 'achievable'" in " ".join(r["limits"])
    assert "implementation" in r["verdict"]


# --- negative controls -------------------------------------------------------------------------
def test_a_bandwidth_bound_fold_would_read_as_no_headroom(tmp_path, monkeypatch):
    """Drop the bandwidth roof 20x: the shapes become DRAM-bound and the ceiling collapses toward
    today's rate, which is the outcome that would have settled 10.0 s as impossible."""
    floor = json.loads(M.FLOOR.read_text())
    floor["roofs"]["stream_GBps"] = floor["roofs"]["stream_GBps"] / 20.0
    p = tmp_path / "f.json"
    p.write_text(json.dumps(floor))
    monkeypatch.setattr(M, "FLOOR", p)
    c = M.analyse()["class"]
    assert c["structural_ceiling"]["cold"]["ceiling_TFLOPs"] < c["today_TFLOPs"] * 1.5


def test_every_shape_is_derived_from_the_census_not_hardcoded(tmp_path, monkeypatch):
    census = json.loads(M.CENSUS.read_text())
    census["top_shapes"] = {k: v for k, v in census["top_shapes"].items()
                            if not k.startswith(("ttnn.linear", "ttnn.matmul"))}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(census))
    monkeypatch.setattr(M, "CENSUS", p)
    assert M.analyse()["shapes"] == []


def test_the_two_named_shape_families_are_actually_in_the_data():
    r = M.analyse()
    pair = [s for s in r["shapes"] if s["calls"] == 8448]
    assert pair and all(90 < s["ai_cold"] < 115 for s in pair), "the 16-head pair matmuls"
    fam = [s for s in r["shapes"] if s["K"] == 768 and s["M"] == 512]
    assert fam and all(200 < s["ai_cold"] < 320 for s in fam), "the 768-family linears"


# --- does the conclusion survive every roof the campaign has measured? -------------------------
def test_the_ceiling_is_insensitive_to_the_roof_range():
    s = M.analyse()["roof_sensitivity"]
    assert s["ceiling_spread_pct"] < 15.0, "if the ceiling swings wildly the finding needs the roof"
    assert len(s["rows"]) == len(M.ROOF_RANGE) >= 4


def test_the_finding_holds_under_every_roof_pair():
    """The headline is 'bandwidth does not explain the gap' and 'the target fits inside the
    ceiling'. Both must hold at EVERY recorded roof, or the finding rests on a roof choice."""
    for r in M.analyse()["roof_sensitivity"]["rows"]:
        assert r["x_today"] > 3.0, f"gap collapsed at cube={r['cube_TFLOPs']}"
        assert r["target_pct_of_ceiling"] < 60.0, f"target too close at cube={r['cube_TFLOPs']}"


def test_better_measured_roofs_make_the_finding_stronger_not_weaker():
    rows = M.analyse()["roof_sensitivity"]["rows"]
    lo, hi = rows[0], rows[-1]
    assert hi["ceiling_TFLOPs"] > lo["ceiling_TFLOPs"]
    assert hi["target_pct_of_ceiling"] < lo["target_pct_of_ceiling"]


def test_unpublished_roofs_are_labelled_as_such_and_only_bound_a_sensitivity():
    srcs = [r["source"] for r in M.analyse()["roof_sensitivity"]["rows"]]
    unpub = [s for s in srcs if "UNPUBLISHED" in s]
    assert len(unpub) == 2, "the census roofs must be marked unpublished"
    assert all("c10-fold-census" in s for s in unpub)
    reading = M.analyse()["roof_sensitivity"]["reading"]
    assert "without resting on any unpublished number" in reading


def test_a_collapsed_bandwidth_roof_would_break_the_finding(monkeypatch):
    """Negative control: the insensitivity is a property of these shapes, not a hardcoded claim."""
    monkeypatch.setattr(M, "ROOF_RANGE", M.ROOF_RANGE + [(104.93, 20.0, "synthetic: DRAM starved")])
    rows = M.analyse()["roof_sensitivity"]["rows"]
    assert rows[-1]["x_today"] < 3.0, "a starved DRAM roof must collapse the gap"
