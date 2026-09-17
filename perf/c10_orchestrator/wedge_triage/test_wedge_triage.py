#!/usr/bin/env python3
"""Controls for wedge_triage.py."""
import json

import pytest

import wedge_triage as W


def test_operand_match_is_exact_not_substring():
    """1x1x768x512 CONTAINS 1x768x512 as a substring. A naive `in` test would match a tensor that
    is not the flagged shape, which is how a triage note ends up pointing at the wrong op."""
    assert "1x768x512" in "1x1x768x512"          # the trap this guards
    dims = W._dims("ttnn.permute|out=|in=1x1x768x512")
    assert not any("x".join(str(d) for d in t) == "1x768x512" for t in dims)


def test_the_flagged_rows_each_really_carry_the_shape_as_a_whole_operand():
    for row in W.analyse()["second_sighting_shape_is_in_this_fold"]["rows"]:
        toks = {"x".join(str(d) for d in t) for t in W._dims(row["shape"])}
        assert W.SECOND_SIGHTING_SHAPE in toks, row["shape"]


def test_dims_parses_every_tensor_in_a_signature():
    d = W._dims("ttnn.reshape|out=1x768x512|in=1x16x48x512,16x48x512")
    assert d == [[1, 768, 512], [1, 16, 48, 512], [16, 48, 512]]


def test_sub_tile_detection_finds_the_48_wide_per_head_axis():
    c = W.analyse()["sub_tile_last_axis_candidates"]
    assert len(c) == 1
    assert c[0]["sub_tile_last_axis_extents"] == [48]
    assert "permute" in c[0]["op"]


def test_a_genuinely_sub_tile_last_axis_is_detected(monkeypatch, tmp_path):
    """Positive control: plant the known class's own signature and it must be flagged."""
    census = json.loads(W.CENSUS.read_text())
    census["top_shapes"]["ttnn.chunk|out=|in=1x1x4096x16"] = {
        "calls": 10, "B": 6e8, "s_floor": 0.1, "in_tiles": 100, "out_tiles": 0}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(census))
    monkeypatch.setattr(W, "CENSUS", p)
    c = W.analyse()["sub_tile_last_axis_candidates"]
    assert any(r["sub_tile_last_axis_extents"] == [16] for r in c)


def test_the_verdict_is_derived_from_the_candidate_list_not_asserted():
    r = W.analyse()
    n = len(r["sub_tile_last_axis_candidates"])
    assert f"contain {n} such shape" in r["verdict"]
    assert "48" in r["verdict"]


def test_the_verdict_neither_confirms_nor_excludes():
    v = W.analyse()["verdict"].lower()
    assert "not matched" in v and "not excluded" in v


def test_the_counter_evidence_is_the_campaigns_own_clean_folds():
    s = W.analyse()["second_sighting_shape_is_in_this_fold"]
    assert s["clean_total"] == sum(W.CLEAN_512_FOLDS.values())
    assert s["executions_without_a_hang"] == s["calls_per_512aa_fold"] * s["clean_total"]
    assert s["executions_without_a_hang"] > 100_000


def test_it_does_not_claim_a_root_cause_or_a_rate():
    r = W.analyse()
    assert "no root cause claimed" in r["scope"]
    assert any("is not a rate" in x for x in r["limits"])
    assert any("Nothing here was run" in x for x in r["limits"])


def test_the_next_step_is_a_768_aa_census_and_it_needs_a_chip():
    nxt = W.analyse()["for_the_next_worker_with_a_chip"]
    assert "768 aa census" in nxt[0] and "needs a chip" in nxt[0]


def test_both_prior_sightings_are_named():
    k = W.analyse()["known_class"]
    assert k["prior_sightings"] == 2 and len(k["memories"]) == 2


def test_a_shape_that_only_CONTAINS_the_flagged_token_is_not_flagged(monkeypatch, tmp_path):
    """The real census happens to agree under both matching rules, so the exactness control above
    is vacuous on it. Plant a census whose only candidate is `1x1x768x512` -- which contains
    `1x768x512` as a substring but is a different tensor -- and it must NOT be flagged."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "total_calls": 10, "total_floor_s": 1.0, "by_op": {},
        "top_shapes": {"ttnn.permute|out=|in=1x1x768x512": {
            "calls": 10, "B": 1e9, "s_floor": 0.1, "in_tiles": 100, "out_tiles": 0}},
    }))
    monkeypatch.setattr(W, "CENSUS", p)
    r = W.analyse()
    assert r["second_sighting_shape_is_in_this_fold"]["rows"] == [], (
        "1x1x768x512 was flagged as the 1x768x512 shape -- matching is substring, not exact")
