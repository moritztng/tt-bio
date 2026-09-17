#!/usr/bin/env python3
"""Controls for gate_coverage.py."""
import json

import pytest

import gate_coverage as G


def test_constants_are_parsed_from_the_gate_not_restated():
    src = G.GATE.read_text()
    c = G.analyse()["step_constants"]
    for name, val in c.items():
        assert val is not None, f"{name} no longer parses out of release_gate.py"
        assert f"{name} = {val}" in src


def test_the_parse_follows_the_file_when_the_file_changes(tmp_path, monkeypatch):
    src = G.GATE.read_text().replace("SIZE_LADDER_STEPS = 6", "SIZE_LADDER_STEPS = 200", 1)
    p = tmp_path / "gate.py"
    p.write_text(src)
    monkeypatch.setattr(G, "GATE", p)
    assert G.analyse()["step_constants"]["SIZE_LADDER_STEPS"] == 200


def test_fixture_sizes_are_read_from_the_yaml():
    legs = {l["leg"]: l for l in G.analyse()["boltz2_legs"]}
    assert legs["accuracy (per-model fold)"]["size_aa"] == 117
    assert legs["l1-budget"]["size_aa"] == 107


def test_the_size_ladder_rungs_parse_and_include_768():
    legs = {l["leg"]: l for l in G.analyse()["boltz2_legs"]}
    rungs = legs["size-ladder"]["size_aa"]
    assert isinstance(rungs, list) and 768 in rungs and 512 in rungs


# --- the finding -------------------------------------------------------------------------------
def test_no_boltz2_target_above_117_aa_is_folded_at_production_steps():
    g = G.analyse()["gap"]
    assert g["largest_boltz2_target_folded_at_production_steps_aa"] == 117
    assert g["their_steps"] == [6]
    assert g["step_ratio"] == pytest.approx(200 / 6, rel=1e-9)


def test_the_wedge_size_is_outside_production_step_coverage():
    r = G.analyse()
    assert r["incident"]["size_aa"] > r["gap"]["largest_boltz2_target_folded_at_production_steps_aa"]
    assert r["incident_in_the_gap"]["covered"] is False


# --- staleness guard: if the gap is ever closed, this note must be rewritten --------------------
def test_adding_a_production_length_leg_above_512_closes_the_gap(tmp_path, monkeypatch):
    """Negative control AND staleness guard. If someone raises SIZE_LADDER_STEPS to the production
    count, the gap disappears and the README's central claim stops being true."""
    src = G.GATE.read_text().replace("SIZE_LADDER_STEPS = 6", "SIZE_LADDER_STEPS = 200", 1)
    p = tmp_path / "gate.py"
    p.write_text(src)
    monkeypatch.setattr(G, "GATE", p)
    r = G.analyse()
    assert r["gap"]["largest_boltz2_target_folded_at_production_steps_aa"] >= 768
    assert r["incident_in_the_gap"]["covered"] is True


def test_the_incident_record_matches_what_the_measuring_row_reported():
    i = G.analyse()["incident"]
    assert (i["size_aa"], i["steps"], i["recycles"], i["msa_rows"]) == (768, 200, 3, 35)
    assert i["folds_before_failure"] == 5
    assert i["card"] == "p300c"


def test_the_note_does_not_blame_an_arm_for_doing_what_it_says():
    t = G.analyse()["not_a_criticism_of_any_arm"].lower()
    assert "composition" in t and "doing exactly what it claims" in t


def test_the_proposed_fix_is_flagged_as_release_gated_and_not_implemented():
    fixes = " ".join(G.analyse()["minimal_fix_not_implemented_here"]).lower()
    assert "gated" in fixes and "moritz" in fixes


def test_limits_disclaim_a_failure_rate_from_one_occurrence():
    assert any("is not a rate" in x for x in G.analyse()["limits"])
