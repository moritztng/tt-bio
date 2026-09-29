"""`tt_bio.bcinputs`: the inputs a BindCraft 2 campaign is refused for.

Each test is one input a researcher plausibly brings that used to run to completion and design
against something nobody asked for, measured on qb2 2026-09-29 (`state/bgx-inputs.md`). The
targets are built here rather than shipped, so a failure can be read from the test alone.
"""
import pathlib

import pytest

from tt_bio import bcinputs

HPDL1 = pathlib.Path("/home/ttuser/bcx_e2e/bc2/settings/target/structures/hPDL1.pdb")


def _bindcraft():
    pytest.importorskip("bindcraft", reason="BindCraft 2 is not on sys.path")
    if not HPDL1.is_file():
        pytest.skip("BindCraft 2's shipped hPDL1 target is not on this machine")
    return HPDL1.read_text().splitlines()


def _target(tmp_path, name, keep=lambda number: True):
    """hPDL1 (residues 18-132 of chain A), with the residues `keep` rejects left out."""
    lines = [line for line in _bindcraft()
             if line.startswith("ATOM") and keep(int(line[22:26]))]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def _settings(path, **target):
    return {"binder_lengths": [60, 90],
            "targets": [{"name": "T", "target_path": str(path), **target}]}


def _problems(settings):
    problems, notes = bcinputs.input_problems(settings)
    return problems, notes


# ---------------------------------------------------------------- hotspots, the silent case

def test_a_hotspot_in_an_unresolved_loop_is_refused_and_names_the_stretch(tmp_path):
    path = _target(tmp_path, "gap.pdb", keep=lambda n: not 60 <= n <= 70)
    problems, _ = _problems(_settings(path, hotspots="54,56,66,115"))
    assert len(problems) == 1
    assert "hotspot 66" in problems[0] and "unresolved stretch 60-70" in problems[0]


def test_hotspots_outside_the_gap_still_resolve(tmp_path):
    path = _target(tmp_path, "gap.pdb", keep=lambda n: not 60 <= n <= 70)
    assert _problems(_settings(path, hotspots="54,56,115")) == ([], [])


def test_a_range_across_a_gap_keeps_what_resolved_and_says_what_did_not(tmp_path):
    path = _target(tmp_path, "gap.pdb", keep=lambda n: not 60 <= n <= 70)
    problems, notes = _problems(_settings(path, hotspots="54-70"))
    assert problems == []
    assert len(notes) == 1 and "11 of 17" in notes[0] and "60-70" in notes[0]


def test_a_hotspot_the_structure_does_not_reach_is_refused_with_its_range(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    problems, _ = _problems(_settings(path, hotspots="999"))
    assert len(problems) == 1 and "residues run 18 to 132" in problems[0]


def test_a_hotspot_numbered_from_one_on_a_file_numbered_from_eighteen_is_refused(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    problems, _ = _problems(_settings(path, hotspots="5"))
    assert len(problems) == 1 and "own residue numbering" in problems[0]


def test_a_hotspot_on_a_chain_the_campaign_does_not_design_against_is_refused(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    problems, _ = _problems(_settings(path, hotspots="B54"))
    assert len(problems) == 1 and "chain B is not in its target" in problems[0]


def test_a_coldspot_is_checked_the_same_way(tmp_path):
    path = _target(tmp_path, "gap.pdb", keep=lambda n: not 60 <= n <= 70)
    problems, _ = _problems(_settings(path, hotspots="54", coldspots="62"))
    assert len(problems) == 1 and "coldspot 62" in problems[0]


def test_a_malformed_span_is_left_to_bindcrafts_own_error(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    assert _problems(_settings(path, hotspots="fifty-four")) == ([], [])


def test_a_file_that_is_not_there_is_left_to_bindcrafts_own_error():
    _bindcraft()
    assert _problems(_settings("/tmp/bgx-no-such-target.pdb", hotspots="54")) == ([], [])


def test_a_gzipped_target_says_to_unpack_it(tmp_path):
    import gzip
    path = _target(tmp_path, "whole.pdb")
    gz = tmp_path / "hPDL1.pdb.gz"
    gz.write_bytes(gzip.compress(path.read_bytes()))
    problems, _ = _problems(_settings(gz, hotspots="54"))
    assert len(problems) == 1 and "gunzip -k hPDL1.pdb.gz" in problems[0]


def test_a_binary_target_is_named_as_not_text(tmp_path):
    _bindcraft()
    path = tmp_path / "target.bcif"
    path.write_bytes(bytes(range(256)) * 20)
    problems, _ = _problems(_settings(path, hotspots="54"))
    assert len(problems) == 1 and "is not text" in problems[0]


# ---------------------------------------------------------------- settings, checked without a file

def test_a_confidence_threshold_written_as_a_percentage_is_refused():
    problems = bcinputs.threshold_problems({"min_plddt_final": 80})
    assert len(problems) == 1 and "write 0.8" in problems[0]


def test_a_filter_threshold_above_one_is_refused():
    problems = bcinputs.threshold_problems({"filters": {"i_pTM": {"threshold": 5.0}}})
    assert len(problems) == 1 and "cannot exceed 1" in problems[0]


def test_the_shipped_defaults_are_not_refused():
    assert bcinputs.threshold_problems(
        {"min_plddt_final": 0.7, "min_iptm_final": 0.7, "max_ipae_final": 0.35,
         "filters": {"i_pTM": {"threshold": 0.7, "higher": True},
                     "Interface_Residues": {"threshold": 7, "higher": True}}}) == []


def test_a_binder_length_written_as_a_number_says_how_to_write_it():
    problems = bcinputs.binder_length_problems({"binder_lengths": 80})
    assert len(problems) == 1 and "write [80]" in problems[0]


def test_a_binder_length_range_written_as_a_string_says_how_to_write_it():
    problems = bcinputs.binder_length_problems({"binder_lengths": "60-90"})
    assert len(problems) == 1 and "write [60, 90]" in problems[0]


def test_binder_lengths_as_a_list_is_accepted():
    assert bcinputs.binder_length_problems({"binder_lengths": [60, 90]}) == []
    assert bcinputs.binder_length_problems({"binder_lengths": [80]}) == []
    assert bcinputs.binder_length_problems({}) == []


def test_an_empty_binder_length_list_is_refused():
    assert len(bcinputs.binder_length_problems({"binder_lengths": []})) == 1


# ---------------------------------------------------------------- the refusal itself

def test_refuse_unusable_inputs_raises_with_every_problem_on_it(tmp_path):
    path = _target(tmp_path, "gap.pdb", keep=lambda n: not 60 <= n <= 70)
    with pytest.raises(ValueError) as raised:
        bcinputs.refuse_unusable_inputs(_settings(path, hotspots="62,64,66"))
    assert str(raised.value).count("unresolved stretch") == 3


def test_a_campaign_whose_inputs_are_sound_passes_through(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    bcinputs.refuse_unusable_inputs(_settings(path, hotspots="54,56,66,115"))
