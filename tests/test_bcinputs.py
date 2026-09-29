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


def _hetatm_only(tmp_path, name):
    """A target file whose every record is a heteroatom: a ligand-only download, or a polymer
    written as HETATM. BindCraft 2 drops all of it and then indexes an empty chain list."""
    path = tmp_path / name
    path.write_text("HETATM    1  O   HOH A 401       1.000   2.000   3.000  1.00  0.00           O\n"
                    "HETATM    2 ZN    ZN A 402       4.000   5.000   6.000  1.00  0.00          ZN\n"
                    "END\n")
    return path


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


def test_no_campaign_bindcraft_ships_is_refused():
    """The false-positive guard: 25 real campaigns, and a check that refuses one is wrong."""
    _bindcraft()
    import glob

    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.settings import read_settings

    paths = [p for p in sorted(glob.glob("/home/ttuser/bcx_e2e/bc2/examples/*.json"))
             if not p.endswith("metadata.json")]
    if not paths:
        pytest.skip("BindCraft 2's shipped examples are not on this machine")
    refused = {}
    for path in paths:
        problems, _ = bcinputs.input_problems(
            cleaned_campaign_settings(read_settings(path, {})))
        if problems:
            refused[pathlib.Path(path).name] = problems
    assert refused == {} and len(paths) >= 20


# ---------------------------------------------------------------- the refusal itself

def test_refuse_unusable_inputs_raises_with_every_problem_on_it(tmp_path):
    path = _target(tmp_path, "gap.pdb", keep=lambda n: not 60 <= n <= 70)
    with pytest.raises(ValueError) as raised:
        bcinputs.refuse_unusable_inputs(_settings(path, hotspots="62,64,66"))
    assert str(raised.value).count("unresolved stretch") == 3


def test_a_campaign_whose_inputs_are_sound_passes_through(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    bcinputs.refuse_unusable_inputs(_settings(path, hotspots="54,56,66,115"))


# ---------------------------------------------------------------- the file holds no polymer

def test_a_target_of_heteroatoms_only_is_refused_instead_of_indexing_an_empty_chain_list(tmp_path):
    path = _hetatm_only(tmp_path, "ligand_only.pdb")
    problems, _ = _problems(_settings(path, hotspots="54"))
    assert len(problems) == 1
    assert "holds no protein residue to design against" in problems[0]
    assert "HETATM" in problems[0] and "ligand_only.pdb" in problems[0]


def test_a_target_of_heteroatoms_only_is_refused_even_with_no_hotspot_asked_for(tmp_path):
    path = _hetatm_only(tmp_path, "ligand_only.pdb")
    problems, _ = _problems(_settings(path))
    assert len(problems) == 1 and "holds no protein residue" in problems[0]


# ---------------------------------------------------------------- two targets, one name

def test_two_targets_with_one_name_are_refused_because_only_the_last_is_prepared(tmp_path):
    first = _target(tmp_path, "first.pdb", keep=lambda n: not 60 <= n <= 70)
    second = _target(tmp_path, "second.pdb")
    settings = {"binder_lengths": [60],
                "targets": [{"name": "T", "target_path": str(first), "hotspots": "54"},
                            {"name": "T", "target_path": str(second), "hotspots": "56"}]}
    problems, _ = _problems(settings)
    assert len(problems) == 1
    assert "2 targets are named 'T'" in problems[0]
    assert "first.pdb" in problems[0] and "second.pdb" in problems[0]


def test_two_targets_with_their_own_names_are_not_refused(tmp_path):
    first = _target(tmp_path, "first.pdb")
    second = _target(tmp_path, "second.pdb")
    settings = {"binder_lengths": [60],
                "targets": [{"name": "A", "target_path": str(first), "hotspots": "54"},
                            {"name": "B", "target_path": str(second), "hotspots": "56"}]}
    assert _problems(settings) == ([], [])


# ---------------------------------------------------------------- counts that fail elsewhere

def test_a_negative_recycle_count_is_refused_before_a_card_is_opened(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    problems, _ = _problems(_settings(path, hotspots="54") | {"design_recycles": -3})
    assert len(problems) == 1
    assert "design_recycles -3 is not a recycle count" in problems[0]
    assert "Write 0 for a single pass" in problems[0]


def test_no_recycling_at_all_is_a_choice_and_is_not_refused(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    assert _problems(_settings(path, hotspots="54") | {"design_recycles": 0}) == ([], [])


def test_asking_for_no_final_designs_is_refused_because_the_campaign_stops_at_once(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    problems, _ = _problems(_settings(path, hotspots="54") | {"number_of_final_designs": 0})
    assert len(problems) == 1 and "ends the campaign before it takes a single trajectory" in problems[0]


def test_no_final_designs_with_trajectory_only_is_not_refused(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    settings = _settings(path, hotspots="54") | {"number_of_final_designs": 0,
                                                 "trajectory_only": True,
                                                 "max_trajectories": 2}
    assert _problems(settings) == ([], [])


# ---------------------------------------------------------------- a JSON list where a string goes

def test_hotspots_written_as_a_json_list_are_refused_with_the_string_to_write(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    problems, _ = _problems(_settings(path, hotspots=["54", "56"]))
    assert len(problems) == 1
    assert '"hotspots": "54,56"' in problems[0] and "comma-separated string" in problems[0]


def test_chains_written_as_a_json_list_are_refused_before_bindcraft_splits_them(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    problems, _ = _problems(_settings(path, chains=["A", "B"], hotspots="54"))
    assert len(problems) == 1 and '"chains": "A,B"' in problems[0]


def test_hotspots_with_spaces_around_the_commas_are_not_refused(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    assert _problems(_settings(path, hotspots="54, 56 ,115")) == ([], [])


# ------------------------------------------------- what a filter enforces is what it prints

def _resolved(**settings):
    load_settings = pytest.importorskip("bindcraft.settings").load_settings
    return load_settings({"modality": "binder", "binder_lengths": [60], **settings})


def test_a_filter_written_without_a_direction_is_given_the_metrics_own_direction():
    """A researcher writes {"i_pAE": {"threshold": 0.35}} and means "at most 0.35".

    Nothing in tt_bio checks this, deliberately: `load_settings` fills `higher` from the metric,
    so the campaign enforces and prints the same bound. The two readers disagree about a MISSING
    `higher` -- `campaign_filter.declared_thresholds` defaults it True and
    `campaign_log.acceptance_filters` defaults it False -- so this test is what says the default
    never reaches them. If it fails, a filter is being enforced backwards from how it is printed.
    """
    _bindcraft()
    resolved = _resolved(filters={"i_pAE": {"threshold": 0.35}})
    assert resolved["filters"]["i_pAE"]["higher"] is False
    resolved = _resolved(filters={"i_pTM": {"threshold": 0.8}})
    assert resolved["filters"]["i_pTM"]["higher"] is True


def test_every_resolved_filter_prints_the_bound_it_enforces():
    _bindcraft()
    from bindcraft.campaign_filter import declared_thresholds
    from bindcraft.campaign_log import acceptance_filters

    resolved = _resolved(filters={"i_pAE": {"threshold": 0.35}, "i_pTM": {"threshold": 0.8},
                                  "Backbone_Clashes": {"threshold": 0}})
    enforced = {f"{t.metric} {t.comparison} {t.value:g}"
                for t in declared_thresholds(resolved["filters"])}
    for line in acceptance_filters(resolved):
        assert line in enforced, f"printed {line!r}, which is not a bound the campaign enforces"


# ---------------------------------------------------------------- the extension lies

def test_a_pdb_named_cif_is_refused_with_the_rename_to_make(tmp_path):
    path = _target(tmp_path, "really_a_pdb.cif")
    problems, _ = _problems(_settings(path, hotspots="54"))
    assert len(problems) == 1
    assert "named .cif but holds PDB records" in problems[0] and "Rename it to .pdb" in problems[0]


def test_an_mmcif_named_pdb_is_refused_the_same_way(tmp_path):
    path = tmp_path / "really_a_cif.pdb"
    path.write_text("data_structure\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\nATOM 1\n")
    problems, _ = _problems(_settings(path, hotspots="54"))
    assert len(problems) == 1 and "Rename it to .cif" in problems[0]


def test_a_pdb_named_pdb_and_an_mmcif_named_cif_are_not_refused(tmp_path):
    assert _problems(_settings(_target(tmp_path, "fine.pdb"), hotspots="54")) == ([], [])
    cif = tmp_path / "fine.cif"
    cif.write_text("data_structure\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\nATOM 1\n")
    problems, _ = _problems(_settings(cif, hotspots="54"))
    assert problems == []                # unreadable as a structure, which is upstream's to say


# ---------------------------------------------------------------- a residue numbered below 1

def test_a_hotspot_written_with_a_minus_is_refused_and_names_the_files_numbering(tmp_path):
    lines = [line for line in _bindcraft() if line.startswith("ATOM")]
    tagged = [line[:22] + f"{int(line[22:26]) - 21:>4}" + line[26:] for line in lines]
    path = tmp_path / "negative_numbering.pdb"
    path.write_text("\n".join(tagged) + "\nEND\n")
    problems, _ = _problems(_settings(path, hotspots="-2"))
    assert len(problems) == 1
    assert "numbers its residues from -3" in problems[0]
    assert "read as the range separator" in problems[0]


def test_a_positive_hotspot_on_the_same_file_still_resolves(tmp_path):
    lines = [line for line in _bindcraft() if line.startswith("ATOM")]
    tagged = [line[:22] + f"{int(line[22:26]) - 21:>4}" + line[26:] for line in lines]
    path = tmp_path / "negative_numbering.pdb"
    path.write_text("\n".join(tagged) + "\nEND\n")
    assert _problems(_settings(path, hotspots="33")) == ([], [])


# ------------------------------------------------- a hotspot on a residue the reader dropped

def test_a_hotspot_on_a_ligand_names_the_heteroatom_it_points_at(tmp_path):
    """A researcher reads 401 off a viewer, where the glycan is on screen with a number."""
    lines = [line for line in _bindcraft() if line.startswith("ATOM")]
    lines.append("HETATM 9001  C1  NAG A 401       1.000   2.000   3.000  1.00  0.00           C")
    path = tmp_path / "ligands.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    problems, _ = _problems(_settings(path, hotspots="54,401"))
    assert len(problems) == 1
    assert "residue 401 of chain A in ligands.pdb is NAG, a heteroatom" in problems[0]
    assert "names a protein residue" in problems[0]


def test_a_hotspot_on_no_residue_at_all_still_names_the_range(tmp_path):
    path = _target(tmp_path, "whole.pdb")
    problems, _ = _problems(_settings(path, hotspots="401"))
    assert len(problems) == 1 and "residues run 18 to 132" in problems[0]
