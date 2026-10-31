"""Upstream Protenix / Mol-1 JSON -> tt-bio bio-input document (tt_bio.protenix_json). CPU only."""
import json
from pathlib import Path

import pytest

from tt_bio.protenix_json import int_to_letters, is_job_list, job_to_doc, read_jobs

CASES = Path(__file__).parent / "data" / "mol1_io" / "synthetic.json"


def _jobs():
    return {j.name: j for j in read_jobs(CASES)}


def test_int_to_letters_matches_upstream():
    assert [int_to_letters(n) for n in (1, 26, 27, 28, 52, 53)] == ["A", "Z", "AA", "AB", "AZ", "BA"]


def test_is_job_list(tmp_path):
    assert is_job_list(CASES)
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"sequences": []}))
    assert not is_job_list(p)


def test_chain_ids_skip_explicit_ids_and_cover_ions():
    j = _jobs()["ions_ids"]
    ids = [next(iter(e.values()))["id"] for e in j.doc["sequences"]]
    # explicit H/L/Z are claimed first, so the auto ids for the two MG ions start at A
    assert ids == [["H", "L"], ["A", "B"], ["Z"]]
    assert j.doc["sequences"][1] == {"ligand": {"id": ["A", "B"], "ccd": ["MG"]}}


def test_modifications_and_kinds():
    seqs = _jobs()["doc_ptm_dna_rna"].doc["sequences"]
    assert seqs[0]["protein"]["modifications"] == [{"position": 1, "ccd": "HY3"}, {"position": 5, "ccd": "P1L"}]
    assert seqs[1]["dna"]["modifications"] == [{"position": 1, "ccd": "6OG"}, {"position": 2, "ccd": "6MA"}]
    assert [list(s)[0] for s in seqs] == ["protein", "dna", "dna", "rna"]
    assert [s[list(s)[0]]["id"] for s in seqs] == [["A"], ["B"], ["C"], ["D"]]


def test_multi_ccd_ligand_and_bonds():
    j = _jobs()["glycan_bonds"]
    assert j.doc["sequences"][1] == {"ligand": {"id": ["B"], "ccd": ["NAG", "NAG"]}}
    assert j.doc["constraints"] == [{"bond": {"atom1": ["A", 5, "ND2"], "atom2": ["B", 1, "C1"]}},
                                    {"bond": {"atom1": ["B", 1, "O4"], "atom2": ["B", 2, "C1"]}}]


def test_bond_between_corresponding_copies():
    job = {"name": "x", "sequences": [{"proteinChain": {"sequence": "MKC", "count": 2}},
                                      {"ligand": {"ligand": "CCD_ATP", "count": 2}}],
           "covalent_bonds": [{"left_entity": 1, "left_position": 3, "left_atom": "SG",
                               "right_entity": 2, "right_position": 1, "right_atom": "PG"}]}
    assert [c["bond"] for c in job_to_doc(job).doc["constraints"]] == [
        {"atom1": ["A", 3, "SG"], "atom2": ["C", 1, "PG"]}, {"atom1": ["B", 3, "SG"], "atom2": ["D", 1, "PG"]}]


def test_smiles_and_seeds():
    j = job_to_doc({"name": "s", "modelSeeds": [7, 8], "sequences": [{"ligand": {"ligand": "CCO", "count": 1}}]})
    assert j.doc["sequences"] == [{"ligand": {"id": ["A"], "smiles": "CCO"}}] and j.seeds == [7, 8]


def test_msa_paths_read_per_copy(tmp_path):
    (tmp_path / "p.a3m").write_text(">q\nMKC\n")
    (tmp_path / "u.a3m").write_text(">q\nMKC\n>h\nMKA\n")
    j = job_to_doc({"name": "m", "sequences": [{"proteinChain": {
        "sequence": "MKC", "count": 2, "pairedMsaPath": str(tmp_path / "p.a3m"),
        "unpairedMsaPath": str(tmp_path / "u.a3m"), "templatesPath": "/x/hits.hhr"}}]})
    assert set(j.paired_msa) == set(j.unpaired_msa) == {"A", "B"}
    assert j.unpaired_msa["B"].count(">") == 2 and j.templates == {"A": "/x/hits.hhr", "B": "/x/hits.hhr"}


@pytest.mark.parametrize("bad", [
    {"name": "x", "sequences": [{"proteinChain": {"sequence": "MK", "count": 2, "id": ["A"]}}]},
    {"name": "x", "sequences": [{"proteinChain": {"sequence": "MK", "count": 1, "id": ["A"]}},
                                {"ligand": {"ligand": "CCO", "count": 1, "id": ["A"]}}]},
    {"name": "x", "sequences": [{"glycan": {"sequence": "x"}}]},
])
def test_upstream_validation_errors(bad):
    with pytest.raises(ValueError):
        job_to_doc(bad)
