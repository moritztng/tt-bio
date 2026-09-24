"""The affinity ligand's SMILES standardization (tt_bio/data/parse.py standardize).

Four of the 68 DAVIS kinase compounds are salts. Under RDKit 2026.03 every one of them failed
here with a pre-condition violation, so a Boltz-2 affinity screen silently lost them while
Nesso-1 scored the same SMILES.
"""
import pytest

from tt_bio.data.parse import standardize

SALTS = {
    "Cc1ccc2nc(NCCN)c3ncc(C)n3c2c1.Cl": "Cc1ccc2nc(NCCN)c3ncc(C)n3c2c1",
    "N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1.O=P(O)(O)O":
        "N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1",
}


@pytest.mark.parametrize("salt,parent", SALTS.items())
def test_a_salt_standardizes_to_its_parent(salt, parent):
    assert standardize(salt) == parent


def test_a_single_fragment_is_its_own_parent():
    parent = "N#CCC(C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1"
    assert standardize(parent) == standardize(parent + ".O=P(O)(O)O")


def test_an_unparseable_smiles_still_fails():
    with pytest.raises(Exception):
        standardize("C1CC(N")
