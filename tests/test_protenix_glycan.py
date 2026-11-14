"""Multi-CCD ligand chains (glycans) and covalent leaving-atom removal in the Protenix
featurizer, following upstream Protenix / Mol-1 ``build_ligand`` + ``remove_leaving_atoms``."""
import pytest
import torch

pytest.importorskip("rdkit")
from tt_bio import protenix_data as pd  # noqa: E402

try:
    MOLS = __import__("tt_bio.data.mol", fromlist=["load_molecules"]).load_molecules(
        pd._default_mol_dir(), ["NAG", "ATP"])
except FileNotFoundError:
    pytest.skip("no CCD mols directory", allow_module_level=True)


def _heavy(code):
    return [a.GetProp("name") for a in MOLS[code].GetAtoms() if a.GetAtomicNum() > 1]


def test_ccd_list_accepts_both_spellings():
    assert pd._ccd_list("CCD_NAG,NAG") == pd._ccd_list("CCD_NAG_NAG") == ["NAG", "NAG"]


def test_nag_leaving_group_is_o1_on_c1():
    assert pd.ccd_leaving_groups(MOLS["NAG"]) == {"C1": [["O1"]]}


def test_glycan_chain_is_one_residue_per_component():
    nag = _heavy("NAG")
    chains = [("MKNGTA", None, "protein"), ("CCD_NAG,NAG", None, "ligand")]
    bonds = [(("A", 3, "ND2"), ("B", 1, "C1")), (("B", 1, "O4"), ("B", 2, "C1"))]
    f = pd.build_complex_features(chains, chain_ids=["A", "B"], bonds=bonds)
    lig = f["asym_id"] == 1
    # both C1 are bonded, so each NAG loses its O1
    assert int(lig.sum()) == 2 * (len(nag) - 1)
    assert f["residue_index"][lig].tolist() == [1] * (len(nag) - 1) + [2] * (len(nag) - 1)
    assert (f["restype"][lig].argmax(-1) == 20).all()
    lig_atoms = lig[f["atom_to_token_idx"]]
    uids = f["ref_space_uid"][lig_atoms]
    assert len(set(uids.tolist())) == 2                 # one reference frame per residue
    n = len(nag) - 1
    i_o4 = n * 0 + [x for x in nag if x != "O1"].index("O4")
    i_c1 = n + [x for x in nag if x != "O1"].index("C1")
    t0 = int(torch.nonzero(lig)[0])
    assert f["token_bonds"][t0 + i_o4, t0 + i_c1] == 1


def test_unbonded_ligand_keeps_leaving_atoms():
    f = pd.build_complex_features([("CCD_NAG", None, "ligand")])
    assert f["asym_id"].shape[0] == len(_heavy("NAG"))


def test_bond_to_missing_glycan_residue_is_an_error():
    chains = [("MKNGTA", None, "protein"), ("CCD_NAG,NAG", None, "ligand")]
    with pytest.raises(ValueError, match="residue 3"):
        pd.build_complex_features(chains, chain_ids=["A", "B"],
                                  bonds=[(("A", 3, "ND2"), ("B", 3, "C1"))])
