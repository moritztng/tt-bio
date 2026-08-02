"""CPU regression: OpenDDE covalent ligand featurization (no device, no weights).

Verifies the two blockers the ligand port fixed:
  * build_complex_features marks token_bonds for a protein Cys SG <-> ligand atom
    bond (the ligand endpoint resolves to its named-atom token via _resolve_bond_token).
  * opendde_data.build_structural_token_features represents the ligand: each ligand
    atom-token expands to one "atom"-role structural token (role id 0), and
    atom_to_structural_token_idx aligns 1:1 with feats' atom order (the same per-chain
    alignment that bit the multi-chain OXT bug).

Also asserts a protein-only complex produces no "atom"-role structural tokens (protein
handling unchanged: the ligand branch is never taken; mol_type is unused for protein).
"""
import os

import pytest
import torch

from tt_bio.opendde_data import STRUCTURAL_TOKEN_ROLES, build_structural_token_features
from tt_bio.protenix_data import build_complex_features

_MOL_DIR = os.path.expanduser("~/.boltz/mols")
pytestmark = pytest.mark.skipif(
    not os.path.exists(_MOL_DIR), reason="needs bundled CCD mol library (~/.boltz/mols)")

PROT = "GCGSQWDRSGR"          # Cys at residue 2 (1-indexed)
LIG = "C=CC(=O)N"             # acrylamide warhead; C1 = terminal =CH2 (Michael acceptor)


def test_covalent_ligand_bond_marked():
    chains = [(PROT, None, "protein"), (LIG, None, "ligand")]
    bonds = [(("A", 2, "SG"), ("B", 1, "C1"))]
    feats = build_complex_features(chains, mol_dir=_MOL_DIR, chain_ids=["A", "B"], bonds=bonds)
    tb = feats["token_bonds"]
    cys_tok = 1
    lig_c1_tok = feats["asym_id"].tolist().index(1)   # first ligand token
    assert feats["mol_type"][lig_c1_tok].item() == 3
    assert tb[cys_tok, lig_c1_tok].item() == 1.0 and tb[lig_c1_tok, cys_tok].item() == 1.0
    cross = ((feats["asym_id"][:, None] != feats["asym_id"][None, :]) & (tb > 0))
    assert cross.sum().item() == 2, f"expected one cross-chain bond (2 symmetric), got {cross.sum().item()}"


def test_ligand_structural_tokens():
    feats = build_complex_features([(PROT, None, "protein"), (LIG, None, "ligand")],
                                   mol_dir=_MOL_DIR, chain_ids=["A", "B"])
    ifd = build_structural_token_features(feats)
    role, parent = ifd["subtoken_role_id"], ifd["parent_residue_idx"]
    a2s, a2sa = ifd["atom_to_structural_token_idx"], ifd["atom_to_structural_tokatom_idx"]
    n_atom = feats["atom_to_token_idx"].shape[0]
    lig_toks = [i for i, m in enumerate(feats["mol_type"].tolist()) if m == 3]
    assert len(lig_toks) == 5
    lig_st = [i for i, p in enumerate(parent.tolist()) if p in lig_toks]
    assert len(lig_st) == 5
    assert all(role[i].item() == STRUCTURAL_TOKEN_ROLES["atom"] for i in lig_st)
    assert all(int(t) == -1 for t in ifd["twin_token_idx"][lig_st].tolist())
    assert a2s.shape[0] == n_atom and (a2s >= 0).all() and (a2s < role.shape[0]).all()
    lig_atom_idx = list(range(n_atom - 5, n_atom))
    assert [int(a2s[a]) for a in lig_atom_idx] == lig_st
    assert (a2sa[lig_atom_idx] == 0).all()
    prot_st = [i for i, p in enumerate(parent.tolist()) if p not in lig_toks]
    assert all(role[i].item() in (STRUCTURAL_TOKEN_ROLES["protein_bb"],
                                  STRUCTURAL_TOKEN_ROLES["protein_sc"]) for i in prot_st)


def test_protein_only_no_atom_role_tokens():
    feats = build_complex_features([("ARNDCEKHIL", None, "protein")], chain_ids=["A"])
    ifd = build_structural_token_features(feats)
    assert ifd["parent_residue_idx"].shape[0] == 20   # 10 non-Gly residues -> 20 bb/sc tokens
    assert (ifd["subtoken_role_id"] != STRUCTURAL_TOKEN_ROLES["atom"]).all()
    twin = ifd["twin_token_idx"]
    assert (twin >= 0).all() and (twin < 20).all()
