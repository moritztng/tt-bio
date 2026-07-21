"""CPU regression: covalent `bond` constraints mark token_bonds at the endpoint tokens.

Covers both supported OpenDDE covalent-bond cases:
  * protein-protein (a disulfide-style crosslink between two Cys SG), and
  * protein-ligand (a covalent inhibitor: protein Cys SG <-> a named ligand atom),
    which also exercises the ligand structural-token featurizer
    (opendde_data.build_structural_token_features -> one "atom"-role structural
    token per ligand atom, atom_to_structural_token_idx aligned to feats' atom order).

Run from the repo root:  python3 scripts/covalent_bond_check.py
No device, no weights. Needs the bundled CCD mol library (~/.boltz/mols) for the
SMILES ligand embedding; skips the ligand leg with a notice if it is absent.
"""
import os
import sys

# Run from anywhere: ensure the repo root (parent of scripts/) is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from tt_bio.opendde_data import STRUCTURAL_TOKEN_ROLES, build_structural_token_features
from tt_bio.protenix_data import build_complex_features

MOL_DIR = os.path.expanduser("~/.boltz/mols")


def _protein_protein():
    chains = [("AC", None, "protein"), ("AC", None, "protein")]
    chain_ids = ["A", "B"]
    bonds = [(("A", 2, "SG"), ("B", 2, "SG"))]
    feats = build_complex_features(chains, chain_ids=chain_ids, bonds=bonds)
    tb, asym = feats["token_bonds"], feats["asym_id"]
    a_tok = [i for i in range(len(asym)) if asym[i] == 0 and feats["residue_index"][i] == 2]
    b_tok = [i for i in range(len(asym)) if asym[i] == 1 and feats["residue_index"][i] == 2]
    ok = all(tb[i, j].item() == 1.0 and tb[j, i].item() == 1.0 for i in a_tok for j in b_tok)
    nob = build_complex_features(chains, chain_ids=chain_ids, bonds=None)["token_bonds"].sum().item()
    print(f"[protein-protein] A-res2={a_tok} B-res2={b_tok} "
          f"with-bond sum={tb.sum().item():.0f} no-bond sum={nob:.0f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def _protein_ligand():
    if not os.path.exists(MOL_DIR):
        print("[protein-ligand] SKIP (no ~/.boltz/mols)")
        return True
    chains = [("GCGSQWDRSGR", None, "protein"), ("C=CC(=O)N", None, "ligand")]
    bonds = [(("A", 2, "SG"), ("B", 1, "C1"))]   # Cys SG <-> acrylamide C1 (Michael acceptor)
    feats = build_complex_features(chains, mol_dir=MOL_DIR, chain_ids=["A", "B"], bonds=bonds)
    tb, asym = feats["token_bonds"], feats["asym_id"]
    cys_tok, lig_c1_tok = 1, asym.tolist().index(1)
    bond_ok = (tb[cys_tok, lig_c1_tok].item() == 1.0 and tb[lig_c1_tok, cys_tok].item() == 1.0)
    cross = ((asym[:, None] != asym[None, :]) & (tb > 0)).sum().item()
    ifd = build_structural_token_features(feats)
    role, parent = ifd["subtoken_role_id"], ifd["parent_residue_idx"]
    a2s, a2sa = ifd["atom_to_structural_token_idx"], ifd["atom_to_structural_tokatom_idx"]
    lig_toks = [i for i, m in enumerate(feats["mol_type"].tolist()) if m == 3]
    lig_st = [i for i, p in enumerate(parent.tolist()) if p in lig_toks]
    role_ok = (len(lig_st) == 5 and all(role[i].item() == STRUCTURAL_TOKEN_ROLES["atom"] for i in lig_st))
    align_ok = ([int(a2s[a]) for a in range(a2s.shape[0] - 5, a2s.shape[0])] == lig_st
                and (a2sa[range(a2s.shape[0] - 5, a2s.shape[0])] == 0).all())
    ok = bond_ok and cross == 2 and role_ok and align_ok
    print(f"[protein-ligand] cys={cys_tok} ligC1={lig_c1_tok} cross_bonds={cross} "
          f"lig_struct_tokens={lig_st} role_atom={role_ok} atom_aligned={align_ok} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = _protein_protein() and _protein_ligand()
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)
