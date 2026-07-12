"""Bit-exact check: tt_bio.opendde_data.build_structural_token_features vs the real upstream
opendde.data.tokenizer.AtomArrayTokenizer, on a synthetic single-chain protein built from
tt-bio's OWN atom-name table (tt_bio.data.const.ref_atoms) -- so both sides tokenize the
identical atom layout. No PDB/mmCIF, no checkpoints; CPU-only.

Set OPENDDE_SRC to a checkout pinned at a0d5134 (the reference-build precedent used by
scripts/opendde_structtoken_ref.py). Run:
  OPENDDE_SRC=/tmp/opendde-src PYTHONPATH=<worktree> \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/opendde_structtoken_featurizer_parity.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("OPENDDE_SRC", "/tmp/opendde-src"))

import biotite.structure as struc
from opendde.data.tokenizer import AtomArrayTokenizer

from tt_bio.data import const
from tt_bio.opendde_data import build_structural_token_features, _residue_atom_names, _LETTER_TO_RES
from tt_bio.protenix_data import RESTYPE_ORDER

# A short sequence exercising: a residue with a sidechain (A, W, K, S), glycine (G, the
# single-token-fallback branch), and both again to check adjacency/twin indices repeat cleanly.
SEQ = "AGWKSG"


def build_atom_array(seq):
    names_all, res_ids, res_names = [], [], []
    for i, letter in enumerate(seq):
        aa = RESTYPE_ORDER.index(letter)
        res = _LETTER_TO_RES[RESTYPE_ORDER[aa]]
        names = _residue_atom_names(res, is_c_terminal=(i == len(seq) - 1))
        names_all += names
        res_ids += [i + 1] * len(names)
        res_names += [res] * len(names)

    n = len(names_all)
    arr = struc.AtomArray(n)
    arr.coord = np.zeros((n, 3), dtype=np.float32)
    arr.chain_id = np.array(["A"] * n)
    arr.res_id = np.array(res_ids, dtype=np.int32)
    arr.res_name = np.array(res_names)
    arr.atom_name = np.array(names_all)
    arr.element = np.array([nm[0] for nm in names_all])
    arr.hetero = np.zeros(n, dtype=bool)
    arr.set_annotation("mol_type", np.array(["protein"] * n))
    centre_mask = np.array([1 if nm == "CA" else 0 for nm in names_all], dtype=np.int64)
    arr.set_annotation("centre_atom_mask", centre_mask)
    return arr


def main():
    atom_array = build_atom_array(SEQ)
    tok = AtomArrayTokenizer(atom_array)
    residue_tokens = tok.get_token_array()
    ref_structural = tok.get_structural_token_array(residue_tokens)

    ref_role = np.array(ref_structural.get_annotation("subtoken_role_id"))
    ref_parent = np.array(ref_structural.get_annotation("parent_residue_idx"))
    ref_twin = np.array(ref_structural.get_annotation("twin_token_idx"))
    print(f"upstream: {len(ref_structural)} structural tokens for {len(SEQ)}-residue seq {SEQ}")

    from tt_bio.protenix_data import RESTYPE_DIM
    aatype = torch.tensor([RESTYPE_ORDER.index(c) for c in SEQ], dtype=torch.long)
    feats = {
        "restype": torch.nn.functional.one_hot(aatype, RESTYPE_DIM).float(),
        "asym_id": torch.zeros(len(SEQ), dtype=torch.long),
        "residue_index": torch.arange(1, len(SEQ) + 1, dtype=torch.long),
    }
    mine = build_structural_token_features(feats)

    ok = True
    for name, ref, got in [
        ("subtoken_role_id", ref_role, mine["subtoken_role_id"].numpy()),
        ("parent_residue_idx", ref_parent, mine["parent_residue_idx"].numpy()),
        ("twin_token_idx", ref_twin, mine["twin_token_idx"].numpy()),
    ]:
        match = np.array_equal(ref, got)
        ok &= match
        print(f"  {name}: {'MATCH' if match else 'MISMATCH'}  ref={ref.tolist()}  mine={got.tolist()}")

    # atom<->structural-token maps: check every atom's assigned structural token has the
    # atom's name in the right role-group (backbone name -> a protein_bb-role token, else sc).
    names_all = list(atom_array.atom_name)
    a2s = mine["atom_to_structural_token_idx"].numpy()
    role_of_struct_tok = mine["subtoken_role_id"].numpy()
    from tt_bio.opendde_data import PROTEIN_BACKBONE_ATOMS
    bad = 0
    for atom_idx, nm in enumerate(names_all):
        want_bb = nm in PROTEIN_BACKBONE_ATOMS
        got_role = role_of_struct_tok[a2s[atom_idx]]
        got_bb = got_role == 1  # protein_bb
        if want_bb != got_bb:
            bad += 1
    print(f"  atom_to_structural_token_idx role consistency: {'MATCH' if bad == 0 else f'{bad} MISMATCHES'}")
    ok &= (bad == 0)

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
