"""Structural-token featurization parity: tt_bio.opendde_data vs upstream OpenDDE's own featurizer.

Upstream builds its features from a JSON job (opendde.data.inference.json_to_feature ->
Featurizer.get_all_input_features, which includes get_structural_token_features). tt-bio builds
them from protenix_data.build_complex_features. Both run here on the same sequences, and every
structural-token index feature must be equal. Reference coordinates are compared through each
residue's intra-residue distance matrix, since upstream randomly rotates ref_pos.

Needs a venv with `pip install opendde==1.0.3` (torch CPU is enough) and its CCD assets under
$OPENDDE_ROOT_DIR/common (components.cif + components.cif.rdkit_mol.pkl). Run from the repo root:
  PYTHONPATH=. <venv>/bin/python scripts/opendde_structtoken_featurizer_parity.py
"""
import os
import sys

import numpy as np
import torch

from opendde.data.inference.json_to_feature import SampleDictToFeatures

from tt_bio.opendde_data import build_structural_token_features
from tt_bio.protenix_data import build_complex_features

_JSON_TYPE = {"protein": "proteinChain", "rna": "rnaSequence", "dna": "dnaSequence"}

# Protein with every residue class (glycine, X, all 20), RNA and DNA with every base and the
# unknown nucleotide, a double-stranded DNA, and a two-copy entity.
CASES = {
    "protein+rna": [("protein", "MKTAYIAKQRQISFVKSHFSRQGX"), ("rna", "GGACUAGCNAUCC")],
    "protein+dna": [("protein", "ACDEFGHIKLMNPQRSTVWY"), ("dna", "ATGCNGCAT")],
    "protein+dsDNA": [("protein", "GSHMLEDPKRVG"), ("dna", "CGCGAATTCGCG"), ("dna", "CGCGAATTCGCG")],
    "rna only": [("rna", "AGCUAGCU")],
    "dna only": [("dna", "GATTACA")],
    "protein only": [("protein", "MKTAYIAKQRGWG"), ("protein", "MKTAYIAKQRGWG")],
}

INDEX_KEYS = ["structural_token_index", "parent_residue_idx", "subtoken_role_id",
              "twin_token_idx", "prev_parent_residue_idx", "next_parent_residue_idx",
              "atom_to_structural_token_idx", "atom_to_structural_tokatom_idx"]


def _names(chars):
    codes = chars.reshape(-1, 4, 64).argmax(-1) + 32
    return ["".join(map(chr, r)).rstrip() for r in codes.tolist()]


def upstream(chains):
    job = {"name": "parity", "sequences": [{_JSON_TYPE[mt]: {"sequence": s, "count": 1}}
                                           for mt, s in chains]}
    feats, _atoms, _tokens = SampleDictToFeatures(job).get_feature_dict()
    return feats


def ours(chains):
    return build_complex_features([(s, None, mt) for mt, s in chains],
                                  mol_dir=os.path.expanduser("~/.boltz/mols"))


def _geometry_delta(a, b):
    """Largest change in any intra-residue atom-atom distance between two ref_pos sets."""
    worst = 0.0
    for r in torch.unique(b["ref_space_uid"]).tolist():
        m = b["ref_space_uid"] == r
        if int(m.sum()) < 2 or not torch.equal(m, a["ref_space_uid"] == r):
            continue
        d = lambda x: torch.cdist(x.double(), x.double())
        worst = max(worst, float((d(a["ref_pos"][m]) - d(b["ref_pos"][m])).abs().max()))
    return worst


def compare(name, chains):
    up, f = upstream(chains), ours(chains)
    mine = build_structural_token_features(f)
    fails = []
    names_up, names_me = _names(up["ref_atom_name_chars"]), _names(f["ref_atom_name_chars"])
    if names_up != names_me:
        fails.append(f"atom names differ ({len(names_up)} upstream vs {len(names_me)} ours)")
    for k in INDEX_KEYS:
        a, b = up[k].long(), mine[k].long()
        if a.shape != b.shape or not torch.equal(a, b):
            fails.append(f"{k}: upstream {a.tolist()[:24]} ours {b.tolist()[:24]}")
    # Reference conformer geometry, per residue, rotation-invariant. Upstream draws a fresh
    # RDKit conformer per call, so its own second draw is the noise floor.
    worst, floor = _geometry_delta(up, f), _geometry_delta(up, upstream(chains))
    a2t = f["atom_to_token_idx"]
    roles = mine["subtoken_role_id"].bincount(minlength=7).tolist()
    status = "PASS" if not fails else "FAIL"
    print(f"{status} {name:14s} residue tokens {a2t.max().item() + 1:3d}  structural tokens "
          f"{len(mine['parent_residue_idx']):3d}  atoms {len(names_me):4d}  roles "
          f"atom/p_bb/p_sc/d_bb/d_base/r_bb/r_base={roles}  max intra-residue "
          f"ref_pos distance delta {worst:.2f} A (upstream vs itself {floor:.2f} A)")
    for x in fails:
        print("    ", x)
    return not fails


def main():
    ok = all([compare(n, c) for n, c in CASES.items()])
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
