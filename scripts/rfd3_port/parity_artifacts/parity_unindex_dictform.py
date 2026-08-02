#!/usr/bin/env python3
"""Compare ported featurizer `f` vs real reference captures for the p22
dict-form `unindex` case (`unindex` as a ``Mapping``, not a plain contig
string) -- the gap left open since p14.

Real reference mechanism (`parsing.py::InputSelection.get_tokens` ->
`from_any_` -> `get_name_mask`): a dict-form `unindex`'s VALUE directly
subsets each named residue's real atoms, independent of (and composed as an
INTERSECTION with) any separate `select_fixed_atoms` restriction on the same
residue (`_build_init`'s `unindexed_tokens[k] = tok[is_motif_atom_with_fixed_
coord]` line runs on the ALREADY dict-value-subsetted token). A dict key may
be a single residue (`"A108"`) or a range (`"A108-115"`, expanded via the
reference's own `unravel_components`, same as a contig range).

Three real reference captures, all against `M0255_1mg5.pdb` (enzyme_m0255
fixture, no ligand/select_fixed_atoms unless noted) or `IAI_protein.pdb`:
  1. dict-only:  unindex={"A108": "ND2,CG"}, length=20            (M0255)
  2. composed:   unindex={"A108": "ND2,CG,OD1"} + select_fixed_atoms=
                 {"A108": "ND2,CG"} -> intersection {ND2,CG}, length=20 (M0255)
  3. range-key:  unindex={"A5-6": "CB"}, length=20                (IAI_protein)

Reproduce (CPU, no ckpt -- same method as parity_enzyme.py etc):
    uv venv --python 3.12 /tmp/fndry_venv
    uv pip install --python /tmp/fndry_venv/bin/python "rc-foundry[rfd3]"
    /tmp/fndry_venv/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/enzyme_m0255/M0255_1mg5.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/unindex_dictform/spec.json \
        --out_dir /tmp/ref_unindex_dictform_capture --seed 42
    (same pattern for spec_composed.json -> /tmp/ref_unindex_dictform_composed_capture,
     and unindex_rangekey/spec.json against IAI_protein.pdb -> /tmp/ref_unindex_rangekey_capture)

No documented gaps for any of the three fixtures (no ligand, no stochastic
ref_pos-dependent residues involved -- these fixtures don't diffuse ref_pos
for protein at all, see module docstring).
"""
import json
import os, sys
import torch

DIR = os.path.dirname(__file__)


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() == 0: return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0: return 1.0 if torch.allclose(a, b) else float("nan")
    return float((a * b).sum() / denom)


TOK_KEYS = ["restype", "ref_motif_token_type", "ref_plddt", "is_non_loopy",
            "is_motif_token_unindexed", "is_motif_token_with_fully_fixed_coord",
            "is_protein", "is_rna", "is_dna", "is_ligand", "is_polar",
            "terminus_type", "asym_id", "entity_id", "sym_id",
            "residue_index", "token_index", "token_bonds", "unindexing_pair_mask"]
ATOM_KEYS = ["ref_atom_name_chars", "ref_pos", "ref_mask", "ref_element", "ref_charge",
             "ref_space_uid", "ref_pos_is_ground_truth", "has_zero_occupancy",
             "ref_is_motif_atom_with_fixed_coord", "ref_is_motif_atom_unindexed",
             "ref_atomwise_rasa", "active_donor", "active_acceptor", "is_atom_level_hotspot",
             "is_motif_atom_with_fixed_coord", "is_motif_atom_with_fixed_seq",
             "is_motif_atom_unindexed", "motif_pos", "is_ca", "is_central",
             "is_backbone", "is_sidechain", "is_virtual", "atom_to_token_map"]


def compare(name, pdb, spec_json, ref_pt):
    print("=" * 70)
    print(f"CASE: {name}")
    print("=" * 70)
    if not os.path.exists(ref_pt):
        print(f"SKIPPED: no reference capture at {ref_pt}")
        return True
    sys.path.insert(0, os.path.abspath(os.path.join(DIR, "..", "..", "..")))
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    with open(spec_json) as fh:
        spec_dict = json.load(fh)
    spec_dict["input"] = pdb
    spec = InputSpecification.from_dict(spec_dict)
    spec.validate()
    pf = featurize(pdb, spec)
    rf = torch.load(ref_pt, weights_only=False)
    print(f"PORTED I={pf['restype'].shape[0]} L={pf['ref_pos'].shape[0]} | "
          f"REF I={rf['restype'].shape[0]} L={rf['ref_pos'].shape[0]}")

    mismatches = []
    for k in TOK_KEYS:
        if k not in pf or k not in rf:
            mismatches.append(k); continue
        a, b = pf[k], rf[k]
        if a.shape != b.shape or not torch.equal(a, b):
            mismatches.append(k)
    same_L = pf["ref_pos"].shape[0] == rf["ref_pos"].shape[0]
    atom_mismatches = []
    for k in ATOM_KEYS:
        if k not in pf or k not in rf:
            atom_mismatches.append(k); continue
        a, b = pf[k], rf[k]
        if not same_L or a.shape != b.shape or not torch.equal(a, b):
            atom_mismatches.append(k)
    print(f"TOKEN-LEVEL MISMATCHES ({len(mismatches)}): {mismatches}")
    print(f"ATOM-LEVEL MISMATCHES ({len(atom_mismatches)}): {atom_mismatches}  (same_L={same_L})")
    ok = same_L and not mismatches and not atom_mismatches
    print("RESULT:", "PASS" if ok else "FAIL")
    print()
    return ok


def main():
    m0255 = os.path.join(DIR, "enzyme_m0255", "M0255_1mg5.pdb")
    iai = os.path.join(DIR, "iai_protein", "IAI_protein.pdb")
    results = [
        compare("dict-only (unindex={'A108': 'ND2,CG'})",
                m0255, os.path.join(DIR, "unindex_dictform", "spec.json"),
                "/tmp/ref_unindex_dictform_capture/ref_f.pt"),
        compare("composed (unindex dict + select_fixed_atoms, intersection)",
                m0255, os.path.join(DIR, "unindex_dictform", "spec_composed.json"),
                "/tmp/ref_unindex_dictform_composed_capture/ref_f.pt"),
        compare("range-key dict (unindex={'A5-6': 'CB'})",
                iai, os.path.join(DIR, "unindex_rangekey", "spec.json"),
                "/tmp/ref_unindex_rangekey_capture/ref_f.pt"),
    ]
    print("ALL PASS" if all(results) else "SOME FAILED")


if __name__ == "__main__":
    main()
