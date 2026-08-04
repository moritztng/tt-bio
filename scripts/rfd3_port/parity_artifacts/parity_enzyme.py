#!/usr/bin/env python3
"""Compare ported featurizer `f` vs a reference captured `f` for the F4
enzyme-design case (p17): the real RosettaCommons/foundry
``enzyme_design.md`` example verbatim -- ``M0255_1mg5.pdb`` (alcohol
dehydrogenase, the same PDB the ``intermediate_enzyme_design_tutorial.md``
walks through), spec = {ligand: "NAI,ACT", unindex: "A108,A139,A152,A156",
length: "180-200", select_fixed_atoms: {"A108": "ND2,CG", "A139": "OG,CB,CA",
"A152": "OH,CZ", "A156": "NZ,CE,CD", "ACT": "OXT", "NAI": ""}} -- TWO
different ligand instances (a NAI cofactor + an ACT product) plus FOUR
unindexed catalytic protein residues, each with a `select_fixed_atoms`-
subsetted (not full-residue) atom selection.

Reproduce the reference capture (CPU, no ckpt -- same method as
parity_dna.py/parity_ligand.py). Needs `allow_ligand_on_existing_chain: true`
since this PDB's two hetero groups land on the same raw non-polymer chain:
    uv venv --python 3.12 /tmp/fndry_venv
    uv pip install --python /tmp/fndry_venv/bin/python "rc-foundry[rfd3]"
    /tmp/fndry_venv/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/enzyme_m0255/M0255_1mg5.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/enzyme_m0255/spec.json \
        --out_dir /tmp/ref_enzyme_capture --seed 42

Known, documented gaps (NOT mismatches):
- `ref_pos` for both ligands: real reference-CONFORMER geometry from a
  STOCHASTIC RDKit ETKDG embed (see parity_ligand.py's docstring for the
  full grounding) -- this port reuses its own bundled CCD rdkit-mol library
  instead, a chemically valid but not bit-identical conformer.
"""
import json
import os, sys
import torch

DIR = os.path.dirname(__file__)
PDB = os.path.join(DIR, "enzyme_m0255", "M0255_1mg5.pdb")
SPEC_JSON = os.path.join(DIR, "enzyme_m0255", "spec.json")
REF_PT = "/tmp/ref_enzyme_capture/ref_f.pt"


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() == 0: return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0: return 1.0 if torch.allclose(a, b) else float("nan")
    return float((a * b).sum() / denom)


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(DIR, "..", "..", "..")))
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    if not os.path.exists(REF_PT):
        print(f"SKIPPED: no reference capture at {REF_PT} (see module docstring to reproduce)")
        return
    with open(SPEC_JSON) as fh:
        spec_dict = json.load(fh)
    spec_dict["input"] = PDB
    spec = InputSpecification.from_dict(spec_dict)
    pf = featurize(PDB, spec)
    rf = torch.load(REF_PT, weights_only=False)

    print("=" * 70)
    print(f"PORTED I={pf['restype'].shape[0]} L={pf['ref_pos'].shape[0]} | "
          f"REF I={rf['restype'].shape[0]} L={rf['ref_pos'].shape[0]}")
    print("=" * 70)

    tok_keys = ["restype", "ref_motif_token_type", "ref_plddt", "is_non_loopy",
                "is_motif_token_unindexed", "is_motif_token_with_fully_fixed_coord",
                "is_protein", "is_rna", "is_dna", "is_ligand", "is_polar",
                "terminus_type", "asym_id", "entity_id", "sym_id",
                "residue_index", "token_index", "token_bonds", "unindexing_pair_mask"]
    print("--- TOKEN-LEVEL value comparison ---")
    mismatches = []
    for k in tok_keys:
        if k not in pf or k not in rf:
            print(f"  {k:35s} MISSING (ported={'Y' if k in pf else 'N'} ref={'Y' if k in rf else 'N'})")
            mismatches.append(k); continue
        a, b = pf[k], rf[k]
        if a.shape != b.shape:
            print(f"  {k:35s} SHAPE {list(a.shape)} vs {list(b.shape)}  MISMATCH")
            mismatches.append(k); continue
        be = torch.equal(a, b)
        p = pcc(a, b) if a.dtype.is_floating_point or a.dtype in (torch.int64, torch.int32, torch.int8) else 1.0
        status = "OK" if be else "DIFF"
        if not be: mismatches.append(k)
        print(f"  {k:35s} {status:4s} bitexact={be} pcc={p:.4f}")
        if not be and a.numel() <= 200:
            print(f"      ported: {a.flatten().tolist()}")
            print(f"      ref   : {b.flatten().tolist()}")

    print()
    print("--- ATOM-LEVEL value comparison ---")
    KNOWN_GAP_KEYS = {"ref_pos"}  # real reference-conformer geometry, stochastic (see docstring)
    atom_keys = ["ref_atom_name_chars", "ref_pos", "ref_mask", "ref_element", "ref_charge",
                 "ref_space_uid", "ref_pos_is_ground_truth", "has_zero_occupancy",
                 "ref_is_motif_atom_with_fixed_coord", "ref_is_motif_atom_unindexed",
                 "ref_atomwise_rasa", "active_donor", "active_acceptor", "is_atom_level_hotspot",
                 "is_motif_atom_with_fixed_coord", "is_motif_atom_with_fixed_seq",
                 "is_motif_atom_unindexed", "motif_pos", "is_ca", "is_central",
                 "is_backbone", "is_sidechain", "is_virtual", "atom_to_token_map"]
    atom_mismatches = []
    same_L = pf["ref_pos"].shape[0] == rf["ref_pos"].shape[0]
    for k in atom_keys:
        if k not in pf or k not in rf:
            print(f"  {k:30s} MISSING (ported={'Y' if k in pf else 'N'} ref={'Y' if k in rf else 'N'})")
            atom_mismatches.append(k); continue
        a, b = pf[k], rf[k]
        if not same_L or a.shape != b.shape:
            print(f"  {k:30s} SHAPE {list(a.shape)} vs {list(b.shape)}  (L mismatch, skip value compare)")
            continue
        be = torch.equal(a, b)
        p = pcc(a, b) if a.dtype.is_floating_point else 1.0
        status = "OK" if be else ("KNOWN GAP" if k in KNOWN_GAP_KEYS else "DIFF")
        if not be and k not in KNOWN_GAP_KEYS:
            atom_mismatches.append(k)
        print(f"  {k:30s} {status:9s} bitexact={be} pcc={p:.4f}  {list(a.shape)} {a.dtype}")

    print()
    print(f"TOKEN-LEVEL MISMATCHES ({len(mismatches)}): {mismatches}")
    print(f"ATOM-LEVEL MISMATCHES ({len(atom_mismatches)}) excluding documented gaps {KNOWN_GAP_KEYS}: {atom_mismatches}  (same_L={same_L})")


if __name__ == "__main__":
    main()
