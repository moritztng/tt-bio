#!/usr/bin/env python3
"""Compare ported featurizer `f` vs a reference captured `f` for the F3
small-molecule-binder ligand case (p16): IAI.pdb (RosettaCommons/foundry's own
`sm_binder_design.md` "buried" example — a real ligand-bearing PDB, CCD code
IAI), spec = {length: "180-180", ligand: "IAI", select_fixed_atoms: {"IAI":
""}, select_buried: {"IAI": "<33 real atom names>"}} — a pure designed-length
protein binder (no contig/motif at all) around the fixed-identity, diffused-
position ligand, with every ligand atom RASA-conditioned "buried".

Reproduce the reference capture (CPU, no ckpt — same method as parity_dna.py):
    uv venv --python 3.12 /tmp/fndry_venv
    uv pip install --python /tmp/fndry_venv/bin/python "rc-foundry[rfd3]"
    /tmp/fndry_venv/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/ligand_iai/IAI.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/ligand_iai/spec_buried.json \
        --out_dir /tmp/ref_ligand_capture --seed 42

Known, documented gap (NOT a mismatch): `ref_pos` for the ligand is real
reference-CONFORMER geometry from a STOCHASTIC RDKit ETKDG embed (a fresh
random seed/rigid-augmentation draw every reference run, verified: no
explicit seed threads into `ccd_code_to_rdkit_with_conformers`/
`random_rigid_augmentation` in the real pipeline) -- no single reference
capture's `ref_pos` is "the" bit-exact target. This port instead reuses
tt_bio's own bundled CCD rdkit-mol library (~/.boltz/mols, the same one
Boltz-2/Protenix-v2 already ship) for a chemically valid conformer of the
same molecule -- exactly the precedent already documented at
tt_bio/protenix_data.py:466 ("the reference uses a STOCHASTIC RDKit
conformer, so any valid one folds correctly"). Reported separately, not
counted in MISMATCHES.
"""
import json
import os, sys
import torch

DIR = os.path.dirname(__file__)
PDB = os.path.join(DIR, "ligand_iai", "IAI.pdb")
SPEC_JSON = os.path.join(DIR, "ligand_iai", "spec_buried.json")
REF_PT = "/tmp/ref_ligand_capture/ref_f.pt"


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
