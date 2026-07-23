#!/usr/bin/env python3
"""Compare ported featurizer `f` vs a reference captured `f` for the p23
INDEXED-NA `select_fixed_atoms` case: a dict-form `select_fixed_atoms` naming
residues that are part of a `contig` INDEXED DNA range (not protein) -- the
gap left open since p21 (`_indexed_fixed_atom_names` was wired protein-only).

Real reference mechanism (same as `parity_indexed_atomsubset.py`, verified
generic across residue kind): `input_parsing.py::_assign_types_to_input.
apply_selections` computes the per-atom `is_motif_atom_with_fixed_coord` mask
for EVERY residue identically, protein or NA.

Fixture: real `2r5z.pdb` (RosettaCommons/foundry's own bundled
`models/rfd3/docs/input_pdbs/2r5z.pdb`, a homeodomain/DNA complex) + the
REAL documented `dsDNA_complex` example's own contig/select_fixed_atoms
(`models/rfd3/docs/examples/na_binder_design.json`, `unindex` field dropped
to isolate the indexed-NA-select_fixed_atoms mechanism cleanly, matching the
existing `parity_indexed_atomsubset.py` convention of isolating one new
mechanism per fixture): chains C/D are DNA; C9-14/D28-33 get "ALL" (every
real atom fixed), C5-8+C15-18/D24-27+D34-37 get "" (present, but NOT
fixed-coord -- diffused position, known identity).

Reproduce the reference capture (CPU, no ckpt -- same method as
parity_dna.py/parity_indexed_atomsubset.py):
    /tmp/rfd3_ref_env/bin/python scripts/rfd3_port/capture_ref_f_spec.py \
        --pdb scripts/rfd3_port/parity_artifacts/dsdna_complex_na_sfa/2r5z.pdb \
        --spec_json scripts/rfd3_port/parity_artifacts/dsdna_complex_na_sfa/spec.json \
        --out_dir /tmp/ref_dsdna_complex_na_sfa_capture --seed 42
"""
import json
import os, sys
import torch

DIR = os.path.dirname(__file__)
SPEC_JSON = os.path.join(DIR, "dsdna_complex_na_sfa", "spec.json")
PDB = os.path.join(DIR, "dsdna_complex_na_sfa", "2r5z.pdb")
REF_PT = "/tmp/ref_dsdna_complex_na_sfa_capture/ref_f.pt"


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
        status = "OK" if be else "DIFF"
        if not be:
            atom_mismatches.append(k)
        print(f"  {k:30s} {status:9s} bitexact={be} pcc={p:.4f}  {list(a.shape)} {a.dtype}")

    print()
    print(f"TOKEN-LEVEL MISMATCHES ({len(mismatches)}): {mismatches}")
    print(f"ATOM-LEVEL MISMATCHES ({len(atom_mismatches)}): {atom_mismatches}  (same_L={same_L})")


if __name__ == "__main__":
    main()
