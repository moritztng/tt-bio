#!/usr/bin/env python3
"""Compare ported featurizer `f` vs reference captured `f` for the unindexed-
motif case (F6 unindex field, p14): IAI_protein.pdb, contig 'A1-10,20,A31-40',
unindex 'A100-101,A120' (a tied 2-residue island A100-101 + a separate
singleton island A120).

Token-level keys are compared value-by-value (bit-exact + PCC); the
unindexed tokens land at I=40,41,42 (appended at the end, per the reference).
Atom-level keys are compared the same way as parity_iai.py (structural: ported
uses fixed-14 for designed tokens only, same as the reference for this case).
"""
import os, sys
import torch

PDB = "/tmp/IAI_protein.pdb"
CONTIG = "A1-10,20,A31-40"
UNINDEX = "A100-101,A120"
REF_PT = "/tmp/ref_unindex_capture/ref_f.pt"


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() == 0: return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0: return 1.0 if torch.allclose(a, b) else float("nan")
    return float((a * b).sum() / denom)


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG, "unindex": UNINDEX})
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
            mismatches.append(k)
            continue
        a, b = pf[k], rf[k]
        if a.shape != b.shape:
            print(f"  {k:35s} SHAPE {list(a.shape)} vs {list(b.shape)}  MISMATCH")
            mismatches.append(k); continue
        be = torch.equal(a, b)
        p = pcc(a, b) if a.dtype.is_floating_point or a.dtype in (torch.int64, torch.int32, torch.int8) else 1.0
        status = "OK" if be else "DIFF"
        if not be: mismatches.append(k)
        print(f"  {k:35s} {status:4s} bitexact={be} pcc={p:.4f}")
        if not be and a.numel() <= 120:
            print(f"      ported: {a.flatten().tolist()}")
            print(f"      ref   : {b.flatten().tolist()}")

    print()
    print("--- ATOM-LEVEL value comparison (unindexed tokens' atoms only, L may match now) ---")
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
        print(f"  {k:30s} {status:4s} bitexact={be} pcc={p:.4f}  {list(a.shape)} {a.dtype}")

    print()
    print(f"TOKEN-LEVEL MISMATCHES ({len(mismatches)}): {mismatches}")
    print(f"ATOM-LEVEL MISMATCHES ({len(atom_mismatches)}): {atom_mismatches}  (same_L={same_L})")


if __name__ == "__main__":
    main()
