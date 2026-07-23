#!/usr/bin/env python3
"""Compare ported featurizer `f` vs reference captured `f` for the SAME input
(IAI_protein.pdb, contig 'A1-10,20,A31-40').

Token-level keys (I=40 for both) are compared value-by-value (bit-exact + PCC).
Atom-level keys differ structurally (ported L=560 fixed-14 vs ref L=419 variable)
and are reported as structural mismatches with the per-token atom-count breakdown.
"""
import os, sys, json
import torch

PDB = "/tmp/IAI_protein.pdb"
CONTIG = "A1-10,20,A31-40"
REF_PT = "/tmp/ref_iai_capture/ref_f.pt"

def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() == 0: return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0: return 1.0 if torch.allclose(a, b) else float("nan")
    return float((a * b).sum() / denom)

def main():
    sys.path.insert(0, "/home/moritz/.coworker/wt/tt-bio-rfdiffusion3-port-p10")
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG})
    pf = featurize(PDB, spec)
    rf = torch.load(REF_PT, weights_only=False)

    print("=" * 70)
    print(f"PORTED I={pf['restype'].shape[0]} L={pf['ref_pos'].shape[0]} | "
          f"REF I={rf['restype'].shape[0]} L={rf['ref_pos'].shape[0]}")
    print("=" * 70)

    # per-token atom counts
    def atom_counts(f):
        import collections
        c = collections.Counter(f["atom_to_token_map"].tolist())
        return [c.get(t, 0) for t in range(f["restype"].shape[0])]
    pc = atom_counts(pf); rc = atom_counts(rf)
    print(f"ported atoms/token (first 12): {pc[:12]}")
    print(f"ref    atoms/token (first 12): {rc[:12]}")
    print(f"ported atoms/token (10-15):    {pc[10:16]}")
    print(f"ref    atoms/token (10-15):    {rc[10:16]}")
    print()

    tok_keys = ["restype", "ref_motif_token_type", "ref_plddt", "is_non_loopy",
                "is_motif_token_unindexed", "is_motif_token_with_fully_fixed_coord",
                "is_protein", "is_rna", "is_dna", "is_ligand", "is_polar",
                "terminus_type", "asym_id", "entity_id", "sym_id",
                "residue_index", "token_index", "token_bonds", "unindexing_pair_mask"]
    print("--- TOKEN-LEVEL value comparison (I=40) ---")
    mismatches = []
    for k in tok_keys:
        if k not in pf or k not in rf:
            print(f"  {k:35s} MISSING (ported={'Y' if k in pf else 'N'} ref={'Y' if k in rf else 'N'})")
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
        if not be and a.numel() <= 80:
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
    for k in atom_keys:
        if k not in pf or k not in rf:
            print(f"  {k:30s} MISSING (ported={'Y' if k in pf else 'N'} ref={'Y' if k in rf else 'N'})")
            atom_mismatches.append(k); continue
        a, b = pf[k], rf[k]
        if a.shape != b.shape:
            print(f"  {k:30s} SHAPE {list(a.shape)} vs {list(b.shape)}  MISMATCH")
            atom_mismatches.append(k); continue
        be = torch.equal(a, b)
        p = pcc(a, b) if a.dtype.is_floating_point else 1.0
        status = "OK" if be else "DIFF"
        if not be:
            atom_mismatches.append(k)
        print(f"  {k:30s} {status:4s} bitexact={be} pcc={p:.4f}  {list(a.shape)} {a.dtype}")

    print()
    print(f"TOKEN-LEVEL MISMATCHES ({len(mismatches)}): {mismatches}")
    print(f"ATOM-LEVEL MISMATCHES ({len(atom_mismatches)}): {atom_mismatches}")

if __name__ == "__main__":
    main()
