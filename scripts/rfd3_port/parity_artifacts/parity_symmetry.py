#!/usr/bin/env python3
"""Compare ported featurizer `f` vs a reference captured `f` for the F5
symmetry case (p18): a fully-unconditional cyclic design, spec =
{"length": 12, "is_non_loopy": true, "symmetry": {"id": "C3"}} -- the same
shape as the real docs/examples/symmetry.md "uncond_C5" example, just a
smaller length for a fast/deterministic capture. No `input` PDB at all
(see module docstring's F5 grounding in tt_bio/rfd3_featurize.py for why a
bare `length`-only spec never requires one).

Reproduce the reference capture (CPU, no ckpt -- same method as
parity_dna.py/parity_enzyme.py):
    /home/ttuser/rfd3_local_env/bin/python scripts/rfd3_port/capture_ref_f_uncond.py \
        --spec_json scripts/rfd3_port/parity_artifacts/symmetry_uncond/spec.json \
        --out_dir /tmp/ref_uncond_c3_capture --seed 42

Known, non-mismatch exclusion: `sym_transform` is a dict of (R,t) tensors,
not a plain tensor -- excluded from the bit-exact tensor loop the same way
the reference's OWN capture script excludes non-tensor `f` values from its
saved golden .pt file. Verified separately (module docstring + this script's
own formula check) against `rfd3.inference.symmetry.frames.get_cyclic_frames`.
"""
import json
import os, sys
import torch

DIR = os.path.dirname(__file__)
SPEC_JSON = os.path.join(DIR, "symmetry_uncond", "spec.json")
REF_PT = "/tmp/ref_uncond_c3_capture/ref_f.pt"


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() == 0: return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0: return 1.0 if torch.allclose(a, b) else float("nan")
    return float((a * b).sum() / denom)


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from tt_bio.rfd3_featurize import featurize, _cyclic_frames
    from tt_bio.rfd3_input import InputSpecification
    if not os.path.exists(REF_PT):
        print(f"SKIPPED: no reference capture at {REF_PT} (see module docstring to reproduce)")
        return
    with open(SPEC_JSON) as fh:
        spec_dict = json.load(fh)
    spec = InputSpecification.from_dict(spec_dict)
    pf = featurize(None, spec)
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
    print("--- ATOM-LEVEL value comparison (incl. F5 sym_* keys) ---")
    atom_keys = ["ref_atom_name_chars", "ref_pos", "ref_mask", "ref_element", "ref_charge",
                 "ref_space_uid", "ref_pos_is_ground_truth", "has_zero_occupancy",
                 "ref_is_motif_atom_with_fixed_coord", "ref_is_motif_atom_unindexed",
                 "ref_atomwise_rasa", "active_donor", "active_acceptor", "is_atom_level_hotspot",
                 "is_motif_atom_with_fixed_coord", "is_motif_atom_with_fixed_seq",
                 "is_motif_atom_unindexed", "motif_pos", "is_ca", "is_central",
                 "is_backbone", "is_sidechain", "is_virtual", "atom_to_token_map",
                 "sym_transform_id", "sym_entity_id", "is_sym_asu"]
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
    print("--- sym_transform formula check (dict, not bit-exact-compared as a tensor) ---")
    frames = _cyclic_frames(3)
    for i, (R, t) in enumerate(frames):
        ok_r = torch.allclose(torch.from_numpy(R), pf["sym_transform"][str(i)][0], atol=1e-6)
        ok_t = torch.allclose(torch.from_numpy(t), pf["sym_transform"][str(i)][1], atol=1e-6)
        print(f"  transform {i}: R matches formula={ok_r} t matches formula={ok_t}")

    print()
    print(f"TOKEN-LEVEL MISMATCHES ({len(mismatches)}): {mismatches}")
    print(f"ATOM-LEVEL MISMATCHES ({len(atom_mismatches)}): {atom_mismatches}  (same_L={same_L})")


if __name__ == "__main__":
    main()
