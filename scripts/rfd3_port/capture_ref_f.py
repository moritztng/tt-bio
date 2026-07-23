#!/usr/bin/env python3
"""Capture the reference foundry featurizer's `f` for a given PDB+contig.

Builds the inference transform pipeline standalone (no model/checkpoint needed),
runs ContigJsonDataset.__getitem__, and dumps every tensor under data["f"] plus a
meta.json describing shapes/dtypes.  Output goes to --out_dir.
"""
import argparse, os, sys, json, traceback
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--contig", required=True)
    ap.add_argument("--example_id", default="test")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from rfd3.transforms.pipelines import build_atom14_base_pipeline
    from rfd3.inference.datasets import ContigJsonDataset

    print("[capture] building inference pipeline (CPU) ...", flush=True)
    # Canonical inference kwargs from configs/datasets/design_base.yaml +
    # configs/model/components/rfd3_net.yaml.  sigma_perturb=0 for a deterministic
    # apples-to-apples comparison against the ported (noise-free) featurizer.
    atom_1d_features = {
        "ref_atom_name_chars": 256, "ref_element": 128, "ref_charge": 1,
        "ref_mask": 1, "ref_is_motif_atom_with_fixed_coord": 1,
        "ref_is_motif_atom_unindexed": 1, "has_zero_occupancy": 1, "ref_pos": 3,
        "ref_atomwise_rasa": 3, "active_donor": 1, "active_acceptor": 1,
        "is_atom_level_hotspot": 1,
    }
    token_1d_features = {
        "ref_motif_token_type": 3, "restype": 32, "ref_plddt": 1, "is_non_loopy": 1,
    }
    pipeline = build_atom14_base_pipeline(
        is_inference=True,
        sigma_data=16,
        diffusion_batch_size=8,
        generate_conformers=True,
        provide_reference_conformer_when_unmasked=True,
        ground_truth_conformer_policy="IGNORE",
        use_element_for_atom_names_of_atomized_tokens=True,
        n_atoms_per_token=14,
        central_atom="CB",
        atom_1d_features=atom_1d_features,
        token_1d_features=token_1d_features,
        center_option="diffuse",
        sigma_perturb=0.0,
        sigma_perturb_com=0.0,
    )

    data = {args.example_id: {"input": os.path.abspath(args.pdb), "contig": args.contig}}
    print("[capture] building dataset ...", flush=True)
    ds = ContigJsonDataset(
        data=data,
        cif_parser_args=None,
        transform=pipeline,
        name="capture-dataset",
        subset_to_keys=None,
        eval_every_n=1,
    )
    print(f"[capture] dataset len={len(ds)}; running featurize on '{args.example_id}' ...", flush=True)
    out = ds[0]

    # locate feats
    feats = out.get("feats", out.get("f", None))
    if feats is None:
        # some pipelines nest under a different key; dump top-level keys for diagnosis
        print("[capture] WARNING: no 'f' key. top-level keys:", list(out.keys()), flush=True)
        sys.exit(2)

    print(f"[capture] got f with {len(feats)} keys", flush=True)
    meta = {}
    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            meta[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        else:
            meta[k] = {"type": str(type(v)), "value": str(v)[:200]}
    # also record a few scalar fields for sanity
    with open(os.path.join(args.out_dir, "ref_f.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    # save tensors
    td = os.path.join(args.out_dir, "ref_f.pt")
    torch.save({k: v for k, v in feats.items() if isinstance(v, torch.Tensor)}, td)
    print(f"[capture] saved {len([v for v in feats.values() if isinstance(v, torch.Tensor)])} tensors -> {td}", flush=True)
    print(f"[capture] meta -> {os.path.join(args.out_dir, 'ref_f.meta.json')}", flush=True)
    # quick summary
    if "restype" in feats and isinstance(feats["restype"], torch.Tensor):
        print("  I =", feats["restype"].shape[0], " L =", feats["ref_pos"].shape[0] if "ref_pos" in feats else "?", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
