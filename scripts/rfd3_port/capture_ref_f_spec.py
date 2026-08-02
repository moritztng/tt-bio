#!/usr/bin/env python3
"""Capture the reference foundry featurizer's `f` for a full JSON InputSpecification
(ligand/select_buried/select_exposed/etc, not just --contig). Same pipeline config as
capture_ref_f.py; --spec_json is a path to a one-key JSON dict (the example's spec)."""
import argparse, os, sys, json, traceback

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--spec_json", required=True, help="path to JSON spec dict (minus 'input')")
    ap.add_argument("--example_id", default="test")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from rfd3.transforms.pipelines import build_atom14_base_pipeline
    from rfd3.inference.datasets import ContigJsonDataset

    print("[capture] building inference pipeline (CPU) ...", flush=True)
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

    with open(args.spec_json) as f:
        spec = json.load(f)
    spec["input"] = os.path.abspath(args.pdb)
    data = {args.example_id: spec}
    print("[capture] building dataset with spec:", json.dumps(spec)[:300], flush=True)
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

    feats = out.get("feats", out.get("f", None))
    if feats is None:
        print("[capture] WARNING: no 'f' key. top-level keys:", list(out.keys()), flush=True)
        sys.exit(2)

    print(f"[capture] got f with {len(feats)} keys", flush=True)
    meta = {}
    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            meta[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        else:
            meta[k] = {"type": str(type(v)), "value": str(v)[:200]}
    with open(os.path.join(args.out_dir, "ref_f.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    td = os.path.join(args.out_dir, "ref_f.pt")
    torch.save({k: v for k, v in feats.items() if isinstance(v, torch.Tensor)}, td)
    print(f"[capture] saved {len([v for v in feats.values() if isinstance(v, torch.Tensor)])} tensors -> {td}", flush=True)
    if "restype" in feats and isinstance(feats["restype"], torch.Tensor):
        print("  I =", feats["restype"].shape[0], " L =", feats["ref_pos"].shape[0] if "ref_pos" in feats else "?", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
