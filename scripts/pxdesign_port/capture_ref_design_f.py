#!/usr/bin/env python3
"""Capture the upstream PXDesign design featurizer's features for a target YAML.

Writes a SMALL committed reference so the parity gate needs no upstream install, no
network and no device: the design-specific tensors in full, plus shape/dtype/sha256 for
every key the upstream featurizer produced, so an unexpected key set is caught too.

It also writes `ref_design_inputs.pt`: the model-ready input dict for a generation-only
run, so `ProtenixDesign.design` can be driven end-to-end from an input upstream itself
produced, without a CIF parser in tt-bio.

It also captures the exact INPUTS to `DesignFeaturizer.get_condition_template_feature`
(distogram-atom coords, res_name, mol_type, is_resolved). That function is the port's
sharpest correctness edge -- a 64-bin distogram written only into the sub-block of
resolved non-`xpb` tokens -- and holding its inputs lets the gate score the arithmetic on
its own, separately from the atom-array construction that feeds it.

Needs: a protenix-carrying venv (qb1: ~/protenix_ref_venv/bin/python), the pinned PXDesign
source (qb1: ~/pxdesign_src), and the target already run through `pxdesign parse`.

    ~/protenix_ref_venv/bin/python scripts/pxdesign_port/capture_ref_design_f.py \
        --pxdesign_src ~/pxdesign_src --yaml examples/PDL1_quick_start.yaml \
        --out_dir scripts/pxdesign_port/parity_artifacts/pdl1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

# Every design-specific tensor, plus the token-level identity keys the ported featurizer
# has to agree on for the design keys to even be comparable.
COMMITTED_KEYS = [
    "conditional_templ", "conditional_templ_mask", "hotspot", "restype",
    "condition_token_mask", "design_token_mask", "distogram_rep_atom_mask",
    "asym_id", "entity_id", "sym_id", "residue_index", "token_index",
    "atom_to_token_idx", "is_protein", "is_ligand", "is_dna", "is_rna",
]

# The model-ready input dict for a generation-only run: everything ProtenixDesign.design
# reads, and nothing else. Stored in compact dtypes (see COMPACT) because the two big
# one-hots are int64 upstream and 3.8 MB of the 4.0 MB; `load_design_inputs` casts them
# back. The trunk-only keys (msa, profile, token_bonds, template_*) are deliberately
# absent: PXDesign-d has no trunk and upstream deletes them before the sampler.
MODEL_INPUT_KEYS = [
    # atom level
    "ref_pos", "ref_charge", "ref_element", "ref_atom_name_chars", "ref_mask",
    "ref_space_uid", "atom_to_token_idx",
    # token level
    "restype", "hotspot", "deletion_mean",
    "asym_id", "residue_index", "entity_id", "sym_id", "token_index",
    # pair level (the structural conditioning)
    "conditional_templ", "conditional_templ_mask",
]

# key -> storage dtype. Values are one-hots, small counts or bin indices, so nothing here
# loses information; `load_design_inputs` in tt_bio.pxdesign.fixtures asserts round-trip.
COMPACT = {
    "ref_element": "uint8", "ref_atom_name_chars": "uint8", "ref_mask": "uint8",
    "conditional_templ": "uint8", "conditional_templ_mask": "bool",
    "ref_charge": "float32", "ref_pos": "float32", "restype": "float32",
    "hotspot": "float32", "deletion_mean": "float32",
}


def sha(t):
    """Bytes of the tensor as stored. `reshape(-1)` first: a 0-d tensor cannot be viewed
    as uint8, and several protenix keys (the alignment counts) are 0-d."""
    import torch
    return hashlib.sha256(
        t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pxdesign_src", required=True)
    ap.add_argument("--yaml", required=True, help="path relative to --pxdesign_src")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protenix05", default=None,
                    help="unpacked protenix 0.5.5 wheel, prepended to sys.path. PXDesign is "
                         "written against protenix 0.5; capturing on the box's installed 2.0 "
                         "needs the shim's module re-pointing, and whether that changes any "
                         "feature VALUE is a thing to measure, not to assume. Needs "
                         "PROTENIX_DATA_ROOT_DIR to hold components.cif.")
    args = ap.parse_args()

    if args.protenix05:
        p05 = os.path.abspath(os.path.expanduser(args.protenix05))
        assert os.path.isdir(os.path.join(p05, "protenix")), p05
        sys.path.insert(0, p05)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import upstream_shim
    shims = upstream_shim.install()

    src = os.path.abspath(os.path.expanduser(args.pxdesign_src))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, src)
    os.chdir(src)

    import numpy as np
    import torch

    work = tempfile.mkdtemp(prefix="pxd_capture_")
    from pxdesign.runner.cli import parse_target
    parse_target.callback(args.yaml, work)                 # CIF -> bioassembly pkl.gz + json

    name = os.path.splitext(os.path.basename(args.yaml))[0]
    json_path = os.path.join(work, "tmp", f"{name}.json")
    spec = json.load(open(json_path))
    # parse_target leaves `structure_file` pointing at the CIF it was given; the dataset
    # only accepts the pkl.gz it just wrote beside it.
    pkl = [f for f in os.listdir(os.path.join(work, "tmp")) if f.endswith(".pkl.gz")]
    assert len(pkl) == 1, f"expected one bioassembly pickle, got {pkl}"
    spec[0]["condition"]["structure_file"] = os.path.join(work, "tmp", pkl[0])
    fixed = os.path.join(work, "tmp", f"{name}.fixed.json")
    json.dump(spec, open(fixed, "w"))

    from pxdesign.data.infer_data_pipeline import InferenceDataset
    ds = InferenceDataset(input_json_path=fixed, use_msa=True)
    # __getitem__ swallows exceptions into a placeholder dict, so drive the two stages
    # directly: a capture that silently produced nothing is worse than one that crashes.
    processed = ds.process_sample_dict(ds.inputs[0])
    data, atom_array, _ = ds.process_one(single_sample_dict=processed)
    feats = data["input_feature_dict"]

    missing = [k for k in COMMITTED_KEYS if k not in feats]
    assert not missing, f"upstream did not produce {missing}"

    # Inputs to get_condition_template_feature, verbatim.
    disto = atom_array[atom_array.distogram_rep_atom_mask.astype(bool)]
    cond_in = {
        "coord": torch.tensor(np.asarray(disto.coord)),
        "res_name": list(map(str, disto.res_name)),
        "mol_type": list(map(str, disto.mol_type)),
        "is_resolved": torch.tensor(np.asarray(disto.is_resolved).astype(bool)),
    }

    import protenix as _ptx
    meta = {
        "yaml": args.yaml,
        "protenix": {"file": _ptx.__file__,
                     "version": getattr(_ptx, "__version__", "?")},
        "n_token": int(feats["token_index"].shape[0]),
        "n_atom": int(feats["atom_to_token_idx"].shape[0]),
        "n_distogram_atom": len(disto),
        "shims": shims,
        "committed_keys": COMMITTED_KEYS,
        "all_keys": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256_16": sha(v)}
                     for k, v in sorted(feats.items()) if torch.is_tensor(v)},
    }
    missing = [k for k in MODEL_INPUT_KEYS if k not in feats]
    assert not missing, f"upstream did not produce model inputs {missing}"
    inputs = {}
    for k in MODEL_INPUT_KEYS:
        v = feats[k]
        dt = COMPACT.get(k)
        if dt is None:
            inputs[k] = v.to(torch.int32)
            assert torch.equal(inputs[k].long(), v.long()), f"{k} does not fit int32"
            continue
        c = v.to(getattr(torch, dt))
        assert torch.equal(c.to(v.dtype), v), f"{k} is not exactly representable as {dt}"
        inputs[k] = c
    meta["model_input_keys"] = MODEL_INPUT_KEYS
    meta["model_input_store_dtype"] = {k: str(v.dtype) for k, v in inputs.items()}
    torch.save(inputs, os.path.join(out_dir, "ref_design_inputs.pt"))

    torch.save({k: feats[k] for k in COMMITTED_KEYS}, os.path.join(out_dir, "ref_design_f.pt"))
    torch.save(cond_in, os.path.join(out_dir, "ref_condition_inputs.pt"))
    json.dump(meta, open(os.path.join(out_dir, "ref_design_f.meta.json"), "w"), indent=1)
    shutil.rmtree(work, ignore_errors=True)

    print(f"[capture] {meta['n_token']} tokens, {meta['n_atom']} atoms, "
          f"{len(meta['all_keys'])} upstream keys, {len(COMMITTED_KEYS)} committed, "
          f"{len(MODEL_INPUT_KEYS)} model inputs")
    print(f"[capture] -> {out_dir}")


if __name__ == "__main__":
    main()
