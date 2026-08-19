#!/usr/bin/env python3
"""Capture the upstream RF3 featurizer output for one input, for parity scoring.

RF3 (RosettaCommons/foundry, models/rf3) builds its feature dict inside the
inference engine, and the transform-pipeline config lives in the checkpoint, so
the faithful capture path is to instantiate the real engine and stop right after
``self.pipeline(input_spec.to_pipeline_input())`` -- before the network runs.
That costs a checkpoint load but keeps the captured `f` exactly what inference
would see.

Writes ``ref_f.pt`` (every tensor under the pipeline output, flattened with
``/``-joined keys) and ``ref_f.meta.json`` (shape/dtype per key) into --out_dir.
The committed capture is what ``scripts/rf3_port/parity_gate.py`` scores the
ported featurizer against, so the gate needs neither a foundry install nor a
device.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch


def _split(obj, prefix="", tensors=None, plain=None):
    """Split the pipeline output into {"a/b": tensor} and {"a/b": json-able value}.

    Not every feature is a tensor: ``feats/cyclic_asym_ids`` is a plain Python list
    and the model reads it (``pairformer_layers.py`` RelativePositionEncoding). A
    tensors-only capture drops it silently, so keep both halves.
    """
    tensors = {} if tensors is None else tensors
    plain = {} if plain is None else plain
    key = prefix.rstrip("/")
    if isinstance(obj, torch.Tensor):
        tensors[key] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _split(v, f"{prefix}{k}/", tensors, plain)
    elif isinstance(obj, (list, tuple)) and any(
        isinstance(v, (torch.Tensor, dict)) for v in obj
    ):
        for i, v in enumerate(obj):
            _split(v, f"{prefix}{i}/", tensors, plain)
    else:
        try:
            json.dumps(obj)
            plain[key] = obj
        except (TypeError, ValueError):
            plain[key] = {"__repr__": repr(obj)[:500], "__type__": str(type(obj))}
    return tensors, plain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="cif/pdb/json input path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_recycles", type=int, default=10)
    ap.add_argument("--diffusion_batch_size", type=int, default=5)
    ap.add_argument("--num_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_selection", default=None)
    ap.add_argument("--ground_truth_conformer_selection", default=None)
    ap.add_argument("--cyclic_chains", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.inference import prepare_inference_inputs_from_paths

    print("[capture] building engine (this loads the checkpoint) ...", flush=True)
    engine = RF3InferenceEngine(
        ckpt_path=args.ckpt,
        n_recycles=args.n_recycles,
        diffusion_batch_size=args.diffusion_batch_size,
        num_steps=args.num_steps,
        seed=args.seed,
        devices_per_node=1,
        num_nodes=1,
        metrics_cfg=None,
    )
    engine.initialize()

    def _sel(v):
        if v is None:
            return None
        return [s.strip() for s in v.split(",")]

    inference_inputs = prepare_inference_inputs_from_paths(
        inputs=os.path.abspath(args.input),
        existing_outputs_dir=None,
        sharding_pattern=None,
        template_selection=_sel(args.template_selection),
        ground_truth_conformer_selection=_sel(args.ground_truth_conformer_selection),
        add_missing_atoms=True,
    )
    if args.cyclic_chains:
        for spec in inference_inputs:
            spec.cyclic_chains = _sel(args.cyclic_chains)
    print(f"[capture] {len(inference_inputs)} example(s) from {args.input}", flush=True)

    spec = inference_inputs[0]
    print(f"[capture] running pipeline on {spec.example_id} ...", flush=True)
    torch.manual_seed(args.seed)
    pipeline_output = engine.pipeline(spec.to_pipeline_input())

    flat, plain = _split(pipeline_output)
    meta = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in flat.items()}
    meta["__non_tensor__"] = plain
    meta["__example_id__"] = spec.example_id
    meta["__input__"] = os.path.basename(args.input)
    meta["__n_recycles__"] = args.n_recycles
    meta["__diffusion_batch_size__"] = args.diffusion_batch_size
    meta["__seed__"] = args.seed

    torch.save(flat, os.path.join(args.out_dir, "ref_f.pt"))
    with open(os.path.join(args.out_dir, "ref_f.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f"[capture] saved {len(flat)} tensors + {len(plain)} non-tensor values "
          f"-> {args.out_dir}/ref_f.pt", flush=True)

    f = pipeline_output.get("feats", pipeline_output.get("f"))
    if isinstance(f, dict):
        print(f"[capture] f keys ({len(f)}): {sorted(f.keys())}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
