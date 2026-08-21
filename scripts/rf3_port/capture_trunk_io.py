#!/usr/bin/env python3
"""Capture what a module really sees during an RF3 reference forward.

Scoring a ported block on synthetic tensors measures the input as much as the port.
The first Pairformer block turns N(0,1) into output std 735 where its real operating
point is 79, and that far off-manifold torch's own bf16 only reaches 0.982 against
its own fp32 -- so a synthetic score was capped below the gate before the device was
involved. This hooks the vendored torch model during a real forward and saves the
module's actual input and output, to score against instead.

    python scripts/rf3_port/capture_trunk_io.py --ckpt ... \\
        --input scripts/rf3_port/parity_artifacts/glke/input.json --out golden.pt

`--module` takes a dotted path relative to the network, so any block can be captured:

    --module recycler.pairformer_stack.0     (default)
    --module recycler.msa_module
    --module feature_initializer
    --module diffusion_module.diffusion_transformer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--module", default="recycler.pairformer_stack.0",
                    help="dotted path of the module to hook, relative to the network")
    ap.add_argument("--seed", type=int, default=42)
    # Without these the template and cyclic fixtures capture with their track OFF,
    # and a module scored on that all-off condition passes while proving nothing.
    ap.add_argument("--template_selection", default=None)
    ap.add_argument("--ground_truth_conformer_selection", default=None)
    ap.add_argument("--cyclic_chains", default=None)
    args = ap.parse_args()

    def sel(v):
        return [x.strip() for x in v.split(",")] if v else None

    from tt_bio.rf3.featurize import featurize, n_cycle, network_input
    from tt_bio.rf3.weights import load_reference

    inp = Path(args.input).resolve()
    prev = os.getcwd()
    os.chdir(inp.parent)  # fixture inputs use paths relative to their own directory
    try:
        out = featurize(inp.name, n_recycles=args.n_recycles, diffusion_batch_size=1,
                        seed=args.seed,
                        template_selection=sel(args.template_selection),
                        ground_truth_conformer_selection=sel(
                            args.ground_truth_conformer_selection),
                        cyclic_chains=sel(args.cyclic_chains))[0]
    finally:
        os.chdir(prev)

    net, _ = load_reference(args.ckpt, num_steps=2)
    block = net
    for part in args.module.split("."):
        block = block[int(part)] if part.isdigit() else getattr(block, part)

    captured = {}

    def detach(obj):
        if isinstance(obj, torch.Tensor):
            # Integer tensors are INDEX features (chiral_centers, atom_to_token_map,
            # residue_index, ...). Blanket .float() makes them unusable as indices and
            # the failure surfaces far from here -- `chiral_centers` came back float and
            # IndexError'd inside the loss module. Only cast what is actually float.
            if not obj.is_floating_point():
                return obj.detach().cpu()
            return obj.detach().float().cpu()
        if isinstance(obj, (list, tuple)):
            return tuple(detach(o) for o in obj)
        if isinstance(obj, dict):
            return {k: detach(v) for k, v in obj.items()
                    if isinstance(v, (torch.Tensor, list, tuple))}
        return None

    def hook(_mod, inputs, output):
        # Overwritten each recycle, so what lands is the LAST one: the operating
        # point the trunk has converged to, which is what the port has to match.
        captured["in"] = detach(inputs)
        captured["out"] = detach(output)

    handle = block.register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            net(input=network_input(out),
                n_cycle=min(args.n_recycles, n_cycle(out)),
                coord_atom_lvl_to_be_noised=out["coord_atom_lvl_to_be_noised"])
    finally:
        handle.remove()

    if "in" not in captured:
        print(f"{args.module} never fired; wrong path?")
        return 1

    torch.save(captured, args.out)

    def describe(obj, label):
        if isinstance(obj, torch.Tensor):
            # Integer tensors have no std; they are index features, and their range is
            # the informative thing about them.
            if not obj.is_floating_point():
                lo = int(obj.min()) if obj.numel() else 0
                hi = int(obj.max()) if obj.numel() else 0
                return [f"{label} {list(obj.shape)} {obj.dtype} [{lo}..{hi}]"]
            return [f"{label} {list(obj.shape)} std={float(obj.std()):.4f}"]
        if isinstance(obj, (list, tuple)):
            rows = []
            for i, o in enumerate(obj):
                rows += describe(o, f"{label}[{i}]")
            return rows
        if isinstance(obj, dict):
            rows = []
            for k, v in obj.items():
                rows += describe(v, f"{label}[{k!r}]")
            return rows
        return []

    print(f"module: {args.module}")
    for row in describe(captured["in"], "in") + describe(captured["out"], "out"):
        print("  " + row)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
