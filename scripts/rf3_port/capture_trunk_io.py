#!/usr/bin/env python3
"""Capture the real (S, Z) entering and leaving RF3's first Pairformer block.

The block-0 parity number was measured on synthetic N(0,1), which this block
amplifies to output std 735 -- so far off-manifold that torch's own bf16 only
reaches 0.982 against its own fp32. This hooks the vendored torch model during a
real forward and saves what the block actually sees, so the port can be scored on
its real operating point instead.

    python scripts/rf3_port/capture_trunk_io.py --ckpt ... \\
        --input scripts/rf3_port/parity_artifacts/glke/input.json --out golden.pt
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
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize, n_cycle, network_input
    from tt_bio.rf3.weights import load_reference

    inp = Path(args.input).resolve()
    prev = os.getcwd()
    os.chdir(inp.parent)  # fixture inputs use paths relative to their own directory
    try:
        out = featurize(inp.name, n_recycles=args.n_recycles, diffusion_batch_size=1,
                        seed=args.seed)[0]
    finally:
        os.chdir(prev)

    net, _ = load_reference(args.ckpt, num_steps=2)
    block = net.recycler.pairformer_stack[args.block]

    captured = {}

    def hook(_mod, inputs, output):
        # Keep the LAST recycle: that is the operating point the trunk converges to.
        s_in, z_in = inputs[0], inputs[1]
        s_out, z_out = output
        captured["in"] = (s_in.detach().float().cpu(), z_in.detach().float().cpu())
        captured["out"] = (s_out.detach().float().cpu(), z_out.detach().float().cpu())

    handle = block.register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            net(input=network_input(out),
                n_cycle=min(args.n_recycles, n_cycle(out)),
                coord_atom_lvl_to_be_noised=out["coord_atom_lvl_to_be_noised"])
    finally:
        handle.remove()

    if "in" not in captured:
        print("block never fired; wrong index?")
        return 1

    torch.save(captured, args.out)
    s_in, z_in = captured["in"]
    s_out, z_out = captured["out"]
    print(json.dumps({
        "block": args.block,
        "tokens": int(z_in.shape[-2]),
        "s_in_std": round(float(s_in.std()), 4),
        "z_in_std": round(float(z_in.std()), 4),
        "s_out_std": round(float(s_out.std()), 4),
        "z_out_std": round(float(z_out.std()), 4),
        "out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
