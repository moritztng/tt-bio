#!/usr/bin/env python3
"""The multimer trunk's card-vs-float64 deviation at the sizes BindCraft 2 actually runs.

Every parity number this row has is at 24 tokens, against AlphaFold's own JAX activations. That
arm answers "is the transform right" and it cannot be run at n=186..320: the JAX reference is
48 blocks on CPU and the pair tensor grows as n^2.

This asks the other half of COST 4, which does scale: at the sizes the design loop runs, how far
does the card's bfloat16 trunk sit from the SAME trunk in float64? It cannot catch a transcription
error -- both arms are `tt_bio`'s own code, and the 24-token JAX arm is what rules those out --
so it is a precision reading and is reported as one.

Inputs are the model's own: `afgrad.embed` runs AF2's embedding over a drawn sequence, which is
what BindCraft 2 hands the trunk, rather than random tensors with no relation to the activations
the Evoformer sees.

    PYTHONPATH=.:perf/bcx_afgrad TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> \
        python3 perf/bcx_predictor/size_ladder_parity.py --sizes 192,256,288,320 --blocks 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.double().flatten()
    y = b.double().flatten()
    x = x - x.mean()
    y = y - y.mean()
    d = (x.norm() * y.norm()).item()
    return float((x @ y).item() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="192,256,288,320")
    ap.add_argument("--blocks", type=int, default=4,
                    help="Evoformer blocks per arm. The deviation accumulates with depth, so this "
                         "is part of the reading and is recorded with it.")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params/"
                                        "params_model_1_multimer_v3.npz")
    ap.add_argument("--multimer", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bcx_afgrad"))
    import afgrad as A
    from tt_bio.af2_weights import load_af2_state_dict
    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_reference import load_af2_model

    A._float64_layernorm()
    state = load_af2_state_dict(args.params, multimer=args.multimer)
    dm = load_af2_device_model(state, template=False, multimer=args.multimer,
                               trunk_dtype=torch.bfloat16)
    ref64 = load_af2_model(state, template=False, multimer=args.multimer,
                           trunk_dtype=torch.float64).double()
    for p in ref64.parameters():
        p.requires_grad_(False)
    dev = A.Dev(dm)

    rows = []
    for n in (int(v) for v in args.sizes.split(",")):
        torch.manual_seed(0)
        logits = torch.randn(n, 20)
        idx = torch.arange(n)
        m0, z0 = A.embed(ref64, logits, idx)
        m0, z0 = m0.detach(), z0.detach()

        t0 = time.time()
        m, z = m0.double(), z0.double()
        for i in range(args.blocks):
            m, z = A.ref_evo(ref64, i, m, z)
        host_s = time.time() - t0

        t0 = time.time()
        mo, zo = dev.stack(dev.up(m0.float()), dev.up(z0.float()), 0, args.blocks, ckpt=False)
        dev.sync()
        card_s = time.time() - t0
        m_d = dev.down(mo, tuple(m0.shape))
        z_d = dev.down(zo, tuple(z0.shape))

        row = {"n": n, "blocks": args.blocks,
               "msa_pcc": pcc(m_d, m), "pair_pcc": pcc(z_d, z),
               "pair_rel_max": float(((z_d.double() - z).abs().max()
                                      / z.abs().max()).item()),
               "host_fp64_s": round(host_s, 2), "card_bf16_s": round(card_s, 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"params": os.path.basename(args.params), "multimer": args.multimer,
                       "rows": rows}, fh, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
