#!/usr/bin/env python3
"""Score the host-built relpos block against the reference's own input to `linear`.

Comparing final outputs would let a wrong concat ORDER hide behind the linear. This
hooks `relative_position_encoding.linear` and compares the [I, I, 139] tensor the
reference actually feeds it, which pins the column layout too. Both sides are fp32
host integer work, so anything short of bit-exact is a real difference.

    python scripts/rf3_port/parity_relpos.py --ckpt ... --fixture rna
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

SEL = {"template": dict(template_selection=["9dfn_A"]),
       "cyclic": dict(cyclic_chains=["A"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixture", default="rna")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from tt_bio.rf3.feature_init import relpos_features, token_bond_features
    from tt_bio.rf3.featurize import featurize, n_cycle, network_input
    from tt_bio.rf3.weights import load_reference

    d = REPO / "scripts/rf3_port/parity_artifacts" / args.fixture
    prev = os.getcwd()
    os.chdir(d)
    try:
        out = featurize("input.json", n_recycles=2, diffusion_batch_size=1,
                        seed=args.seed, **SEL.get(args.fixture, {}))[0]
    finally:
        os.chdir(prev)

    net, _ = load_reference(args.ckpt, num_steps=2)
    rpe = net.feature_initializer.relative_position_encoding
    pt = net.feature_initializer.process_token_bonds

    seen = {}
    h1 = rpe.linear.register_forward_pre_hook(
        lambda _m, inp: seen.__setitem__("relpos", inp[0].detach().float().cpu()))
    h2 = pt.register_forward_pre_hook(
        lambda _m, inp: seen.__setitem__("bonds", inp[0].detach().float().cpu()))
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            net(input=network_input(out), n_cycle=min(2, n_cycle(out)),
                coord_atom_lvl_to_be_noised=out["coord_atom_lvl_to_be_noised"])
    finally:
        h1.remove(); h2.remove()

    f = out["feats"]
    mine = {
        "relpos": relpos_features(f, r_max=rpe.r_max, s_max=rpe.s_max),
        "bonds": token_bond_features(f),
    }

    rep = {"fixture": args.fixture, "cyclic": bool(len(f.get("cyclic_asym_ids", []) or []))}
    ok = True
    for k in ("relpos", "bonds"):
        want, got = seen[k], mine[k].float()
        same_shape = list(want.shape) == list(got.shape)
        exact = same_shape and bool(torch.equal(want, got))
        rep[k] = {
            "shape_ref": list(want.shape),
            "shape_port": list(got.shape),
            "bit_exact": exact,
            "maxabs": None if not same_shape else round(float((want - got).abs().max()), 8),
        }
        ok = ok and exact
    rep["verdict"] = "PASS (bit-exact)" if ok else "FAIL"
    print(json.dumps(rep, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
