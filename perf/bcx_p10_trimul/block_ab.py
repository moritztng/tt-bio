#!/usr/bin/env python3
"""A/B the dX route of every product against a 2-D weight, on one AF2 pair block.

Arms alternate at the block boundary inside ONE process on ONE card, so the card's state, the
program cache and the host's load are shared. Arm `off` is today's route (autograd._via2d, which
declines at AF2's ragged token axis, leaving ttnn.matmul with a 4-D activation and transpose_b);
arm `on` is autograd.dgrad_2d through experimental.minimal_matmul, the kernel the forward
already runs on the same operand.

Two things come out of one run: the seconds, and the gradient each arm produces graded against
a float64 reference VJP of the same block on the same bf16-rounded inputs, with the host's own
bf16 arm beside it as the floor nobody can beat.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_stack"))
OUT = ROOT / "perf" / "bcx_p10_trimul"


def rel(a, b):
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=275)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--stack", default="evo")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--grade", action="store_true", help="also grade one block against float64")
    ap.add_argument("--out", default="block_ab.json")
    args = ap.parse_args()
    args.card = int(os.environ.get("TT_VISIBLE_DEVICES", "0"))
    args.params = None
    torch.set_num_threads(args.threads)

    import ttnn                                                       # noqa: F401
    from perf.bcx_stack import stack as S
    from perf.bcx_afgrad import afgrad as A
    from tt_bio import autograd as AG

    args.params = A.DEFAULT_PARAMS
    lv, dev, ref = S.open_all(args)
    lv.mask = True
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)

    for _ in range(2):                                                # JIT + program cache
        S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)

    recs = []
    for rep in range(args.reps):
        for arm in (["off", "on"] if rep % 2 == 0 else ["on", "off"]):
            AG.DGRAD_2D_MINIMAL = (arm == "on")
            before = dict(AG.DGRAD_2D_STATS)
            r, _ = S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
            served = {k: AG.DGRAD_2D_STATS[k] - before[k] for k in before}
            rec = {"rep": rep, "arm": arm, "fwd": round(r["fwd"], 6),
                   "bwd": round(r["bwd"], 6), "served": served,
                   "aiclk": clock.window(r["spans"]), "load1": round(os.getloadavg()[0], 2)}
            recs.append(rec)
            print(json.dumps(rec), flush=True)

    grade = {}
    if args.grade:
        grade = do_grade(args, A, AG, dev, ref, torch)
        print(json.dumps({"grade": grade}, default=str), flush=True)

    clock.stop()
    blob = {"stamp": S.stamp(args, clock), "n": args.n, "k": args.k, "stack": args.stack,
            "reps": args.reps, "records": recs, "grade": grade,
            "loadavg_end": os.getloadavg()}
    for arm in ("off", "on"):
        xs = [r for r in recs if r["arm"] == arm]
        blob[arm] = {"fwd": S.dist([r["fwd"] for r in xs]),
                     "bwd": S.dist([r["bwd"] for r in xs])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print("wrote " + str(OUT / args.out), flush=True)


def do_grade(args, A, AG, dev, ref, torch):
    """One Evoformer block's input gradients per arm, against a float64 reference VJP.

    The reference is `af2_reference` promoted to float64 on the SAME bf16-rounded inputs, which
    is what makes the comparison about the device arm rather than about the input rounding. The
    host's own bf16 arm is reported beside it: that is the error the port already accepts, so an
    arm inside it has spent nothing.
    """
    n = args.n
    torch.manual_seed(args.seed)
    logits = torch.randn(n, 20) * 2.0
    ridx = torch.arange(n)
    with torch.no_grad():
        msa0, pair0 = A.embed(ref["f64"], logits.double(), ridx)
    min_, zin = A.bf(msa0), A.bf(pair0)
    gm = torch.randn(min_.shape, dtype=torch.float64) / min_.numel() ** 0.5
    gz = torch.randn(zin.shape, dtype=torch.float64) / zin.numel() ** 0.5

    arms = {}
    for name in ("f64", "bf16"):
        mod = ref[name]
        dt = mod.trunk_dtype
        g, _ = A.ref_vjp(lambda a, b, mod=mod: A.ref_evo(mod, 0, a, b),
                         [min_.to(dt), zin.to(dt)], [gm, gz])
        arms[name] = {"dm": g[0].double(), "dz": g[1].double()}
    r64 = arms["f64"]

    out = {"host_bf16": {k: rel(arms["bf16"][k], r64[k]) for k in ("dm", "dz")}}
    for arm in ("off", "on"):
        AG.DGRAD_2D_MINIMAL = (arm == "on")
        before = dict(AG.DGRAD_2D_STATS)
        gc.collect()
        zl, ml = dev.leaf(zin), dev.leaf(min_)
        with dev.tt.tape():
            mo, zo = dev.evo(0, ml, zl, dev.up(torch.ones(min_.shape[:-1])))
            roots = [mo, zo]
            seeds = [dev.seed(gm, mo), dev.seed(gz, zo)]
        dev.ag.backward(roots, seeds)
        d = {"dm": dev.grad(ml, min_.shape).double(),
             "dz": dev.grad(zl, zin.shape).double()}
        out[arm] = {"served": {k: AG.DGRAD_2D_STATS[k] - before[k] for k in before},
                    **{f"rel_{k}": rel(d[k], r64[k]) for k in ("dm", "dz")},
                    **{f"cos_{k}": cos(d[k], r64[k]) for k in ("dm", "dz")}}
        out[arm]["_g"] = d
        del mo, zo, ml, zl, roots, seeds
    a, b = out["off"].pop("_g"), out["on"].pop("_g")
    out["on_vs_off"] = {f"rel_{k}": rel(b[k], a[k]) for k in ("dm", "dz")}
    out["on_vs_off"]["bit_exact"] = all(torch.equal(a[k], b[k]) for k in ("dm", "dz"))
    for arm in ("off", "on"):
        out[arm]["x_float64_vs_host_bf16"] = {
            k: out[arm][f"rel_{k}"] / out["host_bf16"][k] for k in ("dm", "dz")}
    return out


if __name__ == "__main__":
    main()
