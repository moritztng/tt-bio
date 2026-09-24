#!/usr/bin/env python3
"""Does the loop's own `EvoformerOnDevice` capture, replay and agree bit for bit?

Drives `splice.EvoformerOnDevice._taped` and `._backward` -- the two functions BindCraft 2's
`custom_vjp` callbacks call -- with a padded, masked complex the loop produces, eager and traced
interleaved in one process, alternating order. Nothing here reimplements the trunk.

The eager arm runs with `autograd.DEVICE_ZEROS` OFF, which is the shipped default, so a
bit-identical verdict covers both changes the traced arm makes at once.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack"),
          str(ROOT / "perf" / "bcx_predictor")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "perf" / "bcx_tracewire"


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float32))
                          .tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--pad", type=int, default=13, help="masked tail residues")
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region-mb", type=int, default=768)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import trace_wire
    trace_wire.open_traced_device(args.region_mb)
    import afgrad as A
    import stack as S
    from splice import EvoformerOnDevice

    lv = S.Levers()
    dm, ref = A.load_models(args.params or A.DEFAULT_PARAMS, refs=("bf16",))
    dev = A.Dev(dm.to_device())
    lv.arm("stack")
    clock = S.Clock(dt=0.1)

    n = args.n
    m0, z0, wm, wz = S.inputs(ref, n, args.seed)
    # BindCraft 2 hands the callback host arrays at the UNPADDED length with the pad already
    # masked out of seq_mask; `splice._pad_inputs` does the padding. Feed it the same way.
    seq = np.ones(n, dtype=np.float32)
    if args.pad:
        seq[n - args.pad:] = 0.0
    # BindCraft 2 hands the callback float32 (`splice.as_jax` casts); `S.inputs` is bf16.
    msa_np = m0.float().numpy()
    pair_np = z0.float().numpy()
    mask_np = seq[None, :].copy()
    pair_mask_np = (seq[:, None] * seq[None, :]).astype(np.float32)
    cot_m, cot_z = wm.float().numpy(), wz.float().numpy()

    def step(evo):
        gc.collect()
        dev.sync()
        t0, c0 = time.time(), time.process_time()
        out_m, out_z, tok = evo._taped(msa_np, pair_np, mask_np, pair_mask_np)
        t1, c1 = time.time(), time.process_time()
        g_m, g_z = evo._backward(tok, cot_m, cot_z)
        t2, c2 = time.time(), time.process_time()
        return ([out_m, out_z, g_m, g_z],
                {"fwd": t1 - t0, "bwd": t2 - t1, "step": t2 - t0,
                 "cpu": (c1 - c0) + (c2 - c1), "load1": os.getloadavg()[0],
                 "spans": [(t0, t2)]})

    blob = {"stamp": A.stamp(args.card) | {"pci": S.sysfs_node()[1], "argv": sys.argv,
                                           "aiclk_node": clock.path,
                                           "region_mb": args.region_mb},
            "n": n, "pad": args.pad, "k_evo": args.evo, "reps": args.reps}

    dev.ag.DEVICE_ZEROS = False
    eager_evo = EvoformerOnDevice(dev, k_evo=args.evo, trace=False)
    step(eager_evo)                                   # warm: compile + program cache
    ref_out, _ = step(eager_evo)
    blob["eager_digest"] = [digest(a) for a in ref_out]
    print("eager digests", blob["eager_digest"], flush=True)

    traced_evo = EvoformerOnDevice(dev, k_evo=args.evo, trace=True)   # flips DEVICE_ZEROS on
    t0 = time.time()
    first, _ = step(traced_evo)
    blob["capture_s"] = round(time.time() - t0, 2)
    blob["trace"] = traced_evo.wire.stats()
    print("captured+first replay in", blob["capture_s"], "s", flush=True)

    runs, digs = {"eager": [], "trace": []}, {"eager": [], "trace": []}
    for rep in range(args.reps):
        for arm in (["eager", "trace"] if rep % 2 == 0 else ["trace", "eager"]):
            dev.ag.DEVICE_ZEROS = (arm == "trace")
            outs, r = step(traced_evo if arm == "trace" else eager_evo)
            runs[arm].append(r)
            digs[arm].append([digest(a) for a in outs])
            print(arm, rep, {k: round(v, 3) for k, v in r.items() if k != "spans"}, flush=True)
    for arm, rs in runs.items():
        blob[arm] = {"per": {k: S.dist([r[k] for r in rs])
                             for k in ("fwd", "bwd", "step", "cpu")},
                     "load1": [round(r["load1"], 1) for r in rs],
                     "aiclk": clock.window([s for r in rs for s in r["spans"]]),
                     "digests": digs[arm],
                     "reps_identical": all(d == digs[arm][0] for d in digs[arm])}
    blob["bit_identical"] = digs["trace"][0] == blob["eager_digest"]
    blob["x_step"] = blob["eager"]["per"]["step"]["median"] / blob["trace"]["per"]["step"]["median"]
    blob["host_removed_s"] = (blob["eager"]["per"]["step"]["median"]
                              - blob["trace"]["per"]["step"]["median"])
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (args.out or f"smoke_n{n}_pad{args.pad}_k{args.evo}.json")
    path.write_text(json.dumps(blob, indent=1, default=str))
    print(json.dumps({k: blob[k] for k in ("bit_identical", "x_step", "host_removed_s",
                                           "capture_s")}), flush=True)
    print("eager", blob["eager"]["per"]["step"], blob["eager"]["aiclk"], flush=True)
    print("trace", blob["trace"]["per"]["step"], blob["trace"]["aiclk"], flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
