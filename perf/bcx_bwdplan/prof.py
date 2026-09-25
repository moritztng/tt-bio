#!/usr/bin/env python3
"""bcx-bwdplan: device kernel time of one real block's backward, per arm, in ONE profiled process.

`realcensus.py prof` with an arm loop: every arm is warmed, the warm-up is drained, and each
timed backward sits between `bwd <arm> <stack> K=<k> rep=<r>` and `end ...` signposts, so the
realcensus numbers (arm `stack`) are re-taken beside the lever arms under the same build and clock.

  prof  run under `python -m tracy -r` on the Tracy build (see run_prof.sh)
  sum   reduce the ops report: device kernel seconds per backward, by op class, and the op
        signatures whose time moved most between the first arm and each other arm
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_realcensus import realcensus as RC  # noqa: E402

S = RC.S
OUT = ROOT / "perf" / "bcx_bwdplan"


def cmd_prof(args):
    import ttnn
    lv, dev, ref = S.open_all(args)
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)
    arms = args.arms.split(",")
    blob = {"stamp": S.stamp(args, clock), "build": RC.build(), "arms": arms, "n": args.n,
            "note": "wall under the device profiler: perturbed, attribution only", "points": []}
    for arm in arms:
        lv.arm(arm)
        for stack_name, k in RC.points(args):
            for _ in range(args.warm):
                S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=k, ckpt=True)
    ttnn.ReadDeviceProfiler(dev.device)               # drain the warm-up out of the buffers
    ag = dev.ag
    for rep in range(args.steps):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            lv.arm(arm)
            for stack_name, k in RC.points(args):
                tag = f"{arm} {stack_name} K={k} rep={rep}"
                ke, kv = (k, 0) if stack_name == "extra" else (0, k)
                ml, zl = dev.leaf(m0), dev.leaf(z0)
                dev.sync()
                mask = dev.up(S.torch.ones(1, args.n)) if lv.mask else None
                with dev.tt.tape():
                    mo, zo = dev.stack(ml, zl, ke, kv, ckpt=True, msa_mask=mask)
                dev.sync()
                roots = [zo] if stack_name == "extra" else [mo, zo]
                seeds = ([dev.seed(wz, zo)] if stack_name == "extra"
                         else [dev.seed(wm, mo), dev.seed(wz, zo)])
                dev.sync()
                RC.signpost(f"bwd {tag}")
                t2 = time.time()
                ag.backward(roots, seeds)
                dev.sync()
                t3 = time.time()
                RC.signpost(f"end {tag}")
                ag.release_pins()
                del mo, zo, ml, zl, roots, seeds
                blob["points"].append({"tag": tag, "bwd": t3 - t2, "aiclk": clock.window([(t2, t3)]),
                                       "load1": os.getloadavg()[0]})
                print(json.dumps(blob["points"][-1]), flush=True)
                ttnn.ReadDeviceProfiler(dev.device)
    clock.stop()
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / args.out}", flush=True)


def cmd_sum(args):
    rep = args.report or sorted(glob.glob(f"{args.profdir}/reports/*/ops_perf_results_*.csv"))[-1]
    seg, segs = None, collections.defaultdict(list)
    for r in RC._read_report(rep):
        if (r.get("OP TYPE") or "").lower() == "signpost":
            seg = r["OP CODE"] if r["OP CODE"].startswith("bwd ") else None
            continue
        if seg is not None and r.get("DEVICE KERNEL DURATION [ns]"):
            segs[seg[4:]].append(r)

    def dur(r):
        return float(r["DEVICE KERNEL DURATION [ns]"]) * 1e-9

    def sig(r):
        shape = "x".join(str(d) for d in (RC._dims(r, "INPUT_0") or []))
        return f"{r['OP CODE']} {shape} cores={r.get('CORE COUNT', '')}"

    by = collections.defaultdict(list)                # (arm, stack, K) -> [per-rep dict]
    for tag, rs in segs.items():
        arm, stack_name, k, _ = tag.split(" ")
        cls, sg = collections.Counter(), collections.Counter()
        for r in rs:
            cls[RC.op_class(r["OP CODE"])] += dur(r)
            sg[sig(r)] += dur(r)
        by[(arm, stack_name, k)].append({"total": sum(dur(r) for r in rs), "ops": len(rs),
                                         "cls": cls, "sig": sg})
    out = {"report": rep, "points": {}}
    for (arm, stack_name, k), reps in sorted(by.items()):
        out["points"][f"{arm} {stack_name} {k}"] = {
            "device_s": [round(x["total"], 5) for x in reps],
            "median_s": round(float(np.median([x["total"] for x in reps])), 5),
            "ops": [x["ops"] for x in reps],
            "by_class_s": {c: round(float(np.median([x["cls"][c] for x in reps])), 5)
                           for c in sorted({c for x in reps for c in x["cls"]})}}
    base = args.arms.split(",")[0] if args.arms else None
    moved = {}
    for (arm, stack_name, k), reps in by.items():
        if base is None or arm == base or (base, stack_name, k) not in by:
            continue
        b = by[(base, stack_name, k)][0]["sig"]
        a = reps[0]["sig"]
        diff = sorted(((b[s] - a[s], s) for s in set(a) | set(b)), reverse=True)
        moved[f"{arm} {stack_name} {k}"] = [[round(d * 1e3, 3), s] for d, s in diff[:args.top]]
    out["moved_vs_" + str(base)] = moved
    for key, v in out["points"].items():
        print(key, v["median_s"], v["ops"], v["by_class_s"])
    (OUT / args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prof", "sum"])
    ap.add_argument("--params", default=S.A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--arms", default="stack,bwd1,bwd2,bwd4,bwd")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--ks", default="1")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profdir", default="/dev/shm/bcx-bp-prof")
    ap.add_argument("--report", default=None)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default="prof_host.json")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "prof":
        import torch
        torch.set_num_threads(args.threads)
    {"prof": cmd_prof, "sum": cmd_sum}[args.cmd](args)


if __name__ == "__main__":
    main()
