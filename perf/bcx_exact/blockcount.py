#!/usr/bin/env python3
"""The exact-training census of ONE BindCraft 2 Evoformer block, and what its float64 costs.

Card-free. Two halves.

CENSUS reads `perf/bcx_nan/optrace.json` and `optrace_pad.json`, which are not source scans:
`perf/bcx_nan/optrace.py` wraps `taped_ttnn._taped_verb` and records every verb the shim
dispatched in Evoformer block 0 of a real depth-48 BindCraft 2 taped forward at n=288, msa
depth 2, bucket 32, in call order with input and output shapes. Those are exactly the calls
`EXACT_SOFTMAX_STATS["verb"]` and `EXACT_LAYER_NORM_STATS["verb"]` count, so the census is a
runtime count of the instrument's reach per block, taken from a run that already happened.
Each file holds two independent reps; they are asserted equal rather than averaged.

FLOOR measures, on this host, the float64 arithmetic the instrument performs for that census:
`torch.softmax(x.double(), -1)` and `_ln_forward64`'s body, at the traced shapes. It is a
LOWER bound on the instrument's forward cost and it is labelled that way. Excluded, and each
excluded term is real and larger than nothing:

  * `ttnn.to_torch` off the card and `ttnn.from_torch` back onto it, which is the "host round
    trip" the docstring names and which needs a card to measure;
  * the backward, where `_v_exact_softmax` and `_v_exact_layer_norm` re-read x off the card
    and redo the Jacobian in float64;
  * the untaped recycle forward and every other fold in the round.

So FLOOR cannot be turned into a share of the round. It bounds one term of it from below.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics as st
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                                           # noqa: E402

EXACT_OPS = ("softmax", "softmax_in_place", "layer_norm")
TRACES = {"n288_evoblock0": "perf/bcx_nan/optrace.json",
          "n288_evoblock0_padded": "perf/bcx_nan/optrace_pad.json"}


def shape_of(token):
    """`'(63, 4, 288, 288):a6b2eefe'` -> `(63, 4, 288, 288)`."""
    return tuple(int(x) for x in token.split(":")[0].strip("()").rstrip(",").split(",") if x.strip())


def census(ops):
    out = {}
    for o in ops:
        if o["op"] not in EXACT_OPS:
            continue
        sh = shape_of(o["in"][0])
        key = (o["op"], sh)
        rec = out.setdefault(key, {"op": o["op"], "shape": list(sh), "calls": 0,
                                   "elements_each": math.prod(sh)})
        rec["calls"] += 1
    for rec in out.values():
        rec["elements"] = rec["calls"] * rec["elements_each"]
    return sorted(out.values(), key=lambda r: (-r["elements"], r["op"]))


def totals(rows):
    sm = [r for r in rows if r["op"].startswith("softmax")]
    ln = [r for r in rows if r["op"] == "layer_norm"]
    return {"softmax_calls": sum(r["calls"] for r in sm),
            "softmax_elements": sum(r["elements"] for r in sm),
            "layer_norm_calls": sum(r["calls"] for r in ln),
            "layer_norm_elements": sum(r["elements"] for r in ln)}


def time_softmax(shape, reps):
    x = torch.randn(*shape, dtype=torch.float32)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y = torch.softmax(x.double(), -1)
        ts.append(time.perf_counter() - t0)
        del y
    del x
    return ts


def time_layer_norm(shape, reps):
    """`_ln_forward64`'s arithmetic, without the two ttnn conversions around it."""
    x = torch.randn(*shape, dtype=torch.float32)
    d = shape[-1]
    g = torch.randn(d, dtype=torch.float32)
    b = torch.randn(d, dtype=torch.float32)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        x64 = x.double()
        xc = x64 - x64.mean(-1, keepdim=True)
        y64 = xc * torch.rsqrt((xc * xc).mean(-1, keepdim=True) + 1e-5)
        y64 = y64 * g.double() + b.double()
        ts.append(time.perf_counter() - t0)
        del x64, xc, y64
    del x, g, b
    return ts


def floor(rows, reps, threads):
    torch.set_num_threads(threads)
    out = []
    for r in rows:
        shape = tuple(r["shape"])
        ts = (time_softmax(shape, reps) if r["op"].startswith("softmax")
              else time_layer_norm(shape, reps))
        out.append({**r, "threads": threads, "reps": reps,
                    "median_s_per_call": round(st.median(ts), 6),
                    "min_s_per_call": round(min(ts), 6),
                    "s_for_this_block": round(st.median(ts) * r["calls"], 6),
                    "loadavg1": round(os.getloadavg()[0], 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", default="8,1")
    ap.add_argument("--out", default=str(HERE / "BLOCKCOUNT.json"))
    args = ap.parse_args()

    blob = {"host": os.uname().nodename, "when_utc": time.strftime("%FT%TZ", time.gmtime()),
            "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
            "opened_a_device": False, "nproc": os.cpu_count(),
            "loadavg_start": os.getloadavg(), "traces": {}}

    rows = None
    for tag, rel in TRACES.items():
        d = json.loads((ROOT / rel).read_text())
        reps = {k: census(v) for k, v in d.items() if isinstance(v, list) and v
                and isinstance(v[0], dict)}
        keys = sorted(reps)
        agree = all(reps[k] == reps[keys[0]] for k in keys)
        blob["traces"][tag] = {"file": rel, "reps": keys, "reps_agree": agree,
                               "rows": reps[keys[0]], "totals": totals(reps[keys[0]])}
        if tag == "n288_evoblock0":
            rows = reps[keys[0]]

    blob["floor"] = {}
    for th in (int(x) for x in args.threads.split(",")):
        f = floor(rows, args.reps, th)
        sm = sum(r["s_for_this_block"] for r in f if r["op"].startswith("softmax"))
        ln = sum(r["s_for_this_block"] for r in f if r["op"] == "layer_norm")
        blob["floor"][f"threads_{th}"] = {
            "rows": f, "softmax_s_per_block": round(sm, 4),
            "layer_norm_s_per_block": round(ln, 4), "total_s_per_block": round(sm + ln, 4),
            "x48_evoformer_blocks_s": round((sm + ln) * 48, 3),
            "loadavg1": round(os.getloadavg()[0], 1)}

    blob["loadavg_end"] = os.getloadavg()
    pathlib.Path(args.out).write_text(json.dumps(blob, indent=1))
    for tag, t in blob["traces"].items():
        print(tag, json.dumps(t["totals"]), "reps_agree", t["reps_agree"])
    for k, v in blob["floor"].items():
        print(k, json.dumps({x: v[x] for x in v if x != "rows"}))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
