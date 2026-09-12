#!/usr/bin/env python3
"""Rank the diffusion step's under-filled programs and price each one against a roof.

"Idle core-fraction-seconds" is what a grid census naturally produces, but on its own it
over-promises: it assumes a program re-spread from 16 cores to 110 runs 6.9x faster, which
stops being true the moment the program hits the byte roof. So every group here carries three
times per call:

  measured      what it costs now, on the cores it actually got
  core-ideal    measured * cores / C -- perfect linear re-spread, the census's own bound
  roof          max(bytes / 429.9 GB/s, flops / 85.96 TFLOP/s) -- where re-spreading stops

and the recoverable time is `measured - max(core_ideal, roof)`, which is the honest one.
Roofs are the campaign's measured Blackhole roofs, not vendor numbers.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

BW = 429.9e9        # measured streaming roof, B/s
TFLOPS = 85.96e12   # measured bf16 compute roof


def vol(sh):
    v = 1
    for x in sh:
        v *= max(int(x), 1)
    return v


def bytes_flops(op, in0, in1, out0, dtype_bytes=2):
    """Bytes moved and FLOPs, from the shapes the report carries. 0 flops = not a matmul."""
    b = (vol(in0) + vol(out0)) * dtype_bytes
    fl = 0.0
    if "Matmul" in op:
        # in1 is logged as the second operand; the weight is K x N with K = in0[-1]
        k, n = in0[-1], out0[-1]
        b += k * n * dtype_bytes
        fl = 2.0 * vol(in0[:-1]) * k * n
    return b, fl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--calls-per-fold", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.report.read_text())
    C = d["available_worker_cores"]
    g = defaultdict(lambda: {"n": 0, "ns": 0.0})
    for o in d["ops"]:
        k = (o["op"], o["cores"], tuple(o["in0"]), tuple(o["out0"]))
        g[k]["n"] += 1
        g[k]["ns"] += o["kernel_ns"]

    groups = []
    for (op, cores, in0, out0), b in g.items():
        per = b["ns"] / b["n"]
        by, fl = bytes_flops(op, list(in0), None, list(out0))
        roof_ns = max(1e9 * by / BW, 1e9 * fl / TFLOPS if fl else 0.0)
        ideal_ns = per * cores / C
        rec_ns = max(0.0, per - max(ideal_ns, roof_ns))
        groups.append({
            "op": op, "cores": cores, "in0": list(in0), "out0": list(out0), "n": b["n"],
            "ms_per_call": b["ns"] / 1e6,
            "us_each": per / 1e3,
            "core_ideal_us": ideal_ns / 1e3,
            "roof_us": roof_ns / 1e3,
            "bytes": by, "flops": fl,
            "byte_roof_frac": (1e9 * by / BW) / per,
            "flop_roof_frac": (1e9 * fl / TFLOPS) / per if fl else 0.0,
            "idle_core_s_per_fold": per * (C - cores) / C * b["n"] * a.calls_per_fold / 1e9,
            "recoverable_s_per_fold": rec_ns * b["n"] * a.calls_per_fold / 1e9,
        })
    groups.sort(key=lambda x: -x["idle_core_s_per_fold"])
    tot_idle = sum(x["idle_core_s_per_fold"] for x in groups)
    tot_rec = sum(x["recoverable_s_per_fold"] for x in groups)
    out = {"available_worker_cores": C, "bw_roof_GBs": BW / 1e9, "tflops_roof": TFLOPS / 1e12,
           "idle_core_seconds_per_fold": tot_idle,
           "recoverable_seconds_per_fold_at_the_roof": tot_rec,
           "groups": groups}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{'op':26s}{'cores':>6s}{'n':>5s}{'us ea':>8s}{'ideal':>8s}{'roof':>8s}"
          f"{'idle s/f':>10s}{'recov s/f':>10s}  shape")
    for x in groups[:14]:
        print(f"{x['op'][:26]:26s}{x['cores']:6d}{x['n']:5d}{x['us_each']:8.2f}"
              f"{x['core_ideal_us']:8.2f}{x['roof_us']:8.2f}"
              f"{x['idle_core_s_per_fold']:10.4f}{x['recoverable_s_per_fold']:10.4f}  "
              f"{x['in0']}->{x['out0']}")
    print(f"\nidle core-seconds/fold {tot_idle:.4f}   recoverable at the roof {tot_rec:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
