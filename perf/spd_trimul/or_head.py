#!/usr/bin/env python3
"""Compute headroom of RFD4's outer_residual (the pair write that produces the z trimul_out's T1 normalises).

LN statistics forwarding moves T1's row mean / rstd into the kernel that writes z. It only pays if that kernel has idle
compute: this times outer_residual on a bf16 z (the pair_bf16z stream) with its data movement on (DM 0) and with reads
and writes skipped (DM 7, outer_residual.DM bits 1 z reads, 2 pj rows, 4 writes), AICLK sampled during the timed calls.
DM 0 minus DM 7 is the time compute could absorb before the program gets longer.

    python perf/spd_trimul/or_head.py --ns 352,256 --c 384 --out or.jsonl
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # perf/ for clocksample
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="352,256")
ap.add_argument("--c", type=int, default=384)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--dm", default="0,1,4,7")
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, outer_residual as OR
import clocksample

dev = T.get_device()


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


gen = torch.Generator().manual_seed(0)
for n in (int(v) for v in A.ns.split(",")):
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    z = put(torch.randn(1, n, n, A.c, generator=gen))
    pi = put(torch.randn(1, n, A.c, generator=gen))
    pj = put(torch.randn(1, n, A.c, generator=gen))
    rec = {"n": n, "c": A.c, "bytes_mb": round(2 * 2 * n * n * A.c / 1e6, 1)}
    clk = clocksample.during(period=0.5).__enter__()
    for dm in (int(v) for v in A.dm.split(",")):
        OR.DM = dm
        outs = []

        def f():
            r = OR.outer_residual(z, pi, pj)
            assert r is not None, "outer_residual declined"
            outs.append(r[0])
            while len(outs) > 1:
                ttnn.deallocate(outs.pop(0))
        rec[f"dm{dm}_ms"] = timed(f, A.reps)
        for t in outs:
            ttnn.deallocate(t)
    OR.DM = 0
    clk.__exit__(None, None, None)
    rec["aiclk"] = clk.summary()
    if "dm0_ms" in rec:
        rec["GBps"] = round(rec["bytes_mb"] / rec["dm0_ms"], 1)
    for t in (z, pi, pj):
        ttnn.deallocate(t)
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as fh:
            fh.write(line + "\n")
