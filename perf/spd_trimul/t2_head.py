#!/usr/bin/env python3
"""Compute headroom of RFD4's trimul tail (T2CM on the bf16 pair stream, `pair_tail_cm`), the kernel that writes the z
trimul_in's T1 normalises.

LN statistics forwarding adds a row-statistics stage to the kernel that writes z; it is free only where that kernel's
compute waits on data movement. This times fused_tail_ln_cm(stream16=True) at RFD4's shape (hidden K -> c_z N) as is,
with all data movement skipped (T2_DMABL 15: gather L1 reads, stage DRAM reads, g / z reads, writes) and with each
side skipped alone, AICLK sampled during the timed calls. dm0 - dm15 is what compute could absorb.

    python perf/spd_trimul/t2_head.py --ns 352,256 --k 256 --c 384 --out t2.jsonl
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # perf/ for clocksample
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="352,256")
ap.add_argument("--k", type=int, default=256, help="trimul hidden (x channels)")
ap.add_argument("--c", type=int, default=384, help="c_z (z channels)")
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--dm", default="0,3,12,15", help="T2_DMABL masks to time")
ap.add_argument("--abl", default="1", help="T2_ABL masks to time (1: T2's own layer_norm skipped)")
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config
import clocksample

dev = T.get_device()
ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
TT.T2CM = True
K, C = A.k, A.c
gen = torch.Generator().manual_seed(0)
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ms), 4)


for n in (int(v) for v in A.ns.split(",")):
    d = {"x": up(torch.randn(1, K, n, n, generator=gen)), "g": up(torch.randn(1, n, n, C, generator=gen)),
         "z": up(torch.randn(1, n, n, C, generator=gen)), "w": up(torch.randn(K, C, generator=gen) / K ** 0.5),
         "gamma": up(1 + 0.1 * torch.randn(K, generator=gen)), "beta": up(0.1 * torch.randn(K, generator=gen))}
    rec = {"n": n, "k": K, "c": C,
           "why": TT.t2cm_eligible(d["x"], d["gamma"], d["beta"], d["w"], d["g"], d["z"], ckc, stream16=True)}
    f = lambda: TT.fused_tail_ln_cm(d["x"], d["gamma"], d["beta"], d["w"], d["g"], d["z"], ckc, grid,  # noqa: E731
                                    stream16=True)
    if rec["why"] is None and f() is not None:
        clk = clocksample.during(period=0.5).__enter__()
        for m in (int(v) for v in A.dm.split(",") if v):
            TT.T2_DMABL = m
            rec[f"dm{m}_ms"] = timed(f, A.reps)
        TT.T2_DMABL = 0
        for m in (int(v) for v in A.abl.split(",") if v):
            TT.T2_ABL = m
            rec[f"abl{m}_ms"] = timed(f, A.reps)
        TT.T2_ABL = 0
        clk.__exit__(None, None, None)
        rec["aiclk"] = clk.summary()
    else:
        rec["err"] = "declined"
    for t in d.values():
        ttnn.deallocate(t)
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as fh:
            fh.write(line + "\n")
