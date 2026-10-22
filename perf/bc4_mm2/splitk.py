#!/usr/bin/env python3
"""bc4-bet-mm2: split-K for the deep, narrow transpose_a products at autograd._matmul.

BindCraft 2's backward issues [28672,64]^T @ [28672,224] (fp32 out) and [8192,64]^T @ [8192,224] through
`_matmul`. Their output is 2x7 tiles, so ttnn puts them on ~14 cores that each walk all of K: 0.607 and 0.177 ms
on a p300c against byte roofs of 0.042 and 0.012 (bench2). Splitting K into S slabs makes a batched
[S,K/S,M]^T @ [S,K/S,N] product with S times the output tiles, then an fp32 sum over S.

    TT_VISIBLE_DEVICES=2 python perf/bc4_mm2/splitk.py --out x.json
"""
import argparse
import json
import statistics
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--burst", type=int, default=20)
ap.add_argument("--reps", type=int, default=9)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import os  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T, autograd as AG  # noqa: E402

dev = T.get_device()
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
DRAM = ttnn.DRAM_MEMORY_CONFIG
cfg = AG.precise_config()
rows = []


def aiclk():
    return int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read())


def timed(f):
    ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    outs = []
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(A.burst):
        outs.append(f())
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    ms, clk = [], []
    for _ in range(A.reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
        clk.append(aiclk())
    ttnn.release_trace(dev, tid)
    for o in outs:
        ttnn.deallocate(o)
    return dict(dev_ms=round(statistics.median(ms), 5), dev_min=round(min(ms), 5), spread=round(max(ms) - min(ms), 5),
                aiclk=[min(clk), int(statistics.median(clk))])


def split(a, b, S, out_dtype, plan):
    K, M = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    a3 = ttnn.reshape(a, [S, K // S, M])
    b3 = ttnn.reshape(b, [S, K // S, N])
    pc = AG.bmm_program_config(a3, b3, True, False, dtype=ttnn.float32, compute_kernel_config=cfg) if plan else None
    if plan and pc is None:
        return None

    def f():
        p = ttnn.matmul(a3, b3, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg, program_config=pc)
        o = ttnn.sum(p, dim=0, compute_kernel_config=cfg)
        ttnn.deallocate(p)
        if out_dtype != ttnn.float32:
            o2 = ttnn.typecast(o, out_dtype)
            ttnn.deallocate(o)
            o = o2
        return o
    return f


gen = torch.Generator().manual_seed(0)
for K, M, N, od in ((28672, 64, 224, ttnn.float32), (8192, 64, 224, ttnn.bfloat16)):
    ha = torch.randn(K, M, generator=gen).bfloat16()
    hb = (torch.randn(K, N, generator=gen) / K ** 0.5).bfloat16()
    ref = ha.double().T @ hb.double()
    a = ttnn.from_torch(ha, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    b = ttnn.from_torch(hb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    arms = [("base", lambda: AG._matmul(a, b, transpose_a=True, compute_kernel_config=cfg,
                                       **({"dtype": od} if od == ttnn.float32 else {})))]
    for S in (2, 4, 7, 8, 14, 16, 28, 32, 56, 64):
        if (K // 32) % S:
            continue
        for plan in (False, True):
            f = split(a, b, S, od, plan)
            if f is not None:
                arms.append((f"S{S}{'_plan' if plan else ''}", f))
    for name, f in arms:
        r = dict(K=K, M=M, N=N, out=str(od), arm=name)
        try:
            o = f()
            got = ttnn.to_torch(o).double().reshape(M, N)
            ttnn.deallocate(o)
            r["rel_f64"] = round(float((got - ref).norm() / ref.norm()), 6)
            r.update(timed(f))
        except Exception as e:                                           # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(json.dumps(r), flush=True)
        rows.append(r)
json.dump(dict(rows=rows, loadavg=open("/proc/loadavg").read().split()[:3]), open(A.out, "w"), indent=1)
