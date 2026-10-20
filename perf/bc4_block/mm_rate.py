#!/usr/bin/env python3
"""Matmul rate at the Evoformers own shapes: the ceiling a fused triangle kernels inner matmul can reach.

The block rewrite is compute-bound (LEDGER "The floor is FLOPs"): its saving depends on how fast a 224-wide matmul
runs once its operands no longer come from DRAM. This times ttnn.matmul at the trimul contraction (per-channel
[N,N] @ [N,N], 128 channels) and the in-/out-projections ([N*N,128] @ [128,640|128]) with operands and output in
DRAM and in L1, HiFi2 and LoFi. Calls are issued in bursts of --burst between syncs so the ~30 us sync is amortised.

    TT_VISIBLE_DEVICES=<card> python perf/bc4_block/mm_rate.py --ns 224,288,320 --out x.jsonl
"""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="224,288,320")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--burst", type=int, default=20)
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

dev = T.get_device()
MEM = dict(dram=ttnn.DRAM_MEMORY_CONFIG, l1=ttnn.L1_MEMORY_CONFIG)
FID = dict(hifi2=ttnn.MathFidelity.HiFi2, lofi=ttnn.MathFidelity.LoFi)


def ckc(fid):
    return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=FID[fid], math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(f):
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(A.reps):
        t0 = time.perf_counter()
        for _ in range(A.burst): f()
        ttnn.synchronize_device(dev); ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
    return statistics.median(ms), (max(ms) - min(ms))


gen = torch.Generator().manual_seed(0)
recs = []
for N in map(int, A.ns.split(",")):
    shapes = dict(contraction=((1, 128, N, N), (1, 128, N, N)),
                  in_proj=((1, 1, N * N, 128), (128, 640)),
                  out_proj=((1, 1, N * N, 128), (128, 128)))
    for name, (sa, sb) in shapes.items():
        ha = torch.randn(*sa, generator=gen).bfloat16(); hb = torch.randn(*sb, generator=gen).bfloat16() / 16
        ref = (ha.double() @ hb.double())
        flop = 2 * ref.numel() * sa[-1]
        for mem in ("dram", "l1"):
            a = ttnn.from_torch(ha, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM[mem])
            b = ttnn.from_torch(hb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM[mem])
            for fid in ("hifi2", "lofi"):
                cfg = ckc(fid)
                f = lambda: ttnn.deallocate(ttnn.matmul(a, b, memory_config=MEM[mem], compute_kernel_config=cfg))
                try:
                    o = ttnn.matmul(a, b, memory_config=MEM[mem], compute_kernel_config=cfg)
                    got = ttnn.to_torch(o).double(); ttnn.deallocate(o)
                    rel = float((got - ref).norm() / ref.norm())
                    ms, spread = timed(f)
                    r = dict(n=N, op=name, mem=mem, fid=fid, ms=round(ms, 4), spread_ms=round(spread, 4),
                             tflops=round(flop / ms / 1e9, 2), rel_l2_vs_f64=round(rel, 5))
                except Exception as e:
                    r = dict(n=N, op=name, mem=mem, fid=fid, error=str(e)[:300])
                print(json.dumps(r), flush=True); recs.append(r)
            for t in (a, b): ttnn.deallocate(t)
if A.out:
    open(A.out, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
