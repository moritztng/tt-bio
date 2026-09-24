#!/usr/bin/env python3
"""Cheaper K-block-1 plans for the M = 1536 linears: every candidate against ttnn's own plan.

  auto   ttnn.linear with no plan (guard off): the speed to match, and it faults
  k1_2d  tenstorrent._k1_program_config, the guard's plan today
  1d_in0 MatmulMultiCoreReuseMultiCast1DProgramConfig, in0 multicast, N split over the grid
  1d_in1 the same with in1 multicast, M split over the grid
All k1 plans at in0_block_w = 1, each scored for faults (|err| > 0.25 x scale) and timed.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from perf.clocksample import during  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
gx, gy = T.COMPUTE_GRID_MAIN
cores = gx * gy


def sub(h, w):
    sw = max(d for d in (4, 3, 2, 1) if w % d == 0)
    sh = max(d for d in (4, 3, 2, 1) if h % d == 0 and d * sw <= 4)
    return sh, sw


def one_d(mt, nt, mcast_in0):
    pm, pn = (mt, -(-nt // cores)) if mcast_in0 else (-(-mt // cores), nt)
    sh, sw = sub(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=1, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, fuse_batch=True,
        fused_activation=None, mcast_in0=mcast_in0)


def timed(fn):
    ms = []
    for rep in range(a.reps + 1):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        if rep:
            ms.append((time.perf_counter() - t0) * 1e3)
        last = o if rep == a.reps else (ttnn.deallocate(o) or None)
    ms.sort()
    return round(ms[len(ms) // 2], 4), last


rows = []
T._DEST_CARRY_GUARD = False  # every arm below names its own plan; "auto" is ttnn's
with during() as clk:
    for m, k, n in [(1536, 384, 1536), (1536, 384, 384), (1536, 768, 3072), (1536, 768, 1536), (1536, 768, 768),
                    (1536, 1536, 768), (1536, 128, 512), (1536, 512, 128)]:
        A = torch.randn(1, 1, m, k).bfloat16()
        B = torch.randn(k, n).bfloat16()
        ref = A.double() @ B.double()
        scale = ((A.double() ** 2) @ (B.double() ** 2)).sqrt()
        x = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
        w = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
        mt, nt = m // 32, n // 32
        arms = {"auto": None, "k1_2d": T._k1_program_config(mt, nt),
                "1d_in0": one_d(mt, nt, True), "1d_in1": one_d(mt, nt, False)}
        r = {"m": m, "k": k, "n": n}
        for name, cfg in arms.items():
            try:
                t, o = timed(lambda: ttnn.linear(x, w, compute_kernel_config=ckc, program_config=cfg))
            except RuntimeError as ex:
                r[name] = str(ex).splitlines()[0][:100]
                continue
            q = (ttnn.to_torch(o).double().reshape(ref.shape) - ref).abs() / scale
            ttnn.deallocate(o)
            r[name] = {"ms": t, "fault": int((q > 0.25).sum()), "max_q": round(float(q.max()), 4)}
        rows.append(r)
        print(json.dumps(r), flush=True)
        ttnn.deallocate(x); ttnn.deallocate(w)
print(clk.line(), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"aiclk": clk.summary(), "rows": rows}, indent=1) + "\n")
