#!/usr/bin/env python3
"""Where the dest-carry guard's fold cost comes from, at the shapes a 1536-token fold runs.

Linears at M = 1536 (the census's biggest sites), the outer-product-mean contraction and the
triangle product, guard off (ttnn's own plan / the trimul band) against guard on (k1), timed
interleaved after a warm call, AICLK DURING. For the triangle product the k1 plan is also tried
at wider output subblocks, which the fp32 dest allows up to 4 tiles.
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
up = lambda *s: ttnn.from_torch(torch.randn(*s).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731


def timed(fn, reps):
    ms = []
    for rep in range(reps + 1):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        if rep:
            ms.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    ms.sort()
    return round(ms[len(ms) // 2], 4)


rows = []
with during() as clk:
    linears = [(1536, 384, 1536), (1536, 384, 384), (1536, 768, 3072), (1536, 768, 1536), (1536, 768, 768),
               (1536, 1536, 768), (1536, 128, 512), (1536, 512, 128)]
    for m, k, n in linears:
        x, w = up(1, 1, m, k), up(k, n)
        r = {"op": "linear", "m": m, "k": k, "n": n}
        for guard in (False, True):
            T._DEST_CARRY_GUARD = guard
            r["on" if guard else "off"] = timed(lambda: ttnn.linear(x, w, compute_kernel_config=ckc), a.reps)
        r["ratio"] = round(r["on"] / r["off"], 3)
        rows.append(r)
        print(json.dumps(r), flush=True)
        ttnn.deallocate(x); ttnn.deallocate(w)
    # outer product mean: (rows*C, S) x (D*J, S)^T, boltz2 3abq_1536 census shape
    x, w = up(2048, 7168), up(49152 // 8, 7168)  # N cut 8x to fit beside the probe; same per-core plan family
    r = {"op": "opm", "m": 2048, "k": 7168, "n": 49152 // 8}
    for guard in (False, True):
        T._DEST_CARRY_GUARD = guard
        r["on" if guard else "off"] = timed(lambda: ttnn.matmul(x, w, transpose_b=True, compute_kernel_config=ckc),
                                            a.reps)
    r["ratio"] = round(r["on"] / r["off"], 3)
    rows.append(r)
    print(json.dumps(r), flush=True)
    ttnn.deallocate(x); ttnn.deallocate(w)
    # triangle product, 32 channels as the 1536 fold issues it
    T._DEST_CARRY_GUARD = True
    kt = 48
    A, B = up(1, 32, kt * 32, kt * 32), up(1, 32, kt * 32, kt * 32)
    band = T._triangle_mul_program_config(kt)
    base = timed(lambda: ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=band, transpose_b=True,
                                     dtype=ttnn.bfloat16), a.reps)
    for sb in ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)):
        cfg = T._triangle_mul_program_config(kt, True)
        if cfg.per_core_M % sb[0] or cfg.per_core_N % sb[1]:
            continue
        cfg.out_subblock_h, cfg.out_subblock_w = sb
        t = timed(lambda: ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=cfg, transpose_b=True,
                                      dtype=ttnn.bfloat16), a.reps)
        r = {"op": "trimul", "kt": kt, "channels": 32, "band_w": band.in0_block_w, "off": base, "sb": sb,
             "on": t, "ratio": round(t / base, 3)}
        rows.append(r)
        print(json.dumps(r), flush=True)
print(clk.line(), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"aiclk": clk.summary(), "rows": rows}, indent=1) + "\n")
