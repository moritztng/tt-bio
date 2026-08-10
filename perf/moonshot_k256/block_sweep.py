#!/usr/bin/env python3
"""Do the blocking knobs a hand-written kernel would control move the K=256 rate?

The leg asked "what rate can a hand-written kernel achieve at K=256". It answered "a matmul
kernel is the wrong instrument" from the fidelity and bandwidth arms, but tt-bio has never
constructed a MinimalMatmulConfig: every ttnn.experimental.minimal_matmul call in the repo
runs the op's internal default blocking, and the op's own docstring says performance is
"sensitive to block sizes and subblock shapes". Those five fields ARE the mechanisms the leg
named -- larger output tiles per core (M/N block, subblock), deeper K reuse (K_block_size) --
reachable from Python with no kernel.

Two arms, because they differ in parity:
  * M_block_size, N_block_size, subblock_h, subblock_w partition the OUTPUT. Every output
    tile's K-sum is unchanged, so these must be BIT-EXACT against the default.
  * K_block_size rechunks the contraction into partial sums accumulated through DEST /
    packer_l1_acc, so it changes accumulation order and is NOT expected to be bit-exact --
    the same parity class as _NARROW_PROJ_BW.
This script measures both rather than asserting either.

Shape is the production one: M=102400, K=256, bf16 A from DRAM, bf16 result to L1, production
compute kernel config (HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True). N=256 is what qb1's
trimul in_proj actually executes (chunk 64 x 4); N=128 is the 512 aa shape, the one class
bw_bound.py found responds to nothing.

Config list is swept TWICE, interleaved at config granularity, with the default re-measured in
each pass, so drift on a loaded host shows up as cross-pass spread instead of a fake winner.
"""
import json
import os
import statistics as st
import subprocess
import sys
import time

import torch
import ttnn
from tt_bio.tenstorrent import get_device

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
MM = ttnn.experimental.minimal_matmul
M, K = 102400, 256
M_PAR = 8192          # parity check runs small; timing runs the real M


def timed(dev, fn, warm=3, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return st.median(out), (max(out) - min(out)) / st.median(out)


def loadavg():
    return round(os.getloadavg()[0], 2)


dev = get_device()
g = dev.compute_with_storage_grid_size()
GX, GY = g.x, g.y
print(f"grid {GX}x{GY} = {GX*GY} cores   loadavg {loadavg()}", flush=True)
PROD = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True, packer_l1_acc=True)

# (M_block, K_block, N_block, subblock_h, subblock_w), all in TILES.
# fp32_dest_acc_en=True holds 4 tiles in DEST, so subblock_h*subblock_w <= 4.
# K=256 is 8 tiles; K_block 8 keeps the whole contraction in one accumulation pass.
def plan(nt):
    w = min(4, nt)
    cfgs = [
        # output-partition arm: full K, vary how much output each core owns per pass
        (1, 8, nt, 1, w), (2, 8, nt, 2, min(2, nt)), (2, 8, nt, 1, w),
        (4, 8, nt, 1, w), (4, 8, nt, 2, min(2, nt)), (4, 8, nt, 4, 1),
        (8, 8, nt, 1, w), (8, 8, nt, 2, min(2, nt)),
        (16, 8, nt, 1, w), (32, 8, nt, 1, w),
        # narrower N blocks
        (4, 8, max(nt // 2, 1), 1, min(w, max(nt // 2, 1))),
        (4, 8, max(nt // 4, 1), 1, min(w, max(nt // 4, 1))),
        # K-depth arm: rechunk the contraction (expected NOT bit-exact)
        (4, 4, nt, 1, w), (4, 2, nt, 1, w), (4, 1, nt, 1, w),
        (16, 4, nt, 1, w), (16, 1, nt, 1, w),
    ]
    seen, out = set(), []
    for c in cfgs:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


res = {"grid": [GX, GY], "load_start": loadavg(), "shapes": []}
torch.manual_seed(0)

for N in (256, 128):
    nt = N // 32
    print(f"\n=== M={M} K={K} N={N}  (bf16 A DRAM -> bf16 L1, production ckc) ===", flush=True)
    a = ttnn.from_torch(torch.randn(M, K) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(K, N) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)

    # ---- parity reference at M_PAR, from the default (no config) path
    ap = ttnn.from_torch(torch.randn(M_PAR, K) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=DRAM)
    ref = ttnn.to_torch(MM(ap, b, memory_config=L1, dtype=ttnn.bfloat16,
                           compute_kernel_config=PROD)).float()

    rows = {}
    for p in (1, 2):
        ms, aa = timed(dev, lambda: ttnn.deallocate(
            MM(a, b, memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=PROD)))
        rows.setdefault("default", []).append((ms, aa))
        print(f"  pass{p} default (no config)          {ms:8.3f} ms  "
              f"{2*M*K*N/(ms/1e3)/1e12:7.2f} TFLOP/s  A/A {aa:.2%}", flush=True)
        for c in plan(nt):
            mb, kb, nb, sh, sw = c
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb, K_block_size=kb, N_block_size=nb,
                subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY))
            try:
                ms, aa = timed(dev, lambda: ttnn.deallocate(
                    MM(a, b, memory_config=L1, dtype=ttnn.bfloat16,
                       compute_kernel_config=PROD, config=cfg)))
            except Exception as e:
                if p == 1:
                    print(f"  pass1 M{mb} K{kb} N{nb} s{sh}x{sw}  ERR {str(e)[:80]}", flush=True)
                rows.setdefault(c, []).append(None)
                continue
            rows.setdefault(c, []).append((ms, aa))
            print(f"  pass{p} M{mb:<2d} K{kb} N{nb} s{sh}x{sw}            {ms:8.3f} ms  "
                  f"{2*M*K*N/(ms/1e3)/1e12:7.2f} TFLOP/s  A/A {aa:.2%}", flush=True)

    # ---- parity, once per surviving config
    out = []
    for key, vals in rows.items():
        good = [v for v in vals if v]
        if not good:
            out.append({"config": list(key), "error": True})
            continue
        mms = [v[0] for v in good]
        med = st.median(mms)
        spread = (max(mms) - min(mms)) / med if len(mms) > 1 else 0.0
        rec = {"config": "default" if key == "default" else list(key),
               "ms": round(med, 4), "tflops": round(2 * M * K * N / (med / 1e3) / 1e12, 2),
               "cross_pass_spread": round(spread, 4),
               "aa": round(max(v[1] for v in good), 4)}
        if key != "default":
            mb, kb, nb, sh, sw = key
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY))
            try:
                got = ttnn.to_torch(MM(ap, b, memory_config=L1, dtype=ttnn.bfloat16,
                                       compute_kernel_config=PROD, config=cfg)).float()
                rec["bit_exact"] = bool(torch.equal(got, ref))
                rec["max_abs"] = float((got - ref).abs().max())
            except Exception as e:
                rec["parity_error"] = str(e)[:200]
        out.append(rec)
    out.sort(key=lambda r: -r.get("tflops", 0))
    res["shapes"].append({"M": M, "K": K, "N": N, "results": out})

    base = next(r["tflops"] for r in out if r["config"] == "default")
    print(f"\n  -- N={N} ranked (default {base} TFLOP/s) --", flush=True)
    for r in out[:8]:
        if "tflops" not in r:
            continue
        be = r.get("bit_exact")
        print(f"    {str(r['config']):26s} {r['tflops']:7.2f}  {r['tflops']/base:5.3f}x  "
              f"bit_exact={be}  max_abs={r.get('max_abs')}  drift {r['cross_pass_spread']:.2%}",
              flush=True)

    for t in (a, b, ap):
        ttnn.deallocate(t)

res["load_end"] = loadavg()
json.dump(res, open(sys.argv[1], "w"), indent=2)
print("\nwrote", sys.argv[1], " loadavg", res["load_start"], "->", res["load_end"], flush=True)
