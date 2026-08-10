#!/usr/bin/env python3
"""Is the K=256 trunk matmul contraction-depth limited, or is it simply DRAM-bandwidth bound?

`rate_knobs.py` found that at the real trunk shapes the achieved rate is invariant to math
fidelity above HiFi2 and to `fp32_dest_acc_en`, i.e. it is not FPU-bound. What it does track is
arithmetic intensity: for M >> K,N and bf16, AI = K*N/(K+N) FLOP/byte, and the measured rates
divided by that AI land within a narrow band of GB/s. That is the signature of a bandwidth bound.

If that is right, the rate must move when the BYTES move and only when the bytes move:
  - result to L1 instead of DRAM         -> removes the DRAM write
  - bfloat8_b result                     -> halves the write bytes
  - bfloat8_b operands and result        -> halves read and write bytes
and it must not move when only the arithmetic moves (already shown).

Roofs are re-measured in this same session so every percentage is same-card same-run.

    PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:moonshot-4x-k256-kernel-rate \
        python3 perf/moonshot_k256/bw_bound.py out.json
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
MM = ttnn.experimental.minimal_matmul
BPE = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1.0625}  # bfp8_b: 16 datums + 1 shared exponent byte


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
    return st.median(out)


dev = get_device()
g = dev.compute_with_storage_grid_size()
print(f"grid {g.x}x{g.y} = {g.x*g.y} cores", flush=True)
res = {"grid": [g.x, g.y]}
torch.manual_seed(0)
KC4 = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

# ------------------------------------------------------------------ DRAM roofs, this session
print("\n== DRAM roofs (same session) ==", flush=True)
roof = {}
for mb in (32, 64):
    rows = int(mb * 1e6 / 2) // 4096
    nb = rows * 4096 * 2
    xd = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=dev, memory_config=DRAM)
    rd = nb / (timed(dev, lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1))) / 1e3) / 1e9
    xl = ttnn.clone(xd, memory_config=L1)
    wr = nb / (timed(dev, lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM))) / 1e3) / 1e9
    roof[f"{mb}MB"] = {"read_GBs": round(rd, 1), "write_GBs": round(wr, 1)}
    print(f"  {nb/1e6:6.1f} MB  read {rd:7.1f} GB/s   write {wr:7.1f} GB/s", flush=True)
    ttnn.deallocate(xl); ttnn.deallocate(xd)
res["dram_roof"] = roof
READ_ROOF = max(v["read_GBs"] for v in roof.values())
WRITE_ROOF = max(v["write_GBs"] for v in roof.values())

# --------------------------------------------------------- trunk shapes, bytes-moving arms
SHAPES = [
    ("trimul.in_proj  @[256,128]", (1, 320, 320, 256), (256, 128)),
    ("trimul.out_proj @[256,256]", (1, 320, 320, 256), (256, 256)),
    ("triatt.qkv      @[256,768]", (320, 320, 256), (256, 768)),
    ("transition.up   @[256,1024]", (1, 30, 320, 256), (256, 1024)),
]
print("\n== trunk shapes: move the bytes, watch the rate ==", flush=True)
print("   arm                 TFLOP/s        read GB/s  write GB/s  %read %write", flush=True)
res["shapes"] = {}
for label, ash, bsh in SHAPES:
    m = 1
    for d in ash[:-1]:
        m *= d
    k, n = ash[-1], bsh[-1]
    gfl = 2 * m * k * n / 1e9
    row = {"m": m, "k": k, "n": n, "gflop": round(gfl, 2),
           "AI_bf16_flop_per_byte": round(k * n / (k + n), 1), "arms": {}}
    print(f"  {label}  M={m} K={k} N={n}  {gfl:.1f} GFLOP  AI={k*n/(k+n):.0f}", flush=True)
    for arm, adt, odt, omem in (
        ("bf16 -> DRAM", ttnn.bfloat16, ttnn.bfloat16, DRAM),
        ("bf16 -> L1", ttnn.bfloat16, ttnn.bfloat16, L1),
        ("bf16 -> bfp8 DRAM", ttnn.bfloat16, ttnn.bfloat8_b, DRAM),
        ("bfp8 -> bfp8 DRAM", ttnn.bfloat8_b, ttnn.bfloat8_b, DRAM),
        ("bfp8 -> bfp8 L1", ttnn.bfloat8_b, ttnn.bfloat8_b, L1),
    ):
        try:
            at = ttnn.from_torch(torch.randn(*ash) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=adt)
            bt = ttnn.from_torch(torch.randn(*bsh) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=adt)
            r = [gfl / (timed(dev, lambda: ttnn.deallocate(
                MM(at, bt, memory_config=omem, dtype=odt, compute_kernel_config=KC4))) / 1e3) / 1e3
                 for _ in range(2)]
        except Exception as e:
            print(f"      {arm:20s} ERR {str(e)[:60]}", flush=True)
            try:
                ttnn.deallocate(at); ttnn.deallocate(bt)
            except Exception:
                pass
            continue
        ms = gfl / (max(r) * 1e3) * 1e3
        rb = m * k * BPE[adt] / 1e9
        wb = m * n * BPE[odt] / 1e9
        rgb = rb / (ms / 1e3)
        wgb = 0.0 if omem is L1 else wb / (ms / 1e3)
        row["arms"][arm] = {"tflops": [round(x, 2) for x in r], "read_GBs": round(rgb, 1),
                            "write_GBs": round(wgb, 1)}
        print(f"      {arm:20s} {r[0]:6.2f}/{r[1]:6.2f}   {rgb:8.1f}   {wgb:8.1f}   "
              f"{rgb/READ_ROOF:5.0%} {wgb/WRITE_ROOF:5.0%}   (A/A {abs(r[0]-r[1])/max(r):.2%})", flush=True)
        ttnn.deallocate(at); ttnn.deallocate(bt)
    res["shapes"][label] = row

res["read_roof_GBs"] = READ_ROOF
res["write_roof_GBs"] = WRITE_ROOF
json.dump(res, open(sys.argv[1], "w"), indent=2)
print(f"\nroofs: read {READ_ROOF:.1f} GB/s  write {WRITE_ROOF:.1f} GB/s", flush=True)
print("wrote", sys.argv[1], flush=True)
