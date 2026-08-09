#!/usr/bin/env python3
"""T3 probe, round 2: the EXACT shapes the 298-aa fold runs, from perf/t3/census_fold_qb2c1.json.

Round 1 built the rig at N=320, which gives the Transition a 32-row chunk (mt=320). The real fold
carries z as (1, 298, 320, 256) -- dim 1 is the LOGICAL 298, only the last two dims tile-pad -- so
ttnn.chunk(x, 10, dim=1) yields nine (1,30,320,256) chunks at mt=300 and one (1,28,320,256) at
mt=280. Every byte count for a z-shaped op is 298/320 = 0.93125 of the round-1 rig's.
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
CKC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
B2 = 2
res = {}


def timed(fn, warm=4, pipe=5, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def T(shape, mc=DRAM):
    return ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=dev, memory_config=mc)


def rec(key, s, flops, dram_bytes, all_bytes, note=""):
    e = {"us": round(s * 1e6, 2), "TFLOPs": round(flops / s / 1e12, 2) if flops else 0.0,
         "dram_GBs": round(dram_bytes / s / 1e9, 1) if dram_bytes else 0.0,
         "all_GBs": round(all_bytes / s / 1e9, 1),
         "AI_dram": round(flops / dram_bytes, 1) if dram_bytes else None,
         "dram_MB": round(dram_bytes / 1e6, 2), "GFLOP": round(flops / 1e9, 4), "note": note}
    res[key] = e
    print(f"  {key:<38} {e['us']:9.2f} us {e['TFLOPs']:7.2f} TF/s  dram {e['dram_GBs']:7.1f} GB/s "
          f" all {e['all_GBs']:7.1f} GB/s  AI_dram {e['AI_dram']}  {note}", flush=True)


print("=== the Transition at its real chunked shapes: mt=300 (x9) and mt=280 (x1) ===", flush=True)
wn, bn = T((32, 256)), T((32, 256))
w1, w3 = T((256, 1024)), T((1024, 256))
for h, mt in ((30, 300), (28, 280)):
    CH = (1, h, 320, 256)
    M, K, N = h * 320, 256, 1024
    x = T(CH, DRAM)
    tag = f"mt{mt}"
    rec(f"[{tag}] layer_norm@2038 DRAM->L1",
        timed(lambda: ttnn.deallocate(ttnn.layer_norm(x, weight=wn, bias=bn, epsilon=1e-5,
                                                     compute_kernel_config=CKC, memory_config=L1))),
        0, M * K * B2, 2 * M * K * B2, "ledger scored 0.0 s")
    rec(f"[{tag}] CONTROL clone DRAM->L1",
        timed(lambda: ttnn.deallocate(ttnn.clone(x, memory_config=L1))),
        0, M * K * B2, 2 * M * K * B2, "same bytes as the norm")
    xn = ttnn.clone(x, memory_config=L1)
    rec(f"[{tag}] fc1 activation=silu @2046",
        timed(lambda: ttnn.deallocate(ttnn.linear(xn, w1, activation="silu",
                                                  compute_kernel_config=CKC, memory_config=L1,
                                                  core_grid=CORE_GRID_MAIN))),
        2 * M * K * N, K * N * B2, (M * K + K * N + M * N) * B2, "ledger scored 0.0 s")
    rec(f"[{tag}] fc1 CONTROL no activation",
        timed(lambda: ttnn.deallocate(ttnn.linear(xn, w1, compute_kernel_config=CKC,
                                                 memory_config=L1, core_grid=CORE_GRID_MAIN))),
        2 * M * K * N, K * N * B2, (M * K + K * N + M * N) * B2, "prices the fused silu")
    rec(f"[{tag}] fc2@2055",
        timed(lambda: ttnn.deallocate(ttnn.linear(xn, w1, compute_kernel_config=CKC,
                                                 memory_config=L1, core_grid=CORE_GRID_MAIN))),
        2 * M * K * N, K * N * B2, (M * K + K * N + M * N) * B2, "ledger scored 0.0 s")
    h1, h2 = T((1, h, 320, 1024), L1), T((1, h, 320, 1024), L1)
    rec(f"[{tag}] multiply_@2064", timed(lambda: ttnn.multiply_(h1, h2)),
        M * N, 0, 3 * M * N * B2, "all L1")
    rec(f"[{tag}] CONTROL clone L1->L1",
        timed(lambda: ttnn.deallocate(ttnn.clone(h1, memory_config=L1))),
        0, 0, 2 * M * N * B2, "L1 copy roof")
    rec(f"[{tag}] fc3@2066 (W1's AI-930.9 row)",
        timed(lambda: ttnn.deallocate(ttnn.linear(h1, w3, compute_kernel_config=CKC,
                                                 memory_config=DRAM, core_grid=CORE_GRID_MAIN))),
        2 * M * N * K, (N * K + M * K) * B2, (M * N + N * K + M * K) * B2, "in L1, out DRAM")
    if mt == 300:
        print("  --- occupancy A/B at mt=300 ---", flush=True)
        for lbl, fn in (("fc1 kt8 nt32", lambda g: ttnn.linear(
                            xn, w1, compute_kernel_config=CKC, memory_config=L1, core_grid=g)),
                        ("fc3 kt32 nt8", lambda g: ttnn.linear(
                            h1, w3, compute_kernel_config=CKC, memory_config=DRAM, core_grid=g))):
            ser = []
            for gx, gy in ((4, 4), (6, 6), (8, 8), (10, 10), (11, 10)):
                try:
                    s = timed(lambda: ttnn.deallocate(fn(ttnn.CoreGrid(x=gx, y=gy))),
                              warm=3, pipe=4, reps=5)
                except Exception as e:                                  # noqa: BLE001
                    print(f"    {lbl:14s} {gx}x{gy:<3} ERR {str(e)[:44]}", flush=True)
                    continue
                ser.append({"cores": gx * gy, "us": round(s * 1e6, 2)})
                print(f"    {lbl:14s} {gx}x{gy:<4} {gx * gy:4d} cores {s * 1e6:9.2f} us", flush=True)
            res[f"occ_{lbl}_mt300"] = ser
    for t in (xn, h1, h2, x):
        ttnn.deallocate(t)

print("\n=== the z-shaped ops at the fold's real (1, 298, 320, 256) ===", flush=True)
z, z2 = T((1, 298, 320, 256)), T((1, 298, 320, 256))
ZB = 298 * 320 * 256 * B2
print(f"  z is {ZB / 1e6:.2f} MB (the N=320-padded rig said {320 * 320 * 256 * 2 / 1e6:.2f} MB)",
      flush=True)
rec("add_ x5 @2223..2239", timed(lambda: ttnn.add_(z, z2)), 298 * 320 * 256, 3 * ZB, 3 * ZB,
    "in-place, 2R+1W all DRAM")
rec("CONTROL clone DRAM->DRAM same bytes",
    timed(lambda: ttnn.deallocate(ttnn.clone(z, memory_config=DRAM))), 0, 2 * ZB, 2 * ZB,
    "W6's copy-roof control, re-measured")
wz, bz = T((32, 256)), T((32, 256))
rec("AttentionPairBias.z_norm@1893",
    timed(lambda: ttnn.deallocate(ttnn.layer_norm(z, weight=wz, bias=bz, epsilon=1e-5,
                                                  compute_kernel_config=CKC))),
    0, 2 * ZB, 2 * ZB, "1R+1W DRAM; the biggest norm in the block")
wzp = T((256, 32))
OB = 298 * 320 * 32 * B2
rec("AttentionPairBias.z_proj@1900",
    timed(lambda: ttnn.deallocate(ttnn.linear(z, wzp, compute_kernel_config=CKC,
                                              core_grid=CORE_GRID_MAIN))),
    2 * 298 * 320 * 256 * 32, ZB + 256 * 32 * B2 + OB, ZB + 256 * 32 * B2 + OB, "nt=1")
print("  --- occupancy A/B for z_proj (nt=1: does a second core column ever get work?) ---",
      flush=True)
ser = []
for gx, gy in ((1, 1), (1, 10), (2, 10), (4, 10), (8, 10), (11, 10)):
    try:
        s = timed(lambda: ttnn.deallocate(ttnn.linear(z, wzp, compute_kernel_config=CKC,
                                                      core_grid=ttnn.CoreGrid(x=gx, y=gy))),
                  warm=2, pipe=3, reps=3)
    except Exception as e:                                             # noqa: BLE001
        print(f"    z_proj {gx}x{gy:<4} ERR {str(e)[:50]}", flush=True)
        continue
    ser.append({"cores": gx * gy, "grid": f"{gx}x{gy}", "us": round(s * 1e6, 2)})
    print(f"    z_proj {gx}x{gy:<5} {gx * gy:4d} cores {s * 1e6:9.2f} us", flush=True)
res["occ_z_proj"] = ser
# the same contraction with a wide output: if nt is what starves it, nt=32 should scale
w_wide = T((256, 1024))
rec("z_proj CONTROL same M,K but nt=32",
    timed(lambda: ttnn.deallocate(ttnn.linear(z, w_wide, compute_kernel_config=CKC,
                                              core_grid=CORE_GRID_MAIN)), warm=2, pipe=3, reps=3),
    2 * 298 * 320 * 256 * 1024, ZB + 256 * 1024 * B2 + 298 * 320 * 1024 * B2,
    ZB + 256 * 1024 * B2 + 298 * 320 * 1024 * B2, "32x the FLOPs of z_proj -- how much slower?")
rec("permute@1906 (0,3,1,2) on (1,298,320,32)",
    timed(lambda: ttnn.deallocate(ttnn.permute(
        ttnn.clone(T((1, 298, 320, 32)), memory_config=DRAM), (0, 3, 1, 2))), warm=2, pipe=3, reps=3),
    0, 2 * OB, 2 * OB, "measured with its input clone included -- upper bound")

json.dump(res, open(sys.argv[1], "w"), indent=1)
print("\nwrote " + sys.argv[1], flush=True)
