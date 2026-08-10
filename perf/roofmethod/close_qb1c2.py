#!/usr/bin/env python3
"""Three things the 2-D sweep left open, all on qb1 card 2.

A. **The matmul writer's DRAM write roof, measured independently of the K=256 row it is used to
   score.** The sweep's best K=256 DRAM-output cell writes 197.7 GB/s, which is 13 % ABOVE the
   174.9 GB/s T1 recorded for the same writer on this same card -- so 174.9 is not the roof, and
   confirming "K=256 DRAM-out is a write roof" against a number taken from that same cell would be
   circular. This probe pins the writer's ceiling on shapes that are write-dominated by
   construction (K=32, both inputs in L1 so nothing competes for DRAM) and never touches K=256.

B. **The L1-output cells the sweep could not build.** Every K=256 nt=64 L1-output config threw at
   program.cpp:1052, which is an L1 allocation failure, not a rate. That is exactly the shape class
   T3 reached 95.42 TFLOP/s on, so leaving it empty would under-report the K=256 L1 roof. Retried
   with M walked down until it fits.

C. **Core utilisation, measured rather than derived from the shape.** T4's `engaged_cores` is the
   org's instrument and it is used here as-is for calibration; matmul cannot be driven through its
   block-shard entry point, so the same 1/c fit is applied to a `core_grid` ladder instead, and the
   two calibration points T4 used (DRAM read saturating well below the grid, a full-grid unary
   reaching the whole grid) are re-checked on this card before any matmul row is believed.

Every timed region synchronises immediately before the clock starts and immediately before it stops.
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN
sys.path.insert(0, "perf/ledger_298")
import util_probe                                                            # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
dg = dev.compute_with_storage_grid_size()
CKC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
TILE = 32
out = {}


def timed(fn, warm=2, pipe=3, reps=5):
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


def T(shape, mc):
    return ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


CFGS = [("default", {}),
        ("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
        ("cg_13x10", {"core_grid": ttnn.CoreGrid(y=dg.y, x=dg.x)})]

# --- A. the matmul writer's DRAM write roof, on write-dominated shapes ----------------------------
print("=== A. matmul-writer DRAM write roof (K=32, inputs in L1, output in DRAM) ===", flush=True)
rowsA = []
for N in (256, 1024, 2048, 4096):
    K = 32
    M = min(65536, int(50e6 / (2 * N)) // TILE * TILE)
    try:
        a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
    except Exception as e:                                                   # noqa: BLE001
        print(f"  N={N} alloc ERR {str(e)[:60]}", flush=True)
        continue
    ob = M * N * 2
    for lbl, kw in CFGS:
        try:
            s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                          memory_config=DRAM, **kw)))
        except Exception as e:                                               # noqa: BLE001
            print(f"  M={M} N={N} nt={N//TILE} {lbl:9s} ERR {str(e)[:55]}", flush=True)
            continue
        gbs = ob / s / 1e9
        rowsA.append({"M": M, "N": N, "nt": N // TILE, "cfg": lbl, "us": round(s * 1e6, 2),
                      "out_MB": round(ob / 1e6, 2), "write_GBs": round(gbs, 1)})
        print(f"  M={M:<6} N={N:<5} nt={N//TILE:<3} {lbl:9s} {s*1e6:9.2f} us  {gbs:6.1f} GB/s",
              flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
wr = max((r["write_GBs"] for r in rowsA), default=0.0)
out["matmul_write_roof"] = {"runs": rowsA, "roof_GBs": wr}
print(f"MATMUL_WRITER_WRITE_ROOF {wr} GB/s  -> implied K=256 DRAM-out ceiling "
      f"{wr * 256 / 1e3:.2f} TFLOP/s\n", flush=True)

# --- B. the K=256 L1-output cells the sweep could not allocate ------------------------------------
print("=== B. K=256 L1-output, M walked down until the config builds ===", flush=True)
rowsB = []
for K in (256, 384):
    for nt in (32, 64):
        N = nt * TILE
        for M in (12288, 8192, 6144, 4096, 2048):
            try:
                a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
            except Exception as e:                                           # noqa: BLE001
                print(f"  K={K} nt={nt} M={M} alloc ERR {str(e)[:50]}", flush=True)
                continue
            gflop = 2 * M * K * N / 1e9
            got = False
            for lbl, kw in CFGS:
                try:
                    s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                                  memory_config=L1, **kw)))
                except Exception as e:                                       # noqa: BLE001
                    print(f"  K={K} nt={nt} M={M:<6} {lbl:9s} ERR {str(e)[:50]}", flush=True)
                    continue
                tf = gflop / s / 1e3
                rowsB.append({"K": K, "nt": nt, "M": M, "cfg": lbl, "us": round(s * 1e6, 2),
                              "tflops": round(tf, 2)})
                print(f"  K={K} nt={nt} M={M:<6} {lbl:9s} {s*1e6:9.2f} us {tf:8.2f} TFLOP/s",
                      flush=True)
                got = got or lbl != "default"
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            if got:
                break
out["k256_L1_retry"] = rowsB
for K in (256, 384):
    for nt in (32, 64):
        b = max((r["tflops"] for r in rowsB if r["K"] == K and r["nt"] == nt), default=0.0)
        print(f"  >>> K={K} nt={nt} L1-output best {b:.2f} TFLOP/s", flush=True)
print("", flush=True)

# --- C. core utilisation --------------------------------------------------------------------------
print("=== C.1 calibrating T4's engaged_cores on this card ===", flush=True)
cal = {}
xa = T((1280, 1408), L1)
try:
    r = util_probe.engaged_cores(
        dev, lambda mc: (lambda: ttnn.deallocate(ttnn.mul(ttnn.to_memory_config(xa, mc), 1.0001,
                                                          memory_config=mc))), (1280, 1408))
    cal["unary_full_grid"] = r
    print(f"  unary mul 1280x1408: engaged {r['engaged']} of {r['max_grid_cores']} "
          f"floor_limited={r['floor_limited']}", flush=True)
except Exception as e:                                                       # noqa: BLE001
    print("  unary calibration ERR " + str(e)[:120], flush=True)
ttnn.deallocate(xa)
out["calibration"] = cal


def fit_engaged(times):
    """T4's fit, verbatim in rule: engaged is the largest core count at which t still falls as 1/c."""
    ok = {c: t for c, t in times.items() if t}
    if not ok:
        return None, None
    counts = sorted(ok)
    engaged = counts[0]
    for lo, hi in zip(counts, counts[1:]):
        ideal = ok[lo] * lo / hi
        gained = (ok[lo] - ok[hi]) / max(ok[lo] - ideal, 1e-9)
        if gained >= 0.15:
            engaged = hi
        else:
            break
    return engaged, min(ok.values()) < util_probe.MIN_USABLE_US


print("\n=== C.2 matmul core_grid ladder (same 1/c fit, applied to core_grid) ===", flush=True)
LADDER = [(1, 1), (2, 2), (4, 4), (4, 8), (8, 8), (8, 10), (10, 11), (10, 13)]
REGIMES = [("K256_nt8_oDRAM", 256, 8, DRAM, 16384),
           ("K256_nt32_oL1", 256, 32, L1, 16384),
           ("K1024_nt32_oL1", 1024, 32, L1, 14336),
           ("K4096_nt64_oL1", 4096, 64, L1, 4608)]
rowsC = {}
for name, K, nt, omem, M in REGIMES:
    N = nt * TILE
    try:
        a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
    except Exception as e:                                                   # noqa: BLE001
        print(f"  {name} alloc ERR {str(e)[:60]}", flush=True)
        continue
    gflop = 2 * M * K * N / 1e9
    times = {}
    for gy, gx in LADDER:
        try:
            s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                          memory_config=omem,
                                                          core_grid=ttnn.CoreGrid(y=gy, x=gx))))
            times[gy * gx] = s * 1e6
        except Exception as e:                                               # noqa: BLE001
            times[gy * gx] = None
            print(f"    {name} {gy}x{gx} ERR {str(e)[:50]}", flush=True)
    eng, floor = fit_engaged(times)
    best = min((t for t in times.values() if t), default=None)
    rowsC[name] = {"K": K, "nt": nt, "M": M, "times_us": {k: (round(v, 2) if v else None)
                                                          for k, v in times.items()},
                   "engaged": eng, "floor_limited": floor,
                   "best_us": round(best, 2) if best else None,
                   "best_tflops": round(gflop / (best / 1e6) / 1e3, 2) if best else None}
    print(f"  {name}: engaged {eng} of 130, floor_limited={floor}, best "
          f"{rowsC[name]['best_tflops']} TFLOP/s", flush=True)
    print("    " + json.dumps(rowsC[name]["times_us"]), flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
out["core_ladder"] = rowsC

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("wrote " + sys.argv[1], flush=True)
