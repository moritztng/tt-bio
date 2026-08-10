#!/usr/bin/env python3
"""Every roof this leg scores against, measured on qb1 card 2 in this pass. Nothing inherited.

Three products:
  1. the directional DRAM roofs (read / unary write / copy) and the machine balance;
  2. the *matmul writer's* DRAM write roof, from a write-dominated matmul, because that is the
     writer the K-corrected identity `TFLOP/s = write_GB/s x K` is about;
  3. Deliverable 3 -- the square compute roof taken three ways. `roofs_card.py` only tries square
     NxN at the ttnn default config; T3 showed on qb2 card 1 that adding `core_grid` and an L1
     output lifts the same card 18 %. Nobody has run that here, so the qb1/qb2 gap is currently
     method-confounded. Same shapes, same config ladder, on qb1.

Every timed region synchronises immediately before the clock starts and immediately before it stops.
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
dg = dev.compute_with_storage_grid_size()
CKC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

l1_per_core = None
for probe in (lambda: dev.get_max_worker_l1_unreserved_size(),
              lambda: dev.worker_l1_size(),
              lambda: dev.l1_size_per_core()):
    try:
        l1_per_core = probe()
        break
    except Exception:                                                        # noqa: BLE001
        continue
res = {"card": {"compute_with_storage_grid_size": f"{dg.x}x{dg.y}",
                "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                "l1_unreserved_per_core_B": l1_per_core}}
print(json.dumps(res["card"]), flush=True)


def timed(fn, warm=3, pipe=4, reps=7):
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


# --- 1. directional DRAM roofs -------------------------------------------------------------------
print("\n=== directional DRAM roofs (clone) ===", flush=True)
rows = []
for mb in (8, 16, 32, 64, 128):
    nrow = int(mb * 1e6 / 2) // 4096
    nbytes = nrow * 4096 * 2
    r = {"MB": round(nbytes / 1e6, 2)}
    xd = T((nrow, 4096), DRAM)
    r["read_GBs"] = round(nbytes / timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1))) / 1e9, 1)
    r["copy_GBs"] = round(2 * nbytes / timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=DRAM))) / 1e9, 1)
    ttnn.deallocate(xd)
    xl = T((nrow, 4096), L1)
    r["write_unary_GBs"] = round(nbytes / timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM))) / 1e9, 1)
    ttnn.deallocate(xl)
    rows.append(r)
    print("  " + json.dumps(r), flush=True)
read_roof = max(r["read_GBs"] for r in rows)
copy_roof = max(r["copy_GBs"] for r in rows)
wr_unary = max(r["write_unary_GBs"] for r in rows)
res["dram"] = {"runs": rows, "read_roof_GBs": read_roof, "copy_roof_GBs": copy_roof,
               "write_unary_roof_GBs": wr_unary}
print(f"READ {read_roof} GB/s  COPY {copy_roof} GB/s  WRITE(unary) {wr_unary} GB/s", flush=True)

# --- 2. the matmul writer's write roof -----------------------------------------------------------
# Write-dominated on purpose: M=102400 K=32 N=256 is 1.68 GFLOP against 52.43 MB of output, an
# arithmetic intensity of 32 FLOP/byte, two orders below the machine balance. Whatever rate this
# reaches is the writer's, not the FPU's.
print("\n=== matmul-writer DRAM write roof ===", flush=True)
mw = {}
for (M, K, N) in ((102400, 32, 256), (65536, 32, 512), (32768, 64, 1024)):
    a, b = T((1, 1, M, K), DRAM), T((1, 1, K, N), DRAM)
    ob = M * N * 2
    for lbl, kw in (("default", {}), ("core_grid", {"core_grid": CORE_GRID_MAIN})):
        try:
            s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                          memory_config=DRAM, **kw)), warm=2, pipe=3, reps=5)
        except Exception as e:                                              # noqa: BLE001
            print(f"  {M}x{K}@{K}x{N} {lbl:9s} ERR {str(e)[:60]}", flush=True)
            continue
        gbs = ob / s / 1e9
        mw[f"{M}x{K}x{N}_{lbl}"] = {"us": round(s * 1e6, 1), "out_MB": round(ob / 1e6, 2),
                                    "write_GBs": round(gbs, 1)}
        print(f"  {M}x{K}@{K}x{N} {lbl:9s} {s*1e6:9.1f} us  {gbs:6.1f} GB/s written", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
wr_mm = max((v["write_GBs"] for v in mw.values()), default=0.0)
res["matmul_write_roof"] = {"runs": mw, "roof_GBs": wr_mm}
print(f"WRITE(matmul writer) {wr_mm} GB/s", flush=True)

# --- 3. the square compute roof, three methods (Deliverable 3) ------------------------------------
print("\n=== square compute roof: default vs core_grid vs L1 output ===", flush=True)
GRIDS = (("default", {}),
         ("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
         ("cg_13x10", {"core_grid": ttnn.CoreGrid(y=dg.y, x=dg.x)}))
sq = {}
for n in (2048, 4096, 6144):
    a, b = T((1, 1, n, n), DRAM), T((1, 1, n, n), DRAM)
    gf = 2 * n ** 3 / 1e9
    for omem_lbl, omem in (("oDRAM", DRAM), ("oL1", L1)):
        for lbl, kw in GRIDS:
            try:
                s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                              memory_config=omem, **kw)),
                          warm=2, pipe=3, reps=5)
            except Exception as e:                                          # noqa: BLE001
                print(f"  N={n:<5} {omem_lbl} {lbl:9s} ERR {str(e)[:60]}", flush=True)
                continue
            tf = gf / s / 1e3
            sq[f"{n}_{omem_lbl}_{lbl}"] = {"ms": round(s * 1e3, 4), "tflops": round(tf, 2)}
            print(f"  N={n:<5} {omem_lbl:6s} {lbl:9s} {s*1e3:8.4f} ms {tf:8.2f} TFLOP/s", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

sq_default = max((v["tflops"] for k, v in sq.items() if k.endswith("oDRAM_default")), default=0.0)
sq_best = max((v["tflops"] for v in sq.values()), default=0.0)
res["square_compute"] = {"runs": sq, "square_dram_default_TFLOPs": sq_default, "best_TFLOPs": sq_best,
                         "method_gain_pct": round(100 * (sq_best / sq_default - 1), 1) if sq_default else None}
res["machine_balance_FLOP_per_byte"] = {
    "vs_read_roof_square_default": round(sq_default * 1e12 / (read_roof * 1e9), 1),
    "vs_read_roof_best": round(sq_best * 1e12 / (read_roof * 1e9), 1)}
print(f"\nSQUARE_DRAM_DEFAULT {sq_default:.2f}   BEST_OF_LADDER {sq_best:.2f}  "
      f"(+{res['square_compute']['method_gain_pct']} %)", flush=True)
print(f"MACHINE_BALANCE {res['machine_balance_FLOP_per_byte']} FLOP/byte", flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=1)
print("wrote " + sys.argv[1], flush=True)
