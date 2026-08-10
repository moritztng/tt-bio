#!/usr/bin/env python3
"""The org's canonical compute-roof sweep for qb1, in the four dimensions that turned out to matter.

The rule this replaces ("a K=256 contraction reaches at most 35.5 TFLOP/s") was a DRAM write roof
wearing a FLOP/s label. So the sweep varies the thing that broke it and the thing T3 says may matter
more:

  output buffer type   DRAM interleaved vs L1
  program config       ttnn default vs core_grid vs an explicit 1D program config
  K                    256 (pair track, trimul contraction), 384 (single track), 1024 (fc3), 4096
  output width nt      1, 8, 32, 64 tiles  (N = 32*nt)

BOTH INPUTS ARE HELD IN L1 throughout. That is deliberate and it is what makes the row readable: the
only DRAM traffic in an `oDRAM` cell is the op's own output write, so `write_GB/s = out_bytes / t` is
exact and the identity `TFLOP/s = write_GB/s x K / 1000` can be checked cell by cell rather than
asserted. If a cell sits at the matmul writer's roof, the cell is a write roof; if it does not, it is
not. M is chosen per (K, nt) to keep A + B + out inside an L1 budget.

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

KS = (256, 384, 1024, 4096)
NTS = (1, 8, 32, 64)
L1_BUDGET = 60e6          # bytes for A + output; B is charged separately and is small
TILE = 32


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


def pick_M(K, N):
    m = int(L1_BUDGET / (2 * (K + N)))
    m = max(1024, min(16384, (m // 512) * 512))
    return m


def pc_1d(Mt, Nt, Kt, gx, gy, mcast_in0):
    """An explicit 1D reuse+mcast program config, or None if the shape cannot carry one.

    `fp32_dest_acc_en=True` halves the dest register, so out_subblock_h * out_subblock_w <= 4.
    """
    cores = gx * gy
    if mcast_in0:
        per_core_M, per_core_N = Mt, -(-Nt // cores)
    else:
        per_core_M, per_core_N = -(-Mt // cores), Nt
    ibw = max(w for w in range(1, min(Kt, 8) + 1) if Kt % w == 0)
    sub = None
    for w in range(min(per_core_N, 4), 0, -1):
        if per_core_N % w:
            continue
        for h in range(4 // w, 0, -1):
            if per_core_M % h == 0:
                sub = (h, w)
                break
        if sub:
            break
    if sub is None:
        return None
    try:
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=ibw, out_subblock_h=sub[0], out_subblock_w=sub[1],
            per_core_M=per_core_M, per_core_N=per_core_N,
            fuse_batch=True, fused_activation=None, mcast_in0=mcast_in0)
    except Exception:                                                        # noqa: BLE001
        return None


cells = []
print(f"grid={dg.x}x{dg.y} core_grid_main={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)
for K in KS:
    for nt in NTS:
        N = nt * TILE
        M = pick_M(K, N)
        Mt, Nt, Kt = M // TILE, N // TILE, K // TILE
        try:
            a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
        except Exception as e:                                               # noqa: BLE001
            print(f"K={K} nt={nt} M={M} L1 alloc ERR {str(e)[:70]}", flush=True)
            continue
        gflop = 2 * M * K * N / 1e9
        out_B = M * N * 2
        cfgs = [("default", {}),
                ("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
                ("cg_13x10", {"core_grid": ttnn.CoreGrid(y=dg.y, x=dg.x)})]
        for lbl, mci in (("pc1d_mcast_in1", False), ("pc1d_mcast_in0", True)):
            pc = pc_1d(Mt, Nt, Kt, dg.x, dg.y, mci)
            if pc is not None:
                cfgs.append((lbl, {"program_config": pc}))
        print(f"--- K={K} nt={nt} (N={N})  M={M}  {gflop:.2f} GFLOP  out {out_B/1e6:.2f} MB ---",
              flush=True)
        for omem_lbl, omem in (("L1", L1), ("DRAM", DRAM)):
            for lbl, kw in cfgs:
                try:
                    s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                                  memory_config=omem, **kw)))
                except Exception as e:                                       # noqa: BLE001
                    print(f"  out={omem_lbl:4s} {lbl:15s} ERR {str(e)[:70]}", flush=True)
                    continue
                tf = gflop / s / 1e3
                wgbs = out_B / s / 1e9
                cells.append({"K": K, "nt": nt, "N": N, "M": M, "out": omem_lbl, "cfg": lbl,
                              "us": round(s * 1e6, 2), "tflops": round(tf, 2),
                              "out_GBs": round(wgbs, 1),
                              "identity_tflops_from_write": round(wgbs * K / 1e3, 2)})
                print(f"  out={omem_lbl:4s} {lbl:15s} {s*1e6:9.2f} us {tf:8.2f} TFLOP/s   "
                      f"out {wgbs:6.1f} GB/s   (GB/s x K/1e3 = {wgbs*K/1e3:7.2f})", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

json.dump(cells, open(sys.argv[1], "w"), indent=1)
print("wrote " + sys.argv[1] + f"  ({len(cells)} cells)", flush=True)
