#!/usr/bin/env python3
"""H5 follow-up: what P7 actually fails on, and whether the region's win moves under L1 pressure.

Two sections, one process, qb2 chip 0.

A. P7 disambiguation at [1,512,512,256]. Every alternative L1-output config threw `Out of Memory:
   Not enough space to allocate 134217728 B L1 buffer`, and the entry that threw builds `proj_ms`
   and `region_ms` in one expression, so which of the two threw is unknown. Time them separately.

B. The 512 aa c=64 region under ballast. The isolated OFF-minus-ON delta is 0.172 ms/region against
   an in-fold 2.30 ms/region. If the excess is occupancy pressure the isolated probe does not see,
   the delta grows with ballast; if it is flat until the L1 arm is refused outright, the excess is
   not in this op.
"""
import argparse
import json
import statistics as st
import time
import traceback

import torch
import ttnn

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def ceil32(v):
    return -(-v // 32) * 32


def timed(dev, fn, warm=3, pipe=2, reps=5):
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
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def manual_cfg(gx, gy, bw, obh, obw, pcm, pcn):
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    sw = max(w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
        out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
        per_core_M=pcm, per_core_N=pcn, fuse_batch=True, fused_activation=None, mcast_in0=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", default="AB")
    a = ap.parse_args()

    import importlib.metadata as im
    from tt_bio.tenstorrent import get_device, COMPUTE_GRID_MAIN
    import tt_bio.tenstorrent as tt

    dev = get_device()
    gx, gy = COMPUTE_GRID_MAIN
    ncores = gx * gy
    bank = tt._l1_bank_bytes()
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    res = {"host": "qb2", "chip": 0, "ttnn": im.version("ttnn"), "grid": [gx, gy],
           "l1_bank_bytes": bank, "l1_total_bytes": bank * ncores,
           "note": "qb2 at 0.68.0: every figure is a RATIO owing a qb1/0.67.4 re-take (charter 4.8)"}

    # ---------------------------------------------------------------- A: P7 disambiguation
    if "A" in a.sections:
        N, c = 512, 256
        m_tiles = N * (ceil32(N) // 32)
        per_core_M = -(-(-(-m_tiles // ncores)) // 5) * 5
        n_tiles = k_tiles = ceil32(c) // 32
        T = N * ceil32(N) * ceil32(c) * 2
        A = {"shape": [1, N, N, c], "tensor_bytes": T, "per_core_M": per_core_M,
             "two_live_L1_bytes": 2 * T, "l1_total_bytes": bank * ncores, "cells": {}}

        x = ttnn.from_torch(torch.randn(1, N, N, c, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        x2 = ttnn.from_torch(torch.randn(1, N, N, c, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        wp = ttnn.from_torch(torch.randn(c, c, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        wg = ttnn.from_torch(torch.randn(c, c, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)

        cands = {
            "bw1_obh1_obw8": (1, 1, 8),
            "bw1_obh5_obw2": (1, 5, 2),
            "production_bw8_obh5_obw8": (8, 5, 8),
        }
        for name, (bw, obh, obw) in cands.items():
            e = {"in0_block_w": bw, "out_block_h": obh, "out_block_w": obw}
            pc = manual_cfg(gx, gy, bw, obh, obw, per_core_M, n_tiles)
            # A1: one L1 output alone
            try:
                o = ttnn.linear(x, wp, memory_config=L1, dtype=ttnn.bfloat16,
                                compute_kernel_config=ckc, program_config=pc)
                ttnn.synchronize_device(dev)
                e["single_l1_output"] = "OK"
                ttnn.deallocate(o)

                def proj(pc=pc):
                    ttnn.deallocate(ttnn.linear(x, wp, memory_config=L1, dtype=ttnn.bfloat16,
                                                compute_kernel_config=ckc, program_config=pc))
                e["proj_ms"] = timed(dev, proj) * 1e3
            except Exception as ex:                                               # noqa: BLE001
                e["single_l1_output"] = "THROW: " + str(ex)[:200]
            # A2: two L1 outputs live at once, which is what the region needs
            try:
                p = ttnn.linear(x, wp, memory_config=L1, dtype=ttnn.bfloat16,
                                compute_kernel_config=ckc, program_config=pc)
                g = ttnn.linear(x2, wg, memory_config=L1, dtype=ttnn.bfloat16,
                                compute_kernel_config=ckc, program_config=pc)
                ttnn.synchronize_device(dev)
                e["two_live_l1_outputs"] = "OK"
                ttnn.deallocate(p)
                ttnn.deallocate(g)
            except Exception as ex:                                               # noqa: BLE001
                e["two_live_l1_outputs"] = "THROW: " + str(ex)[:200]
                for nm in ("p", "g"):
                    try:
                        ttnn.deallocate(locals()[nm])
                    except Exception:                                             # noqa: BLE001
                        pass
            # A3: the mixed region -- p in L1, g in DRAM
            try:
                def mixed(pc=pc):
                    p = ttnn.linear(x, wp, memory_config=L1, dtype=ttnn.bfloat16,
                                    compute_kernel_config=ckc, program_config=pc)
                    g = ttnn.linear(x2, wg, memory_config=DRAM, dtype=ttnn.bfloat16,
                                    compute_kernel_config=ckc, program_config=pc)
                    r = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                    ttnn.deallocate(g)
                    ttnn.deallocate(r)
                e["mixed_region_ms"] = timed(dev, mixed) * 1e3
            except Exception as ex:                                               # noqa: BLE001
                e["mixed_region"] = "THROW: " + str(ex)[:200]
            A["cells"][name] = e

        for t_ in (x, x2, wp, wg):
            ttnn.deallocate(t_)
        res["A_p7_disambiguation"] = A

    # ---------------------------------------------------------------- B: region under ballast
    N, c = 512, 64
    m_tiles = N * (ceil32(N) // 32)
    per_core_M = -(-(-(-m_tiles // ncores)) // 5) * 5
    n_tiles = k_tiles = ceil32(c) // 32
    T = N * ceil32(N) * ceil32(c) * 2
    x = ttnn.from_torch(torch.randn(1, N, N, c, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    x2 = ttnn.from_torch(torch.randn(1, N, N, c, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    wp = ttnn.from_torch(torch.randn(c, c, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    wg = ttnn.from_torch(torch.randn(c, c, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    cfg_d = tt._pair_proj_config(x, wp, bw_cap=tt._PAIR_PROJ_BW, out_l1=False)
    cfg_l = tt._pair_proj_config(x, wp, bw_cap=tt._PAIR_PROJ_L1_BW, out_l1=True)

    def region(mc, pc):
        p = ttnn.linear(x, wp, memory_config=mc, dtype=ttnn.bfloat16,
                        compute_kernel_config=ckc, program_config=pc)
        g = ttnn.linear(x2, wg, memory_config=mc, dtype=ttnn.bfloat16,
                        compute_kernel_config=ckc, program_config=pc)
        r = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        ttnn.deallocate(r)

    B = {"shape": [1, N, N, c], "tensor_bytes": T, "two_live_L1_bytes": 2 * T,
         "l1_total_bytes": bank * ncores, "ballast": []}
    piece = 8 * 1024 * 1024                                  # 8 MB per ballast tensor
    rows = 1024
    cols = piece // 2 // rows
    ballast = []
    for target_mb in (0, 24, 48, 72, 88):
        while len(ballast) * 8 < target_mb:
            try:
                ballast.append(ttnn.from_torch(
                    torch.zeros(rows, cols, dtype=torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1))
            except Exception:                                                 # noqa: BLE001
                break
        held_mb = len(ballast) * 8
        row = {"ballast_target_MB": target_mb, "ballast_held_MB": held_mb,
               "free_L1_MB": (bank * ncores) / 1e6 - held_mb}
        for tag, mc, pc in (("dram", DRAM, cfg_d), ("l1", L1, cfg_l)):
            try:
                row[f"region_{tag}_ms"] = timed(dev, lambda mc=mc, pc=pc: region(mc, pc)) * 1e3
            except Exception as ex:                                           # noqa: BLE001
                row[f"region_{tag}"] = "THROW: " + str(ex)[:160]
        if "region_dram_ms" in row and "region_l1_ms" in row:
            row["delta_ms_per_region"] = row["region_dram_ms"] - row["region_l1_ms"]
        B["ballast"].append(row)
        print("ballast", row, flush=True)
    for t_ in ballast:
        try:
            ttnn.deallocate(t_)
        except Exception:                                                     # noqa: BLE001
            pass
    res["B_region_under_ballast"] = B

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
