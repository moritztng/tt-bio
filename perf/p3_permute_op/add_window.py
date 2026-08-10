#!/usr/bin/env python3
"""Z1 deliverable 4: re-derive the gate window on THIS grid (13x10) at THIS chunk width (C=64).

Every bound in the shipped gate `(DRAM and N>=256) or (L1 and 288<=N<=352)` was calibrated on an
11x10 grid at C=32. This sweeps (N, buffer type, C) on the open device and reports, per cell, the
throughput-mode ratio of stock `ttnn.permute(0,3,1,2)` to the kernel, the cores the work split
engages, and `torch.equal` against the stock op.

THROUGHPUT MODE, not per-call sync: `generic_op` re-uploads runtime args per call, so a per-call
synced probe gets the SIGN of this op's ratio wrong (X9: 0.804x synced against 1.095x in
throughput mode, crossover at queue depth 2). A fold enqueues thousands back to back.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn

OUT = Path(__file__).resolve().parent
NS = [256, 288, 298, 320, 352, 384]
CS = [32, 64]


def load():
    return [round(v, 2) for v in os.getloadavg()]


def thru(fn, dev, k, reps=3):
    # Warm up OUTSIDE the timer: the first call of a shape JIT-compiles its program, and a median
    # taken over runs that include it is a compile time wearing a bandwidth label.
    warm = [fn() for _ in range(2)]
    ttnn.synchronize_device(dev)
    for o in warm:
        ttnn.deallocate(o)
    best = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(k)]
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) * 1e6 / k
        for o in outs:
            ttnn.deallocate(o)
        best.append(dt)
    best.sort()
    return best[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default=str(OUT / "add_window.json"))
    a = ap.parse_args()

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    R = {"host": "qb1", "card": int(os.environ.get("TT_VISIBLE_DEVICES", "-1")),
         "wheel": "0.67.4", "grid": [g.x, g.y],
         "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
         "trimul_chunk_size_298_128": T._trimul_chunk_size(298, 128),
         "triangle_mul_memory_config_298": str(T._triangle_mul_memory_config(298).buffer_type),
         "l1_bank_bytes": int(ttnn.get_memory_view(dev, ttnn.BufferType.L1).total_bytes_per_bank),
         "k_calls_per_sync": a.k, "load_start": load(), "cells": []}
    RP.set_enabled(True)

    # --- roofs, measured on THIS card THIS pass (charter §4.1), never inherited ------------------
    roofs = {}
    xt = torch.randn(1, 298, 298, 64, dtype=torch.float32)
    for src, smc in (("L1", ttnn.L1_MEMORY_CONFIG), ("DRAM", ttnn.DRAM_MEMORY_CONFIG)):
        xs = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                             memory_config=smc)
        for dst, dmc in (("L1", ttnn.L1_MEMORY_CONFIG), ("DRAM", ttnn.DRAM_MEMORY_CONFIG)):
            k = 4 if dst == "L1" else a.k
            us = thru(lambda: ttnn.clone(xs, memory_config=dmc), dev, k)
            mb = 298 * 298 * 64 * 2 / 1e6
            roofs[f"clone_{src}_to_{dst}"] = {"us": round(us, 2),
                                              "gb_s_one_way": round(mb / us * 1e3, 1)}
        ttnn.deallocate(xs)
    R["roofs_this_card_this_pass"] = roofs
    print("roofs:", roofs, flush=True)

    for bt, mc in (("L1", ttnn.L1_MEMORY_CONFIG), ("DRAM", ttnn.DRAM_MEMORY_CONFIG)):
        for C in CS:
            for N in NS:
                t = torch.randn(1, N, N, C, dtype=torch.float32)
                try:
                    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                        device=dev, memory_config=mc)
                except Exception as e:                                # noqa: BLE001
                    R["cells"].append({"buffer": bt, "C": C, "N": N,
                                       "error": repr(e)[:140]})
                    continue
                row = {"buffer": bt, "C": C, "N": N,
                       "mb_one_way": round(N * N * C * 2 / 1e6, 2)}
                try:
                    ours = RP.reblock_permute(x, mc)
                    stock = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
                    row["torch_equal"] = bool(torch.equal(ttnn.to_torch(ours), ttnn.to_torch(stock)))
                    entry = RP._prepare(x, ours, dev)
                    cores = sum((cr.end.x - cr.start.x + 1) * (cr.end.y - cr.start.y + 1)
                                for cr in entry["core_grid"].ranges())
                    Nt = (N + 31) // 32
                    row["cores_engaged"] = cores
                    row["groups"] = Nt * Nt
                    ttnn.deallocate(ours); ttnn.deallocate(stock)
                    k = 4 if bt == "L1" else a.k
                    row["k_calls_per_sync"] = k
                    row["kernel_us"] = round(thru(lambda: RP.reblock_permute(x, mc), dev, k), 2)
                    row["stock_us"] = round(
                        thru(lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=mc), dev, k), 2)
                    row["ratio_stock_over_kernel"] = round(row["stock_us"] / row["kernel_us"], 4)
                    row["wins"] = row["ratio_stock_over_kernel"] > 1.0
                    row["shipped_gate_eligible"] = bool(
                        (bt == "DRAM" and N >= 256) or (bt == "L1" and 288 <= N <= 352))
                except Exception as e:                                # noqa: BLE001
                    row["error"] = repr(e)[:200]
                ttnn.deallocate(x)
                row["load"] = load()
                R["cells"].append(row)
                print(row, flush=True)

    # --- co-residency: the three L1 consumers live at once, with the instrument validated -------
    def bank():
        v = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
        return {k: int(getattr(v, k)) for k in
                ("total_bytes_per_bank", "total_bytes_allocated_per_bank",
                 "total_bytes_free_per_bank", "largest_contiguous_bytes_free_per_bank")
                if hasattr(v, k)}

    co = {"idle": bank()}
    # instrument validation: one known L1 tensor must move the allocated figure
    probe = ttnn.from_torch(torch.zeros(1, 298, 320, 256, dtype=torch.float32),
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
    co["one_pair_tensor_48_82_MB"] = bank()
    probe2 = ttnn.from_torch(torch.zeros(1, 298, 320, 256, dtype=torch.float32),
                             layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                             memory_config=ttnn.L1_MEMORY_CONFIG)
    co["two_pair_tensors_x7_merged_path"] = bank()
    chunk = ttnn.from_torch(torch.randn(1, 298, 298, 64, dtype=torch.float32),
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
    co["plus_the_trimul_chunk"] = bank()
    try:
        moved = RP.reblock_permute(chunk, ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(dev)
        co["kernel_ran_with_both_resident"] = True
        co["plus_kernel_output_and_cbs"] = bank()
        ttnn.deallocate(moved)
    except Exception as e:                                            # noqa: BLE001
        co["kernel_ran_with_both_resident"] = False
        co["throw"] = repr(e)[:300]
    co["kernel_static_cb_bytes_per_core"] = (2 + 32 * 2 + 2) * 32 * 32 * 2
    for t in (probe, probe2, chunk):
        ttnn.deallocate(t)
    co["after_free"] = bank()
    R["co_residency"] = co
    print("co_residency:", co, flush=True)

    R["load_end"] = load()
    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
