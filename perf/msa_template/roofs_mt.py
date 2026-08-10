#!/usr/bin/env python3
"""Every roof this leg scores against, measured on THIS card in ONE process.

Charter 4.1: roofs are per-card and per-kernel-config, never inherited. Charter 4.6 (amended
2026-08-10): a compute roof must put its output where the op puts its own and must be taken at that
op's output width `nt`. The MSA and template stages contract K=64 (template pair track), K=108/128
(template `a` projection, MSA c_m) and K=256 (the c_z pair projections), at output widths of 2 to 8
tiles -- none of which the square roof describes.

Four legs:
  base compute : square bf16 matmul, HiFi4, fp32_dest_acc + packer_l1_acc, DRAM and L1 output
  dram         : read swept to 128 MB (the 8-64 MB ladder still climbs and reads ~3.4 % low), write
  L1 op roof   : fastest L1->L1 bytes/s a ttnn eltwise op reaches (an achievable-op roof, not SRAM)
  K x nt sweep : batched matmul at the fold's own blocking -- batch x (320 x K) @ (K x nt*32) --
                 for every (K, nt) pair the two stages actually run, output to L1 and to DRAM

    TT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python3 perf/msa_template/roofs_mt.py \
        --out perf/msa_template/roofs_mt_pc0.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TILE = 32


def timed(dev, fn, warm=4, pipe=5, reps=7):
    """Median of `reps` runs of `pipe` back-to-back calls. Synced on both sides of every region."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--m-rows", type=int, default=320, help="padded M per batch row (298 -> 320)")
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    dg = dev.compute_with_storage_grid_size()
    res = {"device": {"compute_grid": f"{dg.x}x{dg.y}",
                      "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                      "host": "pc", "card": 0,
                      "ckc": "HiFi4 fp32_dest_acc_en=True packer_l1_acc=True"}}
    print(json.dumps(res["device"]), flush=True)

    # ---- 1. base compute roof, square, both output placements -----------------------------
    print("=== square compute roof ===", flush=True)
    comp = {}
    for n in (2048, 4096, 6144):
        a = ttnn.from_torch(torch.randn(n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        b = ttnn.from_torch(torch.randn(n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=DRAM)
        for tag, mc in (("oDRAM", DRAM), ("oL1", L1)):
            try:
                s = timed(dev, lambda: ttnn.deallocate(
                    ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=mc,
                                dtype=ttnn.bfloat16)))
                comp[f"{n}_{tag}"] = {"ms": round(s * 1e3, 4),
                                      "tflops": round(2 * n ** 3 / s / 1e12, 2)}
            except Exception as e:                                    # noqa: BLE001
                comp[f"{n}_{tag}"] = {"err": str(e)[:110]}
            print(f"  {n} {tag}: {comp[f'{n}_{tag}']}", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    ok = [v["tflops"] for v in comp.values() if "tflops" in v]
    res["compute_roof"] = {"runs": comp, "peak_TFLOPs": max(ok) if ok else None}

    # ---- 2. DRAM read / write, swept to 128 MB -------------------------------------------
    print("=== DRAM roofs ===", flush=True)
    ladder = []
    for mb in (16, 32, 64, 96, 128):
        nrow = int(mb * 1e6 / 2) // 4096
        nb = nrow * 4096 * 2
        r = {"MB": round(nb / 1e6, 2)}
        try:
            xd = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            r["read_GBs"] = round(nb / timed(
                dev, lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)), 2, 3, 5) / 1e9, 1)
            r["dram2dram_rw_GBs"] = round(2 * nb / timed(
                dev, lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=DRAM)), 2, 3, 5) / 1e9, 1)
            xl = ttnn.clone(xd, memory_config=L1)
            r["write_GBs"] = round(nb / timed(
                dev, lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)), 2, 3, 5) / 1e9, 1)
            ttnn.deallocate(xl)
            ttnn.deallocate(xd)
        except Exception as e:                                        # noqa: BLE001
            r["err"] = str(e)[:110]
        ladder.append(r)
        print("  " + json.dumps(r), flush=True)
    res["dram_roofs"] = {
        "runs": ladder,
        "read_peak_GBs": max((r.get("read_GBs", 0) for r in ladder), default=None),
        "write_peak_GBs": max((r.get("write_GBs", 0) for r in ladder), default=None)}

    # ---- 3. L1 op roof --------------------------------------------------------------------
    print("=== L1 op roof (achievable-op, not SRAM hardware) ===", flush=True)
    l1 = {}
    for mb in (2, 8, 16):
        nrow = int(mb * 1e6 / 2) // 1024
        nb = nrow * 1024 * 2
        try:
            x = ttnn.from_torch(torch.randn(nrow, 1024), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
            y = ttnn.from_torch(torch.randn(nrow, 1024), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
            sb = timed(dev, lambda: ttnn.deallocate(ttnn.add(x, y, memory_config=L1)), 3, 4, 5)
            su = timed(dev, lambda: ttnn.deallocate(ttnn.relu(x, memory_config=L1)), 3, 4, 5)
            l1[f"{round(nb/1e6,1)}MB"] = {"binary_GBs": round(3 * nb / sb / 1e9, 1),
                                          "unary_GBs": round(2 * nb / su / 1e9, 1),
                                          "binary_us": round(sb * 1e6, 1),
                                          "unary_us": round(su * 1e6, 1)}
            ttnn.deallocate(x)
            ttnn.deallocate(y)
        except Exception as e:                                        # noqa: BLE001
            l1[f"{mb}MB"] = {"err": str(e)[:110]}
        print(f"  {mb}MB: {l1[list(l1)[-1]]}", flush=True)
    bin_pk = max((v.get("binary_GBs", 0) for v in l1.values()), default=0)
    un_pk = max((v.get("unary_GBs", 0) for v in l1.values()), default=0)
    res["l1_roof"] = {"runs": l1, "binary_peak_GBs": bin_pk, "unary_peak_GBs": un_pk,
                      "l1_op_roof_GBs": max(bin_pk, un_pk),
                      "note": "achievable-op roof for L1<->L1 traffic, not an SRAM hardware roof"}

    # ---- 4. K x nt sweep at the fold's own blocking ----------------------------------------
    # The stages' matmuls are BATCHED: (batch, M=320, K) @ (K, N). A flattened-M standin is a
    # different op (perfwar-flattened-m-standin). batch is chosen so every leg moves a comparable
    # number of bytes and the region is long enough to time.
    print("=== K x nt roof, batched (batch,320,K) @ (K,nt*32) ===", flush=True)
    M = args.m_rows
    sweep = {}
    for K in (64, 128, 256):
        for ntile in (2, 4, 8, 16, 32):
            N = ntile * TILE
            batch = 298
            key = f"K{K}_nt{ntile}"
            try:
                a = ttnn.from_torch(torch.randn(1, batch, M, K) * 0.1, dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
                w = ttnn.from_torch(torch.randn(K, N) * 0.1, dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
                fl = 2 * batch * M * K * N
                row = {"batch": batch, "M": M, "K": K, "N": N,
                       "GFLOP": round(fl / 1e9, 3),
                       "in_MB": round(batch * M * K * 2 / 1e6, 2),
                       "out_MB": round(batch * M * N * 2 / 1e6, 2)}
                for tag, mc in (("oDRAM", DRAM), ("oL1", L1)):
                    try:
                        s = timed(dev, lambda: ttnn.deallocate(
                            ttnn.linear(a, w, compute_kernel_config=ckc, memory_config=mc,
                                        dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)), 2, 3, 5)
                        row[tag] = {"us": round(s * 1e6, 1),
                                    "tflops": round(fl / s / 1e12, 2),
                                    "read_GBs": round(batch * M * K * 2 / s / 1e9, 1),
                                    "write_GBs": round(batch * M * N * 2 / s / 1e9, 1)}
                    except Exception as e:                            # noqa: BLE001
                        row[tag] = {"err": str(e)[:110]}
                sweep[key] = row
                ttnn.deallocate(a)
                ttnn.deallocate(w)
            except Exception as e:                                    # noqa: BLE001
                sweep[key] = {"err": str(e)[:110]}
            d = sweep[key].get("oDRAM", {})
            l = sweep[key].get("oL1", {})
            print(f"  {key}: oDRAM {d.get('tflops')} TF/s {d.get('us')} us | "
                  f"oL1 {l.get('tflops')} TF/s {l.get('us')} us", flush=True)
    res["k_nt_sweep"] = sweep

    rd = res["dram_roofs"]["read_peak_GBs"]
    if res["compute_roof"]["peak_TFLOPs"] and rd:
        res["machine_balance_FLOP_per_byte_read"] = round(
            res["compute_roof"]["peak_TFLOPs"] * 1e12 / (rd * 1e9), 1)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nmachine balance = {res.get('machine_balance_FLOP_per_byte_read')} FLOP/byte", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
