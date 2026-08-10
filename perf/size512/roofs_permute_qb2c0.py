#!/usr/bin/env python3
"""Roofs on qb2 chip 0, this session, plus the C2FIX pair permute placed against them.

Roofs are per-card and this org has produced four wrong conclusions from an inherited one, so
nothing here is carried in from `perf/ledger_298/roofs_c0.json` (13x10 grid, a qb1 card) or from
W6's P300c permute numbers. Everything below is measured in this process on this chip.

Four measurements, one device context:

  compute roof  square bf16 HiFi4 matmul, DRAM output -- only to fix the machine balance.
  DRAM roofs    directional: DRAM->L1 clone sees reads only, L1->DRAM clone sees writes only.
  L1 op roof    block-sharded L1->L1 eltwise, the roof an L1-resident row is scored against.
  the permute   ttnn.permute(x, (1,0,2)) at the real pair shape [N, ceil32(N), 256] bf16 for
                every N in the sweep, to DRAM and to L1, against the plain-clone copy roof at
                the same shape and size. `bytes = 2 * volume` (read + write), the convention the
                org's DRAM rows already use.

Both sides of every timed region synchronise.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import ttnn  # noqa: E402

TILE = 32
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def ceil32(v):
    return -(-v // 32) * 32


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
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def shard_cfg(rows, cols, gy, gx):
    cr = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(cr, [rows // gy, cols // gx], ttnn.ShardOrientation.ROW_MAJOR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="298,320,352,384,448,512")
    a = ap.parse_args()

    import importlib.metadata as im
    from tt_bio.tenstorrent import get_device, COMPUTE_GRID_MAIN

    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    gx, gy = COMPUTE_GRID_MAIN
    res = {"host": "qb2", "chip": 0,
           "ttnn": im.version("ttnn"),
           "compute_with_storage_grid_size": f"{dg.x}x{dg.y}",
           "core_grid_main": f"{gx}x{gy}", "cores_main": gx * gy,
           "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size())}
    print(json.dumps(res), flush=True)

    # --- compute roof, only to fix the machine balance -------------------------------------
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    comp = {}
    for n in (2048, 4096):
        x = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        y = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        s = timed(dev, lambda: ttnn.deallocate(ttnn.matmul(x, y, compute_kernel_config=ckc, memory_config=DRAM)),
                  warm=2, pipe=3, reps=3)
        comp[f"{n}_square_oDRAM"] = {"ms": round(s * 1e3, 4), "tflops": round(2 * n ** 3 / s / 1e12, 2)}
        print(f"  compute N={n} {comp[f'{n}_square_oDRAM']}", flush=True)
        ttnn.deallocate(x); ttnn.deallocate(y)
    res["compute_roof"] = {"runs": comp, "peak_TFLOPs": max(v["tflops"] for v in comp.values())}

    # --- directional DRAM roofs ------------------------------------------------------------
    rows = []
    for mb in (16, 32, 64, 128):
        nrow = int(mb * 1e6 / 2) // 4096
        nbytes = nrow * 4096 * 2
        r = {"MB": round(nbytes / 1e6, 2)}
        xd = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=DRAM)
        try:
            t = timed(dev, lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)))
            r["read_GBs"] = round(nbytes / t / 1e9, 1)
        except Exception as e:                                                  # noqa: BLE001
            r["read_err"] = str(e)[:90]
        t = timed(dev, lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=DRAM)))
        r["dram2dram_rw_GBs"] = round(2 * nbytes / t / 1e9, 1)
        ttnn.deallocate(xd)
        try:
            xl = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, memory_config=L1)
            t = timed(dev, lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)))
            r["write_GBs"] = round(nbytes / t / 1e9, 1)
            ttnn.deallocate(xl)
        except Exception as e:                                                  # noqa: BLE001
            r["write_err"] = str(e)[:90]
        rows.append(r)
        print("  dram " + json.dumps(r), flush=True)
    peak_r = max((r.get("read_GBs", 0) for r in rows), default=0.0)
    peak_w = max((r.get("write_GBs", 0) for r in rows), default=0.0)
    res["dram_roofs"] = {"runs": rows, "read_peak_GBs": peak_r, "write_peak_GBs": peak_w,
                         "dram2dram_peak_GBs": max(r.get("dram2dram_rw_GBs", 0) for r in rows)}
    res["machine_balance_FLOP_per_byte_read"] = round(res["compute_roof"]["peak_TFLOPs"] * 1e12
                                                      / (peak_r * 1e9), 1)
    print(f"  MEASURED read {peak_r} GB/s  write {peak_w} GB/s  "
          f"balance {res['machine_balance_FLOP_per_byte_read']} FLOP/byte", flush=True)

    # --- L1 op roof --------------------------------------------------------------------------
    l1runs, best = [], 0.0
    for r_, c_ in [(gy * TILE * 4, gx * TILE * 4), (gy * TILE * 8, gx * TILE * 8),
                   (gy * TILE * 16, gx * TILE * 8)]:
        n = r_ * c_
        mc = shard_cfg(r_, c_, gy, gx)
        try:
            p = ttnn.from_torch(torch.randn(r_, c_), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=mc)
            q = ttnn.from_torch(torch.randn(r_, c_), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=mc)
            o = ttnn.from_torch(torch.zeros(r_, c_), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=mc)
            s = timed(dev, lambda: ttnn.add(p, q, memory_config=mc, output_tensor=o), warm=5, pipe=20, reps=5)
            rec = {"rows": r_, "cols": c_, "MB_per_tensor": round(n * 2 / 1e6, 3), "cores": gx * gy,
                   "binary_add_us": round(s * 1e6, 2), "binary_GBs": round(3 * n * 2 / s / 1e9, 1)}
            best = max(best, rec["binary_GBs"])
            ttnn.deallocate(p); ttnn.deallocate(q); ttnn.deallocate(o)
        except Exception as e:                                                  # noqa: BLE001
            rec = {"rows": r_, "cols": c_, "error": str(e)[:110]}
        l1runs.append(rec)
        print("  l1 " + json.dumps(rec), flush=True)
    res["l1_op_roof"] = {"runs": l1runs, "l1_op_roof_GBs": round(best, 1),
                         "note": "achievable L1<->L1 eltwise op roof, not an SRAM hardware roof"}

    # --- the C2FIX permute at the real pair shape, both destinations --------------------------
    perm = []
    for N in [int(v) for v in a.sizes.split(",")]:
        S = ceil32(N)
        vol = N * S * 256
        nbytes = vol * 2
        rec = {"N": N, "shape": [N, S, 256], "MB": round(nbytes / 1e6, 2),
               "m_tiles": N * S // 32, "cores_if_1tile_per_core": min(N * S // 32, gx * gy),
               "grid_saturation_pct": round(100.0 * min(N * S // 32, gx * gy) / (gx * gy), 1)}
        try:
            xs = ttnn.from_torch(torch.randn(N, S, 256), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        except Exception as e:                                                  # noqa: BLE001
            rec["src_alloc_err"] = str(e)[:150]
            perm.append(rec); print("  perm " + json.dumps(rec), flush=True); continue
        for lbl, mc in (("dram", DRAM), ("l1", L1)):
            try:
                t = timed(dev, lambda: ttnn.deallocate(ttnn.permute(xs, (1, 0, 2), memory_config=mc)),
                          warm=2, pipe=3, reps=5)
                rec[f"permute_{lbl}_ms"] = round(t * 1e3, 4)
                rec[f"permute_{lbl}_GBs"] = round(2 * nbytes / t / 1e9, 1)
            except Exception as e:                                              # noqa: BLE001
                rec[f"permute_{lbl}_err"] = str(e)[:150]
            try:
                t = timed(dev, lambda: ttnn.deallocate(ttnn.clone(xs, memory_config=mc)),
                          warm=2, pipe=3, reps=5)
                rec[f"clone_{lbl}_ms"] = round(t * 1e3, 4)
                rec[f"clone_{lbl}_GBs"] = round(2 * nbytes / t / 1e9, 1)
            except Exception as e:                                              # noqa: BLE001
                rec[f"clone_{lbl}_err"] = str(e)[:150]
        ttnn.deallocate(xs)
        for k in ("dram", "l1"):
            if f"permute_{k}_GBs" in rec and f"clone_{k}_GBs" in rec:
                rec[f"pct_of_clone_roof_{k}"] = round(100.0 * rec[f"permute_{k}_GBs"]
                                                      / rec[f"clone_{k}_GBs"], 1)
            if f"permute_{k}_GBs" in rec and peak_r:
                rec[f"pct_of_dram_rw_roof_{k}"] = round(
                    100.0 * rec[f"permute_{k}_GBs"] / res["dram_roofs"]["dram2dram_peak_GBs"], 1)
        if "permute_dram_ms" in rec and "permute_l1_ms" in rec:
            rec["l1_speedup"] = round(rec["permute_dram_ms"] / rec["permute_l1_ms"], 3)
        perm.append(rec)
        print("  perm " + json.dumps(rec), flush=True)
    res["permute"] = perm

    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
