#!/usr/bin/env python3
"""z-permute-bands: the five crash bands on qb1's own grid, the fix, and the parity at the edges.

Everything here is measured on qb1 at ttnn 0.67.4 through `tt_bio.tenstorrent.get_device()`, so the
production grid rebind and the card lease both apply and `_transpose_memory_config` is read on the
grid the fold actually runs. `y-permute-crossmodel` found the bands on qb2 at 0.68.0 and every one of
its figures owes a qb1 re-take; sections A and B are that re-take.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-permute-bands \
        python3 perf/p3_permute_op/bands_probe.py
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

from tt_bio import reblock_permute as RP


def crs(gx, gy):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])


def throw_text(fn):
    """Verbatim first line of the wheel's exception, or None if it did not throw."""
    try:
        fn()
        return None
    except Exception as e:                                                     # noqa: BLE001
        return str(e).strip().splitlines()[0][:240]


def timeit(device, fn, reps=15, warmup=3):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def free_l1(device):
    try:
        v = ttnn.get_memory_view(device, ttnn.BufferType.L1)
        return {"largest_contiguous_bytes_free_per_bank": int(v.largest_contiguous_bytes_free_per_bank),
                "total_bytes_free_per_bank": int(v.total_bytes_free_per_bank),
                "total_bytes_per_bank": int(v.total_bytes_per_bank)}
    except Exception as e:                                                     # noqa: BLE001
        return {"error": repr(e)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "bands_probe.json"))
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    R = {"host": "qb1", "card": 3, "wheel": "0.67.4",
         "grid": [g.x, g.y], "cores": g.x * g.y,
         "compute_grid_main": list(T.COMPUTE_GRID_MAIN)}
    print("grid:", R["grid"], flush=True)

    # ---- A. the bare utility on this card's own grid, and on 11x10 and 7x10 -----------------------
    R["A_bare_utility"] = {}
    for (gx, gy) in [(g.x, g.y), (11, 10), (7, 10)]:
        cs = crs(gx, gy)
        rows = []
        for Nt in range(1, 65):
            t = throw_text(lambda cs=cs, Nt=Nt: ttnn.split_work_to_cores(cs, Nt * Nt))
            if t is not None:
                rows.append({"Nt": Nt, "units": Nt * Nt, "N_lo": (Nt - 1) * 32 + 1, "N_hi": Nt * 32,
                             "throw": t})
        R["A_bare_utility"][f"{gx}x{gy}"] = {
            "cores": gx * gy, "n_failing_Nt_of_64": len(rows), "failing": rows}
        print(f"A {gx}x{gy}: {len(rows)} failing Nt", [r['Nt'] for r in rows], flush=True)

    # ---- B. the closed form, and what _split_plan recovers ----------------------------------------
    R["B_plan"] = {}
    for Nt in [8, 10, 19, 20, 21, 29, 30, 31, 39, 40, 41, 49, 50, 51, 59, 60, 61, 64]:
        p = RP._split_plan(dev, Nt * Nt)
        R["B_plan"][str(Nt)] = None if p is None else {
            "sub_grid": [p[0], p[1]], "cores": p[0] * p[1],
            "pct_of_full_grid": round(100 * p[0] * p[1] / (g.x * g.y), 1),
            "hole_on_full_grid": RP._split_hole(g.x * g.y, g.y, Nt * Nt)}
    print("B:", {k: (v and v["cores"]) for k, v in R["B_plan"].items()}, flush=True)

    # ---- C. the pre-fix crash, reproduced through _build on this card -----------------------------
    # The production path with the fix removed: full grid only, no fall-through. Nt=20 <-> N=640.
    real_plan = RP._split_plan

    def legacy_plan(device, units):
        gg = device.compute_with_storage_grid_size()
        return (gg.x, gg.y, ttnn.split_work_to_cores(crs(gg.x, gg.y), units))

    RP._split_plan = legacy_plan
    RP._SPLIT_CACHE.clear()
    x640 = ttnn.from_torch(torch.zeros(1, 640, 640, 32, dtype=torch.bfloat16),
                           layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    try:
        RP.reblock_permute(x640, ttnn.DRAM_MEMORY_CONFIG)
        R["C_prefix_crash"] = {"reproduced": False}
    except Exception:                                                          # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        R["C_prefix_crash"] = {"reproduced": True, "Nt": 20, "N": 640,
                               "traceback_tail": tb[-6:]}
    RP._split_plan = real_plan
    RP._SPLIT_CACHE.clear()
    ttnn.deallocate(x640)
    print("C:", R["C_prefix_crash"].get("reproduced"), flush=True)

    # ---- D. parity: production shape, every band interior, every band edge -----------------------
    # C=64 is the production channel width on a 13x10 grid; the big N use C=32 to halve the bytes.
    shapes = [(298, 64, "dram"), (298, 64, "l1"), (320, 64, "l1"), (288, 64, "l1"), (352, 64, "l1"),
              (287, 64, "dram"), (353, 64, "dram")]
    for Nt in (20, 30, 40, 50, 60):
        lo, hi = (Nt - 1) * 32 + 1, Nt * 32
        C = 64 if Nt <= 20 else 32
        shapes += [(lo - 1, C, "dram"), (lo, C, "dram"), (hi, C, "dram"), (hi + 1, C, "dram")]

    MC = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}
    R["D_parity"] = []
    for (N, C, mck) in shapes:
        mc = MC[mck]
        t = torch.randn(1, N, N, C, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        RP.set_enabled(True)
        elig = RP.eligible(x, mc)
        n0 = list(RP.STATS)
        row = {"N": N, "C": C, "out": mck, "Nt": (N + 31) // 32, "eligible": bool(elig)}
        p = RP._split_plan(dev, ((N + 31) // 32) ** 2)
        row["plan_cores"] = None if p is None else p[0] * p[1]
        ref = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
        ref_t = ttnn.to_torch(ref)
        if elig:
            y = RP.reblock_permute(x, mc)
            row["torch_equal"] = bool(torch.equal(ttnn.to_torch(y), ref_t))
            ttnn.deallocate(y)
        else:
            row["torch_equal"] = None
        row["served_delta"], row["fell_through_delta"] = RP.STATS[0] - n0[0], RP.STATS[1] - n0[1]
        RP.set_enabled(False)
        ttnn.deallocate(ref); ttnn.deallocate(x)
        R["D_parity"].append(row)
        print("D:", row, flush=True)

    # ---- E. the fall-through arm is live, not dead code -------------------------------------------
    RP._split_plan = lambda device, units: None
    RP.set_enabled(True)
    n0 = list(RP.STATS)
    t = torch.randn(1, 320, 320, 64, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = T._channel_move(x, ttnn.L1_MEMORY_CONFIG)
    ref = ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    R["E_fallthrough"] = {
        "eligible_after_forced_none": RP.eligible(x, ttnn.L1_MEMORY_CONFIG),
        "served_delta": RP.STATS[0] - n0[0], "fell_through_delta": RP.STATS[1] - n0[1],
        "work_split_rejects": {f"{k[0]}:{list(k[1])}": v for k, v in RP.REJECTS.items()
                               if k[0] == "work_split"},
        "torch_equal_vs_stock": bool(torch.equal(ttnn.to_torch(y), ttnn.to_torch(ref))),
    }
    RP.set_enabled(False)
    RP._split_plan = real_plan
    ttnn.deallocate(y); ttnn.deallocate(ref); ttnn.deallocate(x)
    print("E:", R["E_fallthrough"], flush=True)

    # ---- F. the roofs, on this card, this pass ----------------------------------------------------
    # The op moves bytes and computes nothing: 0 FLOP, so its arithmetic intensity is 0 FLOP/byte and
    # it sits as far from the 338 FLOP/byte machine balance as an op can. The binding roof is the
    # copy roof at the same shape and the same buffer types, measured here.
    R["F_roofs"] = []
    for (N, C, mck) in [(298, 64, "dram"), (298, 64, "l1"), (640, 32, "dram")]:
        mc = MC[mck]
        t = torch.randn(1, N, N, C, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # bytes: tile-padded, both directions (read + write)
        Nt_ = (N + 31) // 32
        by = Nt_ * 32 * Nt_ * 32 * C * 2
        clone_us = timeit(dev, lambda: ttnn.deallocate(ttnn.clone(x, memory_config=mc)))
        stock_us = timeit(dev, lambda: ttnn.deallocate(ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)))
        RP.set_enabled(True)
        kern_us = timeit(dev, lambda: ttnn.deallocate(RP.reblock_permute(x, mc))) \
            if RP.eligible(x, mc) else None
        RP.set_enabled(False)
        p = RP._split_plan(dev, Nt_ * Nt_)
        row = {"N": N, "C": C, "out": mck, "padded_bytes_one_way": by,
               "clone_us": round(clone_us, 2), "stock_permute_us": round(stock_us, 2),
               "kernel_us": None if kern_us is None else round(kern_us, 2),
               "copy_roof_GBs_rw": round(2 * by / clone_us / 1e3, 1),
               "stock_GBs_rw": round(2 * by / stock_us / 1e3, 1),
               "kernel_GBs_rw": None if kern_us is None else round(2 * by / kern_us / 1e3, 1),
               "cores_engaged": None if p is None else p[0] * p[1],
               "work_groups": Nt_ * Nt_}
        for k in ("kernel", "stock"):
            v = row[f"{k}_GBs_rw"]
            row[f"{k}_pct_of_copy_roof"] = None if v is None else round(100 * v / row["copy_roof_GBs_rw"], 1)
        if kern_us:
            row["ratio_stock_over_kernel"] = round(stock_us / kern_us, 4)
        ttnn.deallocate(x)
        R["F_roofs"].append(row)
        print("F:", row, flush=True)

    # ---- G. the interaction checks ----------------------------------------------------------------
    tprobe = ttnn.from_torch(torch.zeros(298, 320, 256, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                             dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    R["G_interaction"] = {
        "L1_OUT_REFUSED": sorted(str(k) for k in T._L1_OUT_REFUSED),
        "transpose_memory_config_buffer_type": str(T._transpose_memory_config(tprobe).buffer_type),
        "free_l1_idle": free_l1(dev),
    }
    ttnn.deallocate(tprobe)
    # per-bank free bytes with the kernel's circular buffers live: measured inside a live call by
    # reading the view straight after a kernel launch, before the fold's own tensors are freed.
    x = ttnn.from_torch(torch.zeros(1, 298, 298, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    RP.set_enabled(True)
    y = RP.reblock_permute(x, ttnn.L1_MEMORY_CONFIG)
    R["G_interaction"]["free_l1_with_kernel_output_live"] = free_l1(dev)
    ttnn.synchronize_device(dev)
    RP.set_enabled(False)
    ttnn.deallocate(y); ttnn.deallocate(x)
    R["G_interaction"]["free_l1_after"] = free_l1(dev)
    R["stats"] = {"served": RP.STATS[0], "fell_through": RP.STATS[1],
                  "rejects": {f"{k[0]}:{list(k[1])}": v for k, v in RP.REJECTS.items()}}
    print("G:", R["G_interaction"], flush=True)
    print("stats:", R["stats"], flush=True)

    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
