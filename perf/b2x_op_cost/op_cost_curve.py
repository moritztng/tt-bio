#!/usr/bin/env python3
"""Achieved DRAM bandwidth against op size on this Blackhole, with the host removed.

This is the experiment `b2x-op-cost-curve` was opened for. It was repointed away on the finding
that the fold is dispatch-bound; `device_floor.py` shows that finding was an artifact of
tt-metal burning the calling thread's CPU while it waits for dispatch-queue room, and the fold
is 81.7 % device. So the question is live again, and the answer prices every op-granularity
lever in the codebase.

The instrument matters. Timing a small op by issuing it in a loop measures the HOST, because a
64 KB add is ~0.5 us of device work against ~6 us of issue. So every point here is a ttnn trace
containing R repetitions of the op, captured once and replayed back-to-back: the host pays about
1 us per replay of the whole trace, so what is measured is device time per op and nothing else.

Reads off the curve:
  t_fixed   the per-op device-side fixed cost, from a least-squares fit of
            t(bytes) = t_fixed + bytes / BW_eff over the small end.
  BW_eff    the large-size asymptote.
  knee      the size above which the curve is within 90 % of its asymptote.
  15.5 MB   the pairformer block's mean per-op size (6.651 GB over 428 ops).

Then the cost model is checked against the real block: does t_fixed * 428 + 6.651 GB / BW_eff
reproduce the 41.4152 ms/call that `device_floor.py` measured by replaying the shipped block?
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

STREAM_ROOF = 429.9e9
BLOCK_BYTES = 6.651e9          # corrected pairformer block traffic, b2x-diffusion-layer-bytes
BLOCK_OPS = 428
BLOCK_DEVICE_MS = 41.4152      # device_floor.py, trace replay of the shipped block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-mb", type=float, default=256.0)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device(trace_region_size=1 << 30)
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "grid": str(dev.compute_with_storage_grid_size()),
                   "arch": T.arch_name(), "loadavg": open("/proc/loadavg").read().split()[:3],
                   "stream_roof_GBs": STREAM_ROOF / 1e9},
           "points": []}
    a.out.write_text(json.dumps(out, indent=1))

    # tile-aligned square-ish shapes whose bf16 size doubles each step
    sizes = []
    tiles = 1
    while True:
        # rows x cols in tiles, keep it 2D and tile aligned
        n_tiles = tiles
        rows = 1
        cols = n_tiles
        while cols > 2 * rows:
            rows *= 2
            cols = max(1, n_tiles // rows)
        el = rows * 32 * cols * 32
        mb = el * 2 / 1e6
        if mb > a.max_mb:
            break
        if mb >= 0.06:
            sizes.append((rows * 32, cols * 32, mb))
        tiles *= 2

    for h, w, mb in sizes:
        nbytes = h * w * 2
        try:
            x = ttnn.from_torch(torch.randn(h, w), layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y = ttnn.from_torch(torch.randn(h, w), layout=ttnn.TILE_LAYOUT,
                               dtype=ttnn.bfloat16, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except Exception as e:                                              # noqa: BLE001
            out["points"].append({"MB": mb, "error": f"alloc {type(e).__name__}: {e}"})
            a.out.write_text(json.dumps(out, indent=1))
            continue
        rec = {"h": h, "w": w, "MB": round(mb, 4), "bytes_per_operand": nbytes}
        for opname, op, n_moved in (("add", ttnn.add, 3), ("clone", ttnn.clone, 2)):
            if opname == "clone" and out.get("clone_unsupported"):
                continue
            try:
                args = (x, y) if opname == "add" else (x,)
                z = op(*args)
                ttnn.deallocate(z)
                t_est = max(n_moved * nbytes / STREAM_ROOF, 1e-6)
                R = int(min(400, max(4, round(0.04 / t_est))))
                for _ in range(3):
                    z = op(*args)
                    ttnn.deallocate(z)
                ttnn.synchronize_device(dev)
                # the output buffer has to be stable inside the trace, so preallocate it
                z = op(*args)
                ttnn.synchronize_device(dev)
                op(*args, output_tensor=z)              # probe the preallocated-output path
                ttnn.synchronize_device(dev)
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                for _ in range(R):
                    op(*args, output_tensor=z)
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                ttnn.synchronize_device(dev)
                for _ in range(2):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                per = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(dev)
                    per.append((time.perf_counter() - t0) / R)
                ttnn.release_trace(dev, tid)
                ttnn.deallocate(z)
                t = st.median(per)
                rec[opname] = {"R": R, "us_per_op": round(1e6 * t, 4),
                               "all_us": [round(1e6 * p, 4) for p in per],
                               "moved_MB": round(n_moved * nbytes / 1e6, 4),
                               "GBs": round(n_moved * nbytes / t / 1e9, 2),
                               "pct_roof": round(100 * n_moved * nbytes / t / STREAM_ROOF, 2)}
            except Exception as e:                                          # noqa: BLE001
                rec[opname] = {"error": f"{type(e).__name__}: {e}"}
                if opname == "clone":
                    out["clone_unsupported"] = rec[opname]["error"]
                try:                            # never leave a capture open for the next point
                    ttnn.end_trace_capture(dev, tid, cq_id=0)
                except Exception:
                    pass
                ttnn.synchronize_device(dev)
        ttnn.deallocate(x)
        ttnn.deallocate(y)
        out["points"].append(rec)
        a.out.write_text(json.dumps(out, indent=1))
        s = rec.get("add", {})
        c = rec.get("clone", {})
        print(f"  {mb:9.3f} MB/operand  add {s.get('us_per_op', -1):9.2f} us "
              f"{s.get('GBs', -1):7.1f} GB/s ({s.get('pct_roof', -1):5.1f} %)   "
              f"clone {c.get('us_per_op', -1):9.2f} us {c.get('GBs', -1):7.1f} GB/s "
              f"({c.get('pct_roof', -1):5.1f} %)", flush=True)

    # ---- fit t = t_fixed + bytes / BW_eff on the small end, and find the knee ---------------
    def fit(op, max_mb):
        pts = [(p[op]["moved_MB"] * 1e6, p[op]["us_per_op"] * 1e-6)
               for p in out["points"] if op in p and "us_per_op" in p[op] and p["MB"] <= max_mb]
        if len(pts) < 3:
            return None
        n = len(pts)
        sx = sum(b for b, _ in pts)
        sy = sum(t for _, t in pts)
        sxx = sum(b * b for b, _ in pts)
        sxy = sum(b * t for b, t in pts)
        den = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / den
        icpt = (sy - slope * sx) / n
        return {"n_points": n, "fit_up_to_MB": max_mb,
                "t_fixed_us": round(1e6 * icpt, 3),
                "BW_eff_GBs": round(1e-9 / slope, 2) if slope > 0 else None,
                "BW_eff_pct_roof": round(100 / slope / STREAM_ROOF, 2) if slope > 0 else None}

    out["fits"] = {op: {f"le_{m}MB": fit(op, m) for m in (2, 8, 32)}
                   for op in ("add", "clone")}
    for op in ("add", "clone"):
        good = [p for p in out["points"] if op in p and "GBs" in p[op]]
        if not good:
            continue
        asym = max(p[op]["GBs"] for p in good)
        knee = next((p["MB"] for p in good if p[op]["GBs"] >= 0.9 * asym), None)
        at155 = min(good, key=lambda p: abs(p[op]["moved_MB"] / 3.0 - 15.5))
        out["fits"][op]["asymptote_GBs"] = asym
        out["fits"][op]["asymptote_pct_roof"] = round(100 * asym * 1e9 / STREAM_ROOF, 2)
        out["fits"][op]["knee_MB_operand"] = knee
        out["fits"][op]["nearest_15_5MB_point"] = {"MB": at155["MB"], **at155[op]}

    # ---- does the cost model reproduce the real pairformer block? ---------------------------
    f = out["fits"]["add"].get("le_8MB") or out["fits"]["add"].get("le_2MB")
    if f and f["BW_eff_GBs"]:
        pred_ms = 1e3 * (BLOCK_OPS * f["t_fixed_us"] * 1e-6 + BLOCK_BYTES / (f["BW_eff_GBs"] * 1e9))
        out["block_model"] = {
            "t_fixed_us": f["t_fixed_us"], "BW_eff_GBs": f["BW_eff_GBs"],
            "ops": BLOCK_OPS, "bytes_GB": BLOCK_BYTES / 1e9,
            "fixed_part_ms": round(1e3 * BLOCK_OPS * f["t_fixed_us"] * 1e-6, 4),
            "byte_part_ms": round(1e3 * BLOCK_BYTES / (f["BW_eff_GBs"] * 1e9), 4),
            "predicted_ms": round(pred_ms, 4), "measured_ms": BLOCK_DEVICE_MS,
            "ratio": round(pred_ms / BLOCK_DEVICE_MS, 4),
        }
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out["fits"][k] for k in out["fits"]}, indent=1), flush=True)
    print(json.dumps(out.get("block_model", {}), indent=1), flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
