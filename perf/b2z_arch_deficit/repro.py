#!/usr/bin/env python3
"""A pairformer-shaped op costs 2-3x what its bytes and its per-op floor say it should.

Standalone: torch and ttnn only, no tt-bio, no model weights, no checkpoint. Runs unchanged on
Wormhole and Blackhole. Everything it prices against is measured in the same process on the same
part, so there is no nameplate number anywhere in it.

    calibration   `ttnn.add` on two tiled bf16 DRAM operands, 64 KB to 67 MB per operand, each
                  point a ttnn trace of R repetitions captured once and replayed so the host is
                  out of the measurement. Gives t_fixed (the flat floor at the small end, where
                  8x of bytes does not change the time) and BW_eff (the asymptote).

    instances     the matmuls a Boltz-2 512 aa pairformer block actually issues, at their real
                  shapes, one op each, same instrument. Predicted = t_fixed + bytes / BW_eff,
                  where bytes = (M*K + K*N + M*N) * 2 for A[M,K] @ B[K,N], each operand counted
                  once. That is a LOWER bound on traffic: a matmul that blocks over K re-reads
                  an operand per block, so a large ratio here is also evidence about how much
                  the block's byte count undercounts.

    control       `ttnn.add` at 16.8 MB/operand, which the calibration curve already puts at
                  ~99 % of the asymptote. If the control is at 1.0x and the matmuls are at 2x,
                  the deficit is a property of these ops at these shapes and not of the
                  instrument.

Run it on both parts and compare the RATIOS. On a shared box the absolutes move with the load;
the ratio of measured to a model built from that same process's own roofs does not.

    TT_VISIBLE_DEVICES=<n> python3 repro.py --out repro_<part>.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time
from pathlib import Path

# Boltz-2 512 aa pair track: z is [512, 512, 128], the trunk runs 64 of these blocks.
# name, A shape, B shape, what issues it
INSTANCES = [
    ("transition_in", (512 * 512, 128), (128, 512),
     "pair-track Transition, the 128 -> 512 expand (one of two)"),
    ("transition_out", (512 * 512, 512), (512, 128),
     "pair-track Transition, the 512 -> 128 contract"),
    ("trimul_core", (128, 512, 512), (128, 512, 512),
     "TriangleMultiplication, the per-channel 512-cube batched matmul"),
    ("pair_proj", (512 * 512, 128), (128, 128),
     "the square 128 -> 128 projections either side of every sub-unit"),
]


def replay(ttnn, dev, issue, R, n_med=5):
    for _ in range(3):
        issue()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(R):
        issue()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(2):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    per = []
    for _ in range(n_med):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        per.append((time.perf_counter() - t0) / R)
    ttnn.release_trace(dev, tid)
    return st.median(per), [round(1e6 * p, 3) for p in per]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn

    # A single Blackhole processor of a p300c reads as a CUSTOM cluster (one chip on a two-chip
    # board), and ttnn.open_device then refuses with "Custom fabric mesh graph descriptor path
    # must be specified". The descriptor ships inside the ttnn wheel; point at it and the open
    # succeeds. No effect on any part that does not need it.
    if not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        for root in (Path(ttnn.__file__).parent, Path(ttnn.__file__).parent / "build"):
            mgd = root / "tt_metal" / "fabric" / "mesh_graph_descriptors" / \
                "p150_mesh_graph_descriptor.textproto"
            if mgd.is_file():
                os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)
                break

    dev = ttnn.open_device(device_id=0, trace_region_size=1 << 30)
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": a.label,
                   "host": os.uname().nodename,
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "arch": str(dev.arch()).rsplit(".", 1)[-1],
                   "mesh_graph_desc": os.environ.get("TT_MESH_GRAPH_DESC_PATH"),
                   "grid": str(dev.compute_with_storage_grid_size()),
                   "loadavg": open("/proc/loadavg").read().split()[:3]},
           "calibration": [], "instances": []}

    def mk(shape):
        return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT,
                               dtype=ttnn.bfloat16, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ---- calibration: t_fixed off the flat floor, BW_eff off the asymptote ------------------
    print(f"=== {out['env']['arch']} {out['env']['grid']} calibration ===", flush=True)
    tiles = 1
    while True:
        rows, cols = 1, tiles
        while cols > 2 * rows:
            rows *= 2
            cols = max(1, tiles // rows)
        h, w = rows * 32, cols * 32
        mb = h * w * 2 / 1e6
        if mb > 70:
            break
        if mb >= 0.06:
            x, y = mk((h, w)), mk((h, w))
            z = ttnn.add(x, y)
            ttnn.synchronize_device(dev)
            nbytes = 3 * h * w * 2
            R = int(min(400, max(4, round(0.04 / max(nbytes / 300e9, 1e-6)))))
            t, _ = replay(ttnn, dev, lambda: ttnn.add(x, y, output_tensor=z), R)
            out["calibration"].append({"MB": round(mb, 4), "us": round(1e6 * t, 3),
                                       "moved_MB": round(nbytes / 1e6, 4),
                                       "GBs": round(nbytes / t / 1e9, 2)})
            print(f"  add {mb:8.3f} MB/operand  {1e6 * t:9.2f} us  "
                  f"{nbytes / t / 1e9:7.1f} GB/s", flush=True)
            for tns in (x, y, z):
                ttnn.deallocate(tns)
        tiles *= 2

    t_fixed = min(p["us"] for p in out["calibration"]) * 1e-6
    bw = max(p["GBs"] for p in out["calibration"]) * 1e9
    flat = [p for p in out["calibration"] if p["us"] <= 1.05 * t_fixed * 1e6]
    out["roofs"] = {"t_fixed_us": round(1e6 * t_fixed, 3), "BW_eff_GBs": round(bw / 1e9, 2),
                    "flat_band_points": len(flat),
                    "flat_band_max_MB": max(p["MB"] for p in flat)}
    print(f"  t_fixed {1e6 * t_fixed:.2f} us over {len(flat)} flat points, "
          f"BW_eff {bw / 1e9:.1f} GB/s", flush=True)

    # ---- the control: one add the calibration already says is at the roof ------------------
    ctrl = max((p for p in out["calibration"] if p["MB"] <= 17), key=lambda p: p["MB"])
    out["control_add"] = {**ctrl, "predicted_us": round(
        1e6 * (t_fixed + ctrl["moved_MB"] * 1e6 / bw), 3)}
    out["control_add"]["ratio"] = round(ctrl["us"] / out["control_add"]["predicted_us"], 4)

    # ---- the instances ---------------------------------------------------------------------
    print("=== pairformer-shaped matmuls, one op each ===", flush=True)
    for name, sa, sb, why in INSTANCES:
        rec = {"name": name, "A": list(sa), "B": list(sb), "issued_by": why}
        try:
            x, y = mk(sa), mk(sb)
            batch = sa[0] if len(sa) == 3 else 1
            m, k, n = sa[-2], sa[-1], sb[-1]
            nbytes = 2 * batch * (m * k + k * n + m * n)
            flop = 2 * batch * m * k * n
            z = ttnn.matmul(x, y)
            ttnn.synchronize_device(dev)
            # ttnn.matmul spells its preallocated output `optional_output_tensor`; the eltwise
            # binaries spell it `output_tensor`. Probe rather than hardcode.
            kw = None
            for cand in ("optional_output_tensor", "output_tensor"):
                try:
                    ttnn.matmul(x, y, **{cand: z})
                    kw = cand
                    break
                except TypeError:
                    continue
            if kw is None:
                raise RuntimeError("ttnn.matmul takes no preallocated output on this build")
            ttnn.synchronize_device(dev)
            R = int(min(64, max(3, round(0.04 / max(nbytes / bw, 1e-6)))))
            t, all_us = replay(ttnn, dev, lambda: ttnn.matmul(x, y, **{kw: z}), R)
            pred = t_fixed + nbytes / bw
            rec.update({"R": R, "out_kwarg": kw, "us": round(1e6 * t, 3), "all_us": all_us,
                        "bytes_MB": round(nbytes / 1e6, 3), "GFLOP": round(flop / 1e9, 3),
                        "predicted_us": round(1e6 * pred, 3),
                        "ratio": round(t / pred, 4),
                        "achieved_GBs": round(nbytes / t / 1e9, 2),
                        "achieved_TFLOPs": round(flop / t / 1e12, 3),
                        "pct_of_BW_eff": round(100 * nbytes / t / bw, 2)})
            print(f"  {name:16s} {1e6 * t:10.2f} us  predicted {1e6 * pred:9.2f} us  "
                  f"ratio {t / pred:5.2f}x  {nbytes / t / 1e9:6.1f} GB/s  "
                  f"{flop / t / 1e12:7.2f} TFLOP/s", flush=True)
            for tns in (x, y, z):
                ttnn.deallocate(tns)
        except Exception as e:                                              # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  {name} FAILED {rec['error']}", flush=True)
            ttnn.synchronize_device(dev)
        out["instances"].append(rec)
        a.out.write_text(json.dumps(out, indent=1))

    print(f"  control add {ctrl['MB']} MB/operand ratio "
          f"{out['control_add']['ratio']:.2f}x", flush=True)
    a.out.write_text(json.dumps(out, indent=1))
    ttnn.close_device(dev)
    print("DONE " + str(a.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
