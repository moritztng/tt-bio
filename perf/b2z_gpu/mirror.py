#!/usr/bin/env python3
"""Put the GPU's deficit next to the Tenstorrent one, under one estimator.

The campaign's deficit is measured device time over a two-parameter cost model built
from the card's own op-cost curve:

    modelled = n_ops * t_fixed + bytes / BW_eff
    deficit  = measured / modelled

with t_fixed the curve's small-op floor (the per-op device time as bytes -> 0) and
BW_eff its large-size plateau. Both parameters are read off each card's own measured
curve by the same code here, so the two columns cannot drift apart on estimator choice:
run against the Blackhole curve it reproduces the campaign's 6.36 us / 445.0 GB/s /
2.34x from CONTEXT.md section 3.

    python3 mirror.py --tt ../b2x_op_cost/op_cost_curve_512_qb2c0.json \
                      --gpu-roofs roofs_h100.json --gpu-decomp decomp_h100_bf16_v3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Blackhole, qb2 card 0, ttnn trace replay of the shipped call (CONTEXT.md section 2,
# perf/b2x_op_cost/device_floor_512_qb2c0.json). Bytes for the pairformer block are the
# corrected figure from b2x-diffusion-layer-bytes.
TT_UNITS = {
    "PairformerLayer": {"device_ms": 41.4152, "calls": 280, "ops": 428, "bytes": 6.651e9},
    "DiffusionModule": {"device_ms": 32.5179, "calls": 200},
    "MSALayer": {"device_ms": 120.2894, "calls": 16},
}


def curve_params(points, key=None):
    """t_fixed (small-op floor) and BW_eff (large-size plateau) off a measured curve."""
    pts = []
    for p in points:
        q = p[key] if key else p
        if "us_per_op" not in q:
            continue
        pts.append({"MB": p["MB"], "us": q["us_per_op"], "GBs": q["GBs"],
                    "moved": q["moved_MB"] * 1e6})
    pts.sort(key=lambda r: r["MB"])
    # BW_eff is the plateau, not the peak: on a card with a large L2 the peak sits at a
    # cache-resident size and is not a bandwidth this block can have. Taking the largest
    # point's rate is the same quantity the Blackhole curve's peak already is (it is flat
    # from 33 MB up) and is the GPU-favourable choice, which is the right direction for a
    # control that is trying to disprove "the GPU misses too".
    return {"t_fixed_us": pts[0]["us"], "t_fixed_at_MB": pts[0]["MB"],
            "BW_plateau_GBs": pts[-1]["GBs"], "BW_plateau_at_MB": pts[-1]["MB"],
            "BW_peak_GBs": max(r["GBs"] for r in pts),
            "BW_peak_at_MB": max(pts, key=lambda r: r["GBs"])["MB"]}


def model(ops, nbytes, t_fixed_us, bw_gbs):
    fixed = ops * t_fixed_us * 1e-3
    byte = nbytes / (bw_gbs * 1e9) * 1e3
    return {"fixed_ms": round(fixed, 4), "byte_ms": round(byte, 4),
            "modelled_ms": round(fixed + byte, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt", type=Path, required=True)
    ap.add_argument("--gpu-roofs", type=Path, required=True)
    ap.add_argument("--gpu-decomp", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    tt_curve = curve_params(json.loads(a.tt.read_text())["points"], key="add")
    gpu = json.loads(a.gpu_roofs.read_text())
    gpu_curve = curve_params(gpu["points"])
    dec = json.loads(a.gpu_decomp.read_text())

    r = {"tt_curve": tt_curve, "gpu_curve": gpu_curve,
         "gpu_compute_roof": gpu["compute_roof"], "gpu_env": dec["env"],
         "gpu_fold": {"warm_s": dec.get("warm_folds_s"), "cold_s": dec.get("cold_fold_s"),
                      "plddt": dec.get("warm_metrics", {}).get("complex_plddt")},
         "units": {}}

    b = TT_UNITS["PairformerLayer"]
    tt_m = model(b["ops"], b["bytes"], tt_curve["t_fixed_us"], tt_curve["BW_plateau_GBs"])
    r["tt_pairformer"] = {**tt_m, "measured_ms": b["device_ms"],
                          "deficit": round(b["device_ms"] / tt_m["modelled_ms"], 3),
                          "ops": b["ops"], "bytes_GB": round(b["bytes"] / 1e9, 4)}

    for name, u in dec["units"].items():
        c = u.get("census", {})
        if "ops" not in c:
            continue
        meas = u["graph"]["median_ms"]
        row = {"measured_graph_ms": meas, "measured_kernel_ms": u["kernel"]["ms_per_call"],
               "measured_eager_ms": u["eager"]["median_ms"],
               "cuda_kernels": u["kernel"]["cuda_events_per_call"],
               "compute_ops": c["ops"], "view_ops": c.get("view_ops"),
               "bytes_GB": round(c["bytes"] / 1e9, 4)}
        for tag, nops in (("by_cuda_kernels", u["kernel"]["cuda_events_per_call"]),
                          ("by_compute_ops", c["ops"])):
            m = model(nops, c["bytes"], gpu_curve["t_fixed_us"], gpu_curve["BW_plateau_GBs"])
            row[tag] = {**m, "deficit": round(meas / m["modelled_ms"], 3)}
        m = model(u["kernel"]["cuda_events_per_call"], c["bytes"],
                  gpu_curve["t_fixed_us"], gpu_curve["BW_peak_GBs"])
        row["sensitivity_peak_BW"] = {**m, "deficit": round(meas / m["modelled_ms"], 3)}
        tt = TT_UNITS.get(name)
        if tt:
            row["tt_device_ms"] = tt["device_ms"]
            row["tt_over_gpu"] = round(tt["device_ms"] / meas, 3)
        r["units"][name] = row

    print(json.dumps(r, indent=1))
    if a.out:
        a.out.write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
