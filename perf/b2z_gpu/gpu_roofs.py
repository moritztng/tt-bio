#!/usr/bin/env python3
"""The GPU's own two roofs and its own op-cost curve, measured, not quoted.

Mirror of perf/b2x_op_cost/op_cost_curve.py for CUDA. Same question, same estimator:

  t(bytes) = t_fixed + bytes / BW_eff

fitted over a size ladder of bf16 elementwise adds, with the host removed. On
Tenstorrent the host is removed by capturing a ttnn trace of R repetitions and
replaying it; here the same thing is a CUDA graph of R repetitions replayed with
events around it. Both measure device time per op and nothing else.

Also measured here, because a roofline claim against a datasheet number is not a
measurement (`roofline-roof-must-be-measured-not-asserted`):

  stream roof   the asymptotic achieved HBM bandwidth of a large elementwise add
  compute roof  achieved bf16 (and tf32/fp32) TFLOP/s of a large square matmul

Usage:  python3 gpu_roofs.py --out roofs_h100.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import time
from pathlib import Path

import torch


def gpu_state() -> dict:
    q = ("name,driver_version,memory.total,clocks.sm,clocks.max.sm,power.draw,"
         "temperature.gpu,utilization.gpu")
    try:
        row = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:                                                  # noqa: BLE001
        row = f"nvidia-smi failed: {e}"
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:                                                  # noqa: BLE001
        apps = f"nvidia-smi failed: {e}"
    # A rented "single GPU" box can share the physical card with another tenant
    # (vast-ai-access): the compute-app list is the cheap detector, so it is recorded
    # with every roof rather than assumed clean.
    return {"gpu": row, "compute_apps": apps, "own_pid": os.getpid()}


def graph_time_per_op(fn, R: int, reps: int = 5) -> float:
    """Seconds of device time per call of ``fn``, host removed by a CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(R):
            fn()
    torch.cuda.synchronize()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        per.append(e0.elapsed_time(e1) * 1e-3 / R)
    del g
    torch.cuda.synchronize()
    return st.median(per)


def fit(points, max_mb):
    """Least squares t = t_fixed + bytes/BW_eff over the points at or below max_mb."""
    pts = [(p["moved_MB"] * 1e6, p["us_per_op"] * 1e-6)
           for p in points if "us_per_op" in p and p["MB"] <= max_mb]
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
            "t_fixed_us": round(1e6 * icpt, 4),
            "BW_eff_GBs": round(1e-9 / slope, 2) if slope > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-mb", type=float, default=512.0)
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device("cuda:0")
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0),
                   "capability": list(torch.cuda.get_device_capability(0)),
                   **gpu_state()},
           "points": []}
    a.out.write_text(json.dumps(out, indent=1))

    # ---- size ladder: the same construction as the TT curve, 2D, 32-aligned, doubling
    sizes = []
    tiles = 1
    while True:
        rows, cols = 1, tiles
        while cols > 2 * rows:
            rows *= 2
            cols = max(1, tiles // rows)
        mb = rows * 32 * cols * 32 * 2 / 1e6
        if mb > a.max_mb:
            break
        if mb >= 0.06:
            sizes.append((rows * 32, cols * 32, mb))
        tiles *= 2

    for h, w, mb in sizes:
        nbytes = h * w * 2
        x = torch.randn(h, w, device=dev, dtype=torch.bfloat16)
        y = torch.randn(h, w, device=dev, dtype=torch.bfloat16)
        z = torch.empty_like(x)
        moved = 3 * nbytes                                # 2 read + 1 written
        rec = {"h": h, "w": w, "MB": round(mb, 4), "moved_MB": round(moved / 1e6, 4)}
        try:
            # aim each graph at ~20 ms of device work so the event timer is not the limit
            t_est = max(moved / 3.0e12, 3e-7)
            R = int(min(2000, max(8, round(0.02 / t_est))))
            t = graph_time_per_op(lambda: torch.add(x, y, out=z), R)
            rec.update({"R": R, "us_per_op": round(1e6 * t, 4),
                        "GBs": round(moved / t / 1e9, 2)})
        except Exception as e:                                              # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        del x, y, z
        torch.cuda.empty_cache()
        out["points"].append(rec)
        a.out.write_text(json.dumps(out, indent=1))
        print(f"  {mb:9.3f} MB/operand  R={rec.get('R', 0):5d}  "
              f"{rec.get('us_per_op', -1):9.3f} us  {rec.get('GBs', -1):8.1f} GB/s", flush=True)

    good = [p for p in out["points"] if "GBs" in p]
    asym = max((p["GBs"] for p in good), default=None)
    knee = next((p["MB"] for p in good if asym and p["GBs"] >= 0.9 * asym), None)
    out["fits"] = {f"le_{m}MB": fit(out["points"], m) for m in (2, 8, 32, 128)}
    out["fits"]["asymptote_GBs"] = asym
    out["fits"]["knee_MB_operand"] = knee

    # ---- compute roof: a large square matmul, the biggest number this card will do ----
    out["compute_roof"] = {}
    for name, dtype, tf32 in (("bf16", torch.bfloat16, False),
                              ("fp16", torch.float16, False),
                              ("tf32", torch.float32, True),
                              ("fp32", torch.float32, False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            n = 8192
            A = torch.randn(n, n, device=dev, dtype=dtype)
            B = torch.randn(n, n, device=dev, dtype=dtype)
            C = torch.empty(n, n, device=dev, dtype=dtype)
            t = graph_time_per_op(lambda: torch.matmul(A, B, out=C), 8)
            out["compute_roof"][name] = {"n": n, "ms": round(1e3 * t, 4),
                                         "TFLOPs": round(2 * n ** 3 / t / 1e12, 2)}
            del A, B, C
            torch.cuda.empty_cache()
        except Exception as e:                                              # noqa: BLE001
            out["compute_roof"][name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  matmul {name:5s} {out['compute_roof'][name]}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False

    out["env_after"] = gpu_state()
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["fits"], indent=1), flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
