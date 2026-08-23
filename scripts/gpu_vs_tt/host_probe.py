#!/usr/bin/env python3
"""Measure the HOST of a rented GPU box, so "weaker host" stops being an assertion.

Four of the eight published B200 cells read slower than the H200's, and the two boxes did not
have the same host (Xeon 8559C at 24 vCPU against Xeon 8480+ at ~27). Those four rows are the
dispatch-sensitive ones, so the host is a live confound -- and the published cells carry no
number that would settle it. This probe is that number. It is deliberately tiny, portable and
identical on every box in the campaign, so two rentals can be compared directly instead of
through a spec sheet.

    python host_probe.py --out /root/results/host_probe_<gpu>.json

Four measurements, each chosen because it maps onto a real cost in these models:

  py_loop_s        single-thread pure-python work. Featurisation (AtomWorks, boltz) is largely
                   single-threaded, so this tracks the phase that swung 8.275 s vs 12.459 s
                   between the two published boxes -- in the B200 box's favour.
  gemm_gflops      single-thread 1024^3 numpy GEMM. Host math, one core, BLAS pinned to 1.
  launch_per_s     kernel launches per second with no sync in the loop: pure host-side dispatch
                   throughput. This is the quantity "a weaker host starves a dispatch-bound
                   trunk" actually refers to.
  aten_per_s       the same, through the full python/ATen dispatcher (F.linear on a tiny tensor),
                   which is what a model's python trunk really pays per op.

Torch is optional: without it the two host-only numbers still come out.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def vcpu_cgroup():
    """The container's share. nproc reports the host's cores and is not it."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return round(float(q) / float(p), 2)
    except Exception:
        pass
    try:
        q = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return round(q / p, 2)
    except Exception:
        pass
    return None


def py_loop(n: int = 16_000_000, reps: int = 3) -> float:
    """Best of 3, so a scheduler hiccup on a shared host cannot become the number."""
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        acc = 0.0
        for i in range(n):
            acc += i * 1.000001
        dt = time.perf_counter() - t0
        assert acc > 0
        best = dt if best is None else min(best, dt)
    return best


def gemm(n: int = 1024, reps: int = 5):
    try:
        import numpy as np
    except Exception:
        return None
    a = np.random.rand(n, n).astype("float32")
    b = np.random.rand(n, n).astype("float32")
    a @ b                                        # warm the BLAS
    best = min(_time(lambda: a @ b) for _ in range(reps))
    return round(2.0 * n ** 3 / best / 1e9, 1)


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def dispatch(reps: int = 20_000):
    try:
        import torch
    except Exception:
        return {"torch": None}
    if not torch.cuda.is_available():
        return {"torch": torch.__version__, "cuda": False}
    x = torch.ones(32, 32, device="cuda")
    w = torch.ones(32, 32, device="cuda")
    import torch.nn.functional as F
    for _ in range(200):                         # warm: allocator, kernel cache, autotune
        x.add_(1.0)
        F.linear(x, w)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        x.add_(1.0)
    torch.cuda.synchronize()
    launch = reps / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    for _ in range(reps):
        F.linear(x, w)
    torch.cuda.synchronize()
    aten = reps / (time.perf_counter() - t0)
    return {"torch": torch.__version__, "cuda": True,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability()),
            "launch_per_s": round(launch, 0), "aten_per_s": round(aten, 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reps", type=int, default=20_000)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    rec = {
        "cpu_model": cpu_model(),
        "vcpu_cgroup": vcpu_cgroup(),
        "nproc_host": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "loadavg": os.getloadavg(),
        "py_loop_s": round(py_loop(), 3),
        "gemm_1t_gflops": gemm(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rec.update(dispatch(args.reps))
    print(json.dumps(rec, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
