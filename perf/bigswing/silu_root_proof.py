"""Step 1a/1b of the exec runbook: prove TT_METAL_RUNTIME_ROOT carries the patched silu LLK,
then time the :2528 shape against its own no-activation twin :2539.

One arm per process. The arm is named by --tag; the runtime root is whatever the caller set.
Nothing here reads or writes the shared site-packages install.
"""
import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

M, K, N, B = 512, 256, 1024, 32


def med(dev, fn, warm, reps):
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)          # every intermediate freed -- a leaked output fabricated a 2.6x once
    return statistics.median(ts), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()

    dev = get_device()
    rec = {"tag": a.tag,
           "runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT", "<unset>"),
           "kernel_cache": os.environ.get("TT_METAL_CACHE", "<unset>"),
           "uptime": open("/proc/loadavg").read().strip(),
           "shape": {"M": M, "K": K, "N": N, "B": B}}
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        rec["ckc"] = {"math_approx_mode": bool(ckc.math_approx_mode),
                      "fp32_dest_acc_en": bool(ckc.fp32_dest_acc_en)}

        x = ttnn.from_torch(torch.randn(1, B, M, K, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        w = ttnn.from_torch(torch.randn(K, N, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

        def silu_arm():
            return ttnn.linear(x, w, activation="silu", compute_kernel_config=ckc,
                               memory_config=ttnn.L1_MEMORY_CONFIG,
                               dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)

        def plain_arm():
            return ttnn.linear(x, w, compute_kernel_config=ckc,
                               memory_config=ttnn.L1_MEMORY_CONFIG,
                               dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)

        m_silu, ts_silu = med(dev, silu_arm, a.warm, a.reps)
        m_plain, ts_plain = med(dev, plain_arm, a.warm, a.reps)
        rec["silu_ms"] = m_silu
        rec["silu_reps"] = ts_silu
        rec["plain_ms"] = m_plain
        rec["plain_reps"] = ts_plain
        rec["activation_cost_ms"] = m_silu - m_plain
        rec["loadavg_end"] = open("/proc/loadavg").read().strip()
        ttnn.deallocate(x); ttnn.deallocate(w)
    except Exception as e:
        rec["error"] = repr(e)
    json.dump(rec, open(a.out, "w"), indent=2)


main()
