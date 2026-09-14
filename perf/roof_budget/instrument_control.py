#!/usr/bin/env python3
"""The known-answer control for both counters, and the per-shape arithmetic roof.

A dense matmul has an exact FLOP count (2*batch*M*N*K) and an exact minimum byte count
((M*K + K*N + M*N) * dtype, every operand read once and the result written once). Run one through
the same ttnn.graph capture the fold's table is built from and both counters either return those
numbers or they are unfit.

The same run measures the rate each shape achieves, at the fold's own kernel config (HiFi4,
fp32 dest accumulate, packer L1 accumulate), because the fold's arithmetic floor cannot be priced
at the 8192-cube rate: no matmul in this fold is an 8192 cube. The ladder below is the fold's own
matmul shapes, read off its captures, plus the three cubes that set the published roof.

Timing: R back-to-back enqueues and one sync, so host dispatch is amortised, and the MINIMUM over
blocks is reported. A roof is a maximum rate, so the minimum time is the right statistic and the
one least contaminated by a co-tenant.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                           # noqa: E402

ensure_p300_mesh_descriptor()                       # a lone p300c chip is a CUSTOM topology and
#                                                     ttnn.open_device refuses it without a 1x1 MGD
import ttnn                                                                   # noqa: E402
import exec_flops as EF                                                       # noqa: E402
from real_traffic import counts as byte_counts                                # noqa: E402

# (label, batch, M, K, N) -- every entry read off a tip capture except the three cubes.
LADDER = [
    ("cube-2048",                1, 2048, 2048, 2048),
    ("cube-4096",                1, 4096, 4096, 4096),
    ("cube-8192",                1, 8192, 8192, 8192),
    ("trimul-inproj",            1, 262144, 128, 640),
    ("trimul-einsum",          128, 512, 512, 512),
    ("trimul-outproj",           1, 262144, 128, 128),
    ("triatt-qkv",               1, 262144, 128, 544),
    ("triatt-outproj",           1, 262144, 128, 128),
    ("pair-transition-up",       1, 262144, 128, 512),
    ("pair-transition-down",     1, 262144, 512, 128),
    ("opm-outer",                1, 16384, 1024, 16384),
    ("opm-proj",                 1, 262144, 1024, 128),
    ("pwa-proj",                 1, 262144, 1024, 128),
    ("token-dit-qkv",            1, 512, 768, 2304),
    ("token-dit-transition",     1, 512, 768, 3072),
    ("atom-dit-qkv",             1, 4480, 128, 384),
]


def cfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def run(dev, batch, M, K, N, reps, blocks):
    ash = (batch, M, K) if batch > 1 else (M, K)
    bsh = (batch, K, N) if batch > 1 else (K, N)
    a = ttnn.from_torch(torch.randn(*ash, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.randn(*bsh, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kc = cfg()
    for _ in range(2):
        ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=kc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))
    ttnn.synchronize_device(dev)

    best = None
    for _ in range(blocks):
        outs = []
        t0 = time.perf_counter()
        for _ in range(reps):
            outs.append(ttnn.matmul(a, b, compute_kernel_config=kc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / reps
        for o in outs:
            ttnn.deallocate(o)
        best = dt if best is None else min(best, dt)

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    o = ttnn.matmul(a, b, compute_kernel_config=kc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = ttnn.graph.end_graph_capture()
    ttnn.deallocate(o)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    nodes = json.loads(g) if isinstance(g, str) else g
    return best, nodes


def stream_roof(dev, N, reps, blocks):
    """The streaming roof, re-measured here so the table's two roofs come off one session.

    A bf16 ttnn.add of two NxN DRAM tensors reads 2*N*N*2 bytes and writes N*N*2. It computes one
    FLOP per element, so it is starved: whatever rate it reaches is the streaming rate. 429.9 GB/s
    (roofs_p300c_qb2_card2.json) and 444.9 GB/s (b2z-arch-deficit) are both on record for this part
    and they are not interchangeable, so this decides which one the table uses.
    """
    a = ttnn.from_torch(torch.randn(N, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.randn(N, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for _ in range(2):
        ttnn.deallocate(ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    ttnn.synchronize_device(dev)
    best = None
    for _ in range(blocks):
        outs = []
        t0 = time.perf_counter()
        for _ in range(reps):
            outs.append(ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / reps
        for o in outs:
            ttnn.deallocate(o)
        best = dt if best is None else min(best, dt)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return {"N": N, "bytes": 3 * N * N * 2, "s": best, "GBps": 3 * N * N * 2 / best / 1e9}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "instrument_control.json")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=4)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    rows = []
    stream = []
    try:
        for N in (4096, 8192):
            r = stream_roof(dev, N, a.reps, a.blocks)
            stream.append(r)
            print("stream %dx%d  %7.3f ms  %6.1f GB/s" % (N, N, r["s"] * 1e3, r["GBps"]), flush=True)
        for label, batch, M, K, N in LADDER:
            try:
                s, nodes = run(dev, batch, M, K, N, a.reps, a.blocks)
            except Exception as e:                                    # noqa: BLE001
                rows.append({"label": label, "error": f"{type(e).__name__}: {e}"[:200]})
                print("%-24s FAILED %s" % (label, type(e).__name__), flush=True)
                continue
            flops_arith = 2 * batch * M * N * K
            bytes_arith = 2 * (batch * M * K + batch * K * N + batch * M * N)
            t = EF.totals(nodes)
            bc = byte_counts({"nodes": nodes})
            r = {
                "label": label, "batch": batch, "M": M, "K": K, "N": N,
                "flops_arith": flops_arith, "flops_counter": t["matmul_logical"],
                "flops_counter_padded": t["matmul_padded"],
                "flop_ratio": t["matmul_logical"] / flops_arith,
                "bytes_arith_min": bytes_arith, "bytes_counter": int(bc["real_MB"] * 1e6),
                "byte_ratio": bc["real_MB"] * 1e6 / bytes_arith,
                "s": s, "TFLOPs": flops_arith / s / 1e12, "GBps": bytes_arith / s / 1e9,
                "AI_flop_per_byte": flops_arith / bytes_arith,
            }
            rows.append(r)
            print("%-24s %7.3f ms  %7.2f TFLOP/s  %6.1f GB/s   flop x%.6f  byte x%.4f"
                  % (label, s * 1e3, r["TFLOPs"], r["GBps"], r["flop_ratio"], r["byte_ratio"]),
                  flush=True)
            a.out.write_text(json.dumps({"rows": rows, "stream": stream}, indent=1))
    finally:
        ttnn.close_device(dev)

    ok = [r for r in rows if "flop_ratio" in r]
    verdict = {
        "n": len(ok),
        "flop_exact": all(abs(r["flop_ratio"] - 1) < 1e-9 for r in ok),
        "byte_max_ratio": max((r["byte_ratio"] for r in ok), default=0),
        "byte_min_ratio": min((r["byte_ratio"] for r in ok), default=0),
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }
    verdict["stream_roof_GBps"] = max((r["GBps"] for r in stream), default=0)
    a.out.write_text(json.dumps({"rows": rows, "stream": stream, "verdict": verdict}, indent=1))
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
