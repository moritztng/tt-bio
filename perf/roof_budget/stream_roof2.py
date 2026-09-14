#!/usr/bin/env python3
"""Second control pass: the streaming roof with dispatch amortised, and the byte counter's
deviation from the exact minimum, itemised per buffer.

The first pass measured 345.8 GB/s for a starved 8192^2 add where roofs_p300c_qb2_card2.json has
429.9 GB/s on the same card, and 100.37 TFLOP/s for the 8192 HiFi4 cube where the same file has
85.96. Both differences are in the same direction as their measurement method: this harness enqueues
R calls and syncs once, the published one syncs per call. So this pass widens the block count, reads
the AI clock on both sides, and keeps the capture so the byte counter's per-buffer charge can be
printed rather than inferred.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import torch                                                                  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                           # noqa: E402

ensure_p300_mesh_descriptor()
import ttnn                                                                   # noqa: E402
from real_traffic import counts as byte_counts                                # noqa: E402
from itemize import itemize                                                   # noqa: E402


def aiclk():
    try:
        out = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"], capture_output=True,
                             text=True, timeout=60).stdout
        return [ln.strip() for ln in out.splitlines() if "aiclk" in ln.lower()][:4]
    except Exception:                                                         # noqa: BLE001
        return []


def timed(dev, fn, reps, blocks):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    best = None
    for _ in range(blocks):
        outs = []
        t0 = time.perf_counter()
        for _ in range(reps):
            outs.append(fn())
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / reps
        for o in outs:
            ttnn.deallocate(o)
        best = dt if best is None else min(best, dt)
    return best


def main() -> int:
    out = {"loadavg_before": open("/proc/loadavg").read().split()[:3], "aiclk_before": aiclk()}
    dev = ttnn.open_device(device_id=0)
    try:
        res = []
        for N in (2048, 4096, 8192):
            a = ttnn.from_torch(torch.randn(N, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            b = ttnn.from_torch(torch.randn(N, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            s = timed(dev, lambda: ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG), 20, 12)
            res.append({"op": "add", "N": N, "s": s, "GBps": 3 * N * N * 2 / s / 1e9})
            print("add    %5d  %8.4f ms  %6.1f GB/s" % (N, s * 1e3, res[-1]["GBps"]), flush=True)
            s = timed(dev, lambda: ttnn.clone(a, memory_config=ttnn.DRAM_MEMORY_CONFIG), 20, 12)
            res.append({"op": "clone", "N": N, "s": s, "GBps": 2 * N * N * 2 / s / 1e9})
            print("clone  %5d  %8.4f ms  %6.1f GB/s" % (N, s * 1e3, res[-1]["GBps"]), flush=True)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        out["stream"] = res
        out["aiclk_after_stream"] = aiclk()

        # the byte counter's charge, per buffer, on a matmul whose minimum is exact by arithmetic
        M = K = N = 8192
        a = ttnn.from_torch(torch.randn(M, K, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(torch.randn(K, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=kc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        o = ttnn.matmul(a, b, compute_kernel_config=kc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.graph.end_graph_capture()
        ttnn.deallocate(o)
        nodes = json.loads(g) if isinstance(g, str) else g
        (HERE / "control_matmul_8192.json").write_text(json.dumps(nodes))
        ops, rows = itemize({"nodes": nodes})
        dram = [r for r in rows if r["kind"] == "DRAM"]
        c = byte_counts({"nodes": nodes})
        arith = 2 * (M * K + K * N + M * N)
        out["byte_control"] = {
            "shape": [M, K, N], "arith_min_bytes": arith,
            "counter_bytes": int(c["real_MB"] * 1e6), "ratio": c["real_MB"] * 1e6 / arith,
            "counter_write_bytes": int(c["real_w_MB"] * 1e6),
            "counter_read_bytes": int(c["real_r_MB"] * 1e6),
            "ops": [o["name"] for o in ops],
            "dram_buffers": [{"MB": r["size"] / 1e6, "alloc_op": r["alloc_op"],
                              "n_consumers": r["n_consumers"],
                              "consumers": sorted(set(r["consumer_names"]))} for r in dram],
        }
        print(json.dumps(out["byte_control"], indent=1), flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    finally:
        ttnn.close_device(dev)
    out["aiclk_after"] = aiclk()
    out["loadavg_after"] = open("/proc/loadavg").read().split()[:3]
    (HERE / "stream_roof2.json").write_text(json.dumps(out, indent=1))
    print("WROTE", HERE / "stream_roof2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
