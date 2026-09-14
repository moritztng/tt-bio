#!/usr/bin/env python3
"""The rate each of the fold's own matmul shapes reaches on this part.

``shape_census.py`` says which shapes carry the FLOPs; this measures them, at the fold's kernel
config (HiFi4, fp32 dest accumulate, packer L1 accumulate), DRAM operands both sides, R enqueues per
sync so host dispatch is amortised, minimum over blocks.

What this bound is and is not. For a shape with high arithmetic intensity the measured rate is an
arithmetic rate. For a thin shape -- K = 32, or M x K activations against a 128-wide weight -- the
standalone matmul is itself DRAM-bound, so its rate carries the traffic wall inside it. So the sum
of FLOPs/rate is NOT an independent arithmetic floor to be added to the traffic floor. It is the
'shapes as issued' bound: what the machine does with this fold's arithmetic using today's op set at
today's shapes, traffic included.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                           # noqa: E402

ensure_p300_mesh_descriptor()
import ttnn                                                                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, default=HERE / "shape_census.json")
    ap.add_argument("--out", type=Path, default=HERE / "shape_rates.json")
    ap.add_argument("--cover", type=float, default=99.0, help="cumulative %% of FLOPs to measure")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=5)
    a = ap.parse_args()

    census = json.loads(a.census.read_text())
    want, cum = [], 0.0
    for r in census["rows"]:
        want.append(r)
        cum += r["pct"]
        if cum >= a.cover:
            break

    kc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    dev = ttnn.open_device(device_id=0)
    out = []
    try:
        for r in want:
            b, M, K, N = r["batch"], r["M"], r["K"], r["N"]
            ash = (b, M, K) if b > 1 else (M, K)
            bsh = (b, K, N) if b > 1 else (K, N)
            try:
                x = ttnn.from_torch(torch.randn(*ash, dtype=torch.bfloat16),
                                    layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                y = ttnn.from_torch(torch.randn(*bsh, dtype=torch.bfloat16),
                                    layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for _ in range(2):
                    ttnn.deallocate(ttnn.matmul(x, y, compute_kernel_config=kc,
                                                memory_config=ttnn.DRAM_MEMORY_CONFIG))
                ttnn.synchronize_device(dev)
                best = None
                for _ in range(a.blocks):
                    outs = []
                    t0 = time.perf_counter()
                    for _ in range(a.reps):
                        outs.append(ttnn.matmul(x, y, compute_kernel_config=kc,
                                                memory_config=ttnn.DRAM_MEMORY_CONFIG))
                    ttnn.synchronize_device(dev)
                    dt = (time.perf_counter() - t0) / a.reps
                    for o in outs:
                        ttnn.deallocate(o)
                    best = dt if best is None else min(best, dt)
                ttnn.deallocate(x)
                ttnn.deallocate(y)
            except Exception as e:                                            # noqa: BLE001
                out.append(dict(r, error=f"{type(e).__name__}: {e}"[:160]))
                print("b=%-5d %6dx%-5dx%-6d FAILED %s" % (b, M, K, N, type(e).__name__),
                      flush=True)
                a.out.write_text(json.dumps({"rows": out}, indent=1))
                continue
            f = 2 * b * M * N * K
            byt = 2 * b * (M * K + K * N + M * N)
            row = dict(r, s=best, TFLOPs=f / best / 1e12, GBps=byt / best / 1e9,
                       AI=f / byt, s_per_fold=r["TFLOP_per_fold"] * 1e12 / (f / best))
            out.append(row)
            print("b=%-5d %6dx%-5dx%-6d  %8.4f ms  %7.2f TFLOP/s  %6.1f GB/s  "
                  "%6.3f TF/fold -> %6.3f s" % (b, M, K, N, best * 1e3, row["TFLOPs"],
                                                row["GBps"], r["TFLOP_per_fold"],
                                                row["s_per_fold"]), flush=True)
            a.out.write_text(json.dumps({"rows": out}, indent=1))
    finally:
        ttnn.close_device(dev)

    ok = [r for r in out if "TFLOPs" in r]
    covered = sum(r["TFLOP_per_fold"] for r in ok)
    floor = sum(r["s_per_fold"] for r in ok)
    total = census["total_TFLOP"]
    summary = {
        "measured_shapes": len(ok), "covered_TFLOP": covered, "total_TFLOP": total,
        "coverage_pct": 100 * covered / total,
        "shapes_as_issued_floor_s": floor,
        "shapes_as_issued_floor_s_extrapolated": floor * total / covered if covered else 0,
        "mean_rate_TFLOPs": covered / floor if floor else 0,
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }
    a.out.write_text(json.dumps({"rows": out, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
