#!/usr/bin/env python3
"""Q9 rescore, probe 4 — why the triangle contraction is 1.58x slower in the fold than at [.,320,320].

The block dump at the TRUE shape times the contraction at 113.94 us/call, where the same batched
matmul built at logical 320x320 stands alone at 72.18 us. The only difference is the LOGICAL size:
in a real fold the token axis is 298 and lands in the contraction's M and N positions, so the last
tile of each row and column is 22/32 real. Isolate it: hold the padded shape at [1, 32, 320, 320]
and vary only which axes are logically short.

    PYTHONPATH=<wt> python3 perf/rescore/r_probe4.py --out perf/rescore/r_probe4_qb2c0.json
"""
import argparse
import json
import statistics as st
import time

import torch
import ttnn

import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import get_device

L1 = ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=3, pipe=6, reps=7):
    for _ in range(warm):
        r = fn()
        del r
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(pipe)]
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
        del keep
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    pc = T._triangle_mul_program_config(10)
    gf = 2 * 32 * 320 ** 3 / 1e9                # padded FLOPs: what the hardware issues
    gf_log = 2 * 32 * 298 ** 3 / 1e9            # logical FLOPs: what the model asks for
    R = {"padded_gflop": round(gf, 3), "logical_gflop_298": round(gf_log, 3),
         "prod_pc": {"in0_block_w": pc.in0_block_w, "per_core_M": pc.per_core_M,
                     "per_core_N": pc.per_core_N}}
    for am, an in ((320, 320), (298, 298), (298, 320), (320, 298)):
        a = ttnn.from_torch(torch.randn(1, 32, am, an), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        b = ttnn.from_torch(torch.randn(1, 32, an, am), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        key = f"a{am}x{an}"
        try:
            s = timed(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                                               program_config=pc, dtype=ttnn.bfloat16))
            R[key] = {"logical_a": [am, an], "padded_a": list(a.padded_shape),
                      "us": round(s * 1e6, 2), "tflops_padded": round(gf / s / 1e3, 2)}
        except Exception as e:                                              # noqa: BLE001
            R[key] = {"err": str(e)[:110]}
        print(f"  a logical {am}x{an} (padded {list(a.padded_shape)}): {R[key]}", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    json.dump(R, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
