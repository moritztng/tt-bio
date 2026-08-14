#!/usr/bin/env python3
"""E0 -- the matmul roof, and the rate at the E-step's own GEMM shapes.

The composed E-step's dominant term is not one of the three built primitives. It is the
orientation x translation squared-difference reduction, whose cross term

    C[p,t,o] = Re sum_j  X_p,t(j) conj(A_o(j))

is a single GEMM of shape [n_particles*n_trans, n_pix] x [n_pix, n_orient]. Projection and
backprojection are transaction-bound; this term is compute-bound, so the composed floor cannot be
the sum of the primitive floors. This screen measures the roof it would run against and the rate
actually reached at the shapes the E-step issues, so the floor in the plan is measured rather than
asserted (memory: roofline-roof-must-be-measured-not-asserted).

Arms:
  roof   -- large square bf16 matmul, the best rate this card gives ttnn.matmul
  shapes -- the E-step shapes, at three resolution crops and two orientation-block widths

Each arm: one warm launch, then best-of-N inside one process, device synchronised at both ends.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
REPS = 5


def timed(dev, a, b, cfg, reps=REPS):
    out = ttnn.matmul(a, b, compute_kernel_config=cfg)
    ttnn.synchronize_device(dev)
    ttnn.deallocate(out)
    walls = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = ttnn.matmul(a, b, compute_kernel_config=cfg)
        ttnn.synchronize_device(dev)
        walls.append(time.perf_counter() - t0)
        ttnn.deallocate(out)
    return min(walls), max(walls)


def run(dev, m, k, n, cfg, tag):
    ah = torch.randn(1, 1, m, k, dtype=torch.bfloat16)
    bh = torch.randn(1, 1, k, n, dtype=torch.bfloat16)
    a = ttnn.from_torch(ah, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(bh, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    lo, hi = timed(dev, a, b, cfg)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    fl = 2.0 * m * k * n
    rec = dict(tag=tag, m=m, k=k, n=n, wall_s=lo, spread=(hi - lo) / lo,
               tflops=fl / lo / 1e12)
    print("%-28s m=%-7d k=%-6d n=%-6d  %8.3f ms  %7.1f TFLOP/s  spread %.1f%%"
          % (tag, m, k, n, lo * 1e3, rec["tflops"], 100 * rec["spread"]), flush=True)
    return rec


def main():
    dev = ttnn.open_device(device_id=0)
    res = []
    try:
        for fid, name in ((ttnn.MathFidelity.LoFi, "LoFi"),
                          (ttnn.MathFidelity.HiFi2, "HiFi2")):
            cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                                   fp32_dest_acc_en=False, packer_l1_acc=True)
            # --- the roof: a large square matmul ---------------------------------------
            for s in (4096, 8192):
                res.append(run(dev, s, s, s, cfg, "roof/%s/%d" % (name, s)))
            # --- the E-step's own shapes -----------------------------------------------
            # m = particles*translations, k = retained Fourier pixels (tile-cropped),
            # n = orientations in a block.
            for (m, k, n) in ((1600, 8192, 4096),
                              (1600, 8192, 18432),
                              (1600, 16384, 4096),
                              (1600, 32768, 4096),
                              (3200, 8192, 4096),
                              (6400, 8192, 4096),
                              (1600, 1024, 4096),
                              (1600, 256, 4096)):
                res.append(run(dev, m, k, n, cfg, "estep/%s" % name))
    finally:
        ttnn.close_device(dev)
    out = HERE / "e0_gemm_roof.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
