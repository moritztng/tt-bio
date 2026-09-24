#!/usr/bin/env python3
"""Do the auto-configured matmuls a fold runs carry the Wormhole HiFi4 fault, and which config
paths are clean?

[M, K] x [K, N] bf16 at the census's shapes, HiFi4 + fp32 dest + packer L1 acc, bf16 out, through:
  auto      ttnn.linear with no program config and no core grid (ttnn's own choice)
  grid      ttnn.linear(core_grid=CORE_GRID_MAIN), the form most tt-bio call sites use
  mm_auto   ttnn.matmul with no program config

fault: |err| > 0.25 * sqrt(sum_k a_k^2 b_k^2), against a float64 product of the same bf16 operands.
The dot product's own scale, so a result that cancels to ~0 is not called wrong, and the -2^k
misses (tens of times that scale on the triangle product) are far above it while HiFi4's ordinary
accumulation error (~1e-3 of it) is far below.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=16384)
ap.add_argument("--shapes", default="768x3072,768x1536,1536x768,384x1536,384x384,128x512,512x128,256x64")
ap.add_argument("--arms", default="auto,grid,mm_auto")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
           packer_l1_acc=True)
rows = []
for shp in a.shapes.split(","):
    k, n = (int(x) for x in shp.split("x"))
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(4000 + k + n + 7 * seed)
        A = torch.randn(1, 1, a.m, k, generator=g).bfloat16()
        B = torch.randn(k, n, generator=g).bfloat16()
        ref = torch.matmul(A.double(), B.double())
        scale = torch.matmul(A.double() ** 2, B.double() ** 2).sqrt()
        ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
        tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
        for arm in a.arms.split(","):
            kw = dict(compute_kernel_config=ckc, dtype=ttnn.bfloat16)
            if arm == "grid":
                kw["core_grid"] = T.CORE_GRID_MAIN
            f = ttnn.matmul if arm == "mm_auto" else ttnn.linear
            out = f(ta, tb, **kw)
            e = ttnn.to_torch(out).double().reshape(ref.shape) - ref
            ttnn.deallocate(out)
            q = e.abs() / scale.clamp(min=1e-30)
            bad = torch.nonzero(q > 0.25)
            r = {"m": a.m, "k": k, "n": n, "seed": seed, "arm": arm, "elems": ref.numel(),
                 "fault": int(bad.shape[0]), "max_err_over_scale": round(float(q.max()), 4),
                 "p99999_err_over_scale": round(float(q.flatten().kthvalue(int(q.numel() * 0.99999)).values), 5),
                 "at": [[int(i), int(j), round(float(ref[0, 0, i, j]), 3), round(float(e[0, 0, i, j]), 3),
                         round(float(scale[0, 0, i, j]), 3)] for _, _, i, j in bad[:4].tolist()]}
            rows.append(r)
            print(json.dumps(r), flush=True)
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
print("DEST_CARRY_STATS", T.DEST_CARRY_STATS, flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "guard": T.DEST_CARRY_STATS, "rows": rows},
                                      indent=1) + "\n")
