#!/usr/bin/env python3
"""Does the -2^k fault of the triangle product reach other matmul shapes on this chip?

[1, 1, M, K] x [1, 1, K, N] bf16 through ttnn.matmul's own program config choice, HiFi4 and HiFi3,
fp32 dest acc and packer L1 acc as the trunk runs them, scored like trimul_mm_variants.py: "gross"
is a miss of at least 5 % of max |ref| against a float64 product of the same bf16 operands.
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
ap.add_argument("--k", type=int, nargs="+", default=[128, 256, 384, 1024, 2048])
ap.add_argument("--n", type=int, nargs="+", default=[128, 512])
ap.add_argument("--fid", nargs="+", default=["HiFi4", "HiFi3"])
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
rows = []
for k in a.k:
    for n in a.n:
        g = torch.Generator().manual_seed(3000 + k + n)
        A = torch.randn(1, 1, a.m, k, generator=g).bfloat16()
        B = torch.randn(1, 1, k, n, generator=g).bfloat16()
        ref = torch.matmul(A.double(), B.double())
        ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
        tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
        for fid in a.fid:
            ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                       fp32_dest_acc_en=True, packer_l1_acc=True)
            out = ttnn.matmul(ta, tb, compute_kernel_config=ckc, dtype=ttnn.float32)
            e = ttnn.to_torch(out).double() - ref
            ttnn.deallocate(out)
            big = torch.nonzero(e.abs() > 0.05 * ref.abs().max())
            r = {"m": a.m, "k": k, "n": n, "fid": fid, "gross": int(big.shape[0]),
                 "rel_l2": float(e.norm() / ref.norm()),
                 "gross_at": [[int(i), int(j), round(float(ref[0, 0, i, j]), 2), round(float(e[0, 0, i, j]), 2)]
                              for _, _, i, j in big[:4].tolist()]}
            rows.append(r)
            print(json.dumps(r), flush=True)
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "rows": rows}, indent=1) + "\n")
