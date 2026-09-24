#!/usr/bin/env python3
"""Does the trimul's reuse-multicast matmul return wrong elements, and at which shapes?

[1, C, S, S] x [1, C, S, S]^T with transpose_b, the trimul's own program config family, scored
against a float64 torch product of the same bf16 operands. An element counts as WRONG when it
misses by more than 16x the bf16 rounding of the reference (an fp32 accumulator's honest error is
far below that). Prints, per (Kt, in0_block_w, packer_l1_acc), the wrong count and where they sit.
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
ap.add_argument("--kt", type=int, nargs="+", default=[16, 32, 40, 48, 49, 56, 64, 65, 70, 72, 77, 80, 81])
ap.add_argument("--channels", type=int, default=4)
ap.add_argument("--l1acc", type=int, nargs="+", default=[1])
ap.add_argument("--all-w", action="store_true", help="every divisor of Kt up to the band, not just band and 1")
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = T.get_device()
gx, gy = T.COMPUTE_GRID_MAIN
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
rows = []
for kt in a.kt:
    s = kt * 32
    g = torch.Generator().manual_seed(1000 + kt)
    A = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
    B = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
    ref = torch.matmul(A.double(), B.double().transpose(-1, -2))
    tol = 16 * (ref.bfloat16().double() - ref).abs().clamp(min=2.0 ** -8) + 0.5
    ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
    M, N = -(-kt // gy), -(-kt // gx)
    band = T._trimul_in0_block_w(kt)
    ws = sorted({d for d in range(1, band + 1) if kt % d == 0}) if a.all_w else sorted({1, band})
    for l1acc in a.l1acc:
        ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                   fp32_dest_acc_en=True, packer_l1_acc=bool(l1acc))
        for w in ws:
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
                out_block_h=M, out_block_w=N, per_core_M=M, per_core_N=N, transpose_mcast=False,
                fused_activation=None, fuse_batch=False)
            out = ttnn.matmul(ta, tb, compute_kernel_config=ckc, program_config=pc, dtype=ttnn.float32,
                              transpose_b=True)
            o = ttnn.to_torch(out).double()
            ttnn.deallocate(out)
            e = (o - ref).abs()
            bad = torch.nonzero(e > tol)
            r = {"kt": kt, "tokens": s, "w": w, "band": band, "l1acc": l1acc, "M": M, "N": N,
                 "wrong": int(bad.shape[0]), "max_abs": float(e.max()),
                 "rel_l2": float((o - ref).norm() / ref.norm()),
                 "where": [[int(c), int(i) // 32, int(j) // 32, round(float(o[0, c, i, j] - ref[0, c, i, j]), 2)]
                           for _, c, i, j in bad[:12].tolist()]}
            rows.append(r)
            print(json.dumps(r), flush=True)
    ttnn.deallocate(ta)
    ttnn.deallocate(tb)
if a.out:
    Path(a.out).write_text(json.dumps({"grid": [gx, gy], "arch": str(dev.arch()), "rows": rows}, indent=1) + "\n")
