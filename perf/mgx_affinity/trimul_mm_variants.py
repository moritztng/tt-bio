#!/usr/bin/env python3
"""Which knob of the triangle-product matmul produces the -2^k wrong elements.

Same scoring as trimul_mm_probe.py, at one Kt, over: in-kernel transpose vs a pre-transposed
operand, fp32_dest_acc_en, packer_l1_acc, math fidelity, and bf16 vs fp32 output.
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
ap.add_argument("--kt", type=int, nargs="+", default=[48, 64])
ap.add_argument("--channels", type=int, default=8)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
gx, gy = T.COMPUTE_GRID_MAIN
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
rows = []
for kt in a.kt:
    s = kt * 32
    g = torch.Generator().manual_seed(2000 + kt)
    A = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
    B = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
    ref = torch.matmul(A.double(), B.double().transpose(-1, -2))
    tol = 16 * (ref.bfloat16().double() - ref).abs().clamp(min=2.0 ** -8) + 0.5
    ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
    tbt = ttnn.from_torch(B.transpose(-1, -2).contiguous(), layout=ttnn.TILE_LAYOUT, device=dev)
    M, N = -(-kt // gy), -(-kt // gx)
    w = T._trimul_in0_block_w(kt)
    base = dict(transpose=1, dest=1, l1acc=1, fid="HiFi4", out="fp32", w=w)
    arms = [base, {**base, "w": 1}, {**base, "transpose": 0}, {**base, "dest": 0}, {**base, "l1acc": 0},
            {**base, "fid": "HiFi2"}, {**base, "out": "bf16"}, {**base, "dest": 0, "l1acc": 0}]
    for arm in arms:
        ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, arm["fid"]), math_approx_mode=False,
                   fp32_dest_acc_en=bool(arm["dest"]), packer_l1_acc=bool(arm["l1acc"]))
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=arm["w"], out_subblock_h=1, out_subblock_w=1,
            out_block_h=M, out_block_w=N, per_core_M=M, per_core_N=N, transpose_mcast=False,
            fused_activation=None, fuse_batch=False)
        out = ttnn.matmul(ta, tb if arm["transpose"] else tbt, compute_kernel_config=ckc, program_config=pc,
                          dtype=ttnn.float32 if arm["out"] == "fp32" else ttnn.bfloat16,
                          transpose_b=bool(arm["transpose"]))
        o = ttnn.to_torch(out).double()
        ttnn.deallocate(out)
        e = o - ref
        big = torch.nonzero(e.abs() > 16)
        r = {"kt": kt, **arm, "wrong": int((e.abs() > tol).sum()), "gross": int(big.shape[0]),
             "rel_l2": float(e.norm() / ref.norm()),
             "gross_at": [[int(c), int(i), int(j), round(float(ref[0, c, i, j]), 2), round(float(e[0, c, i, j]), 2)]
                          for _, c, i, j in big[:6].tolist()]}
        rows.append(r)
        print(json.dumps(r), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"grid": [gx, gy], "arch": str(dev.arch()), "rows": rows}, indent=1) + "\n")
