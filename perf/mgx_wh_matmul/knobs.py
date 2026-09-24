#!/usr/bin/env python3
"""Census of the Wormhole HiFi4 wrong elements over every program-config and compute knob.

The triangle product [1, C, S, S] x [1, C, S, S]^T through trimul's program-config family, operands
as `perf/mgx_affinity/trimul_mm_variants.py` draws them (seed 2000 + Kt, scale 1), scored against a
float64 product of the same bf16 operands:

  wrong: |err| > 16 bf16 roundings of ref (min 2^-8) + 0.5
  gross: |err| > 5 % of max |ref|

Arms, each against the production base (in0_block_w from the trimul band, 1x1 subblock, HiFi4,
fp32 dest, packer L1 acc, fp32 out): every legal in0_block_w, a 2x2 subblock, packer_l1_acc off,
dst_full_sync_en, HiFi3 and HiFi2.
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
ap.add_argument("--kt", type=int, nargs="+", default=[16, 32, 48, 64])
ap.add_argument("--channels", type=int, default=8)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
gx, gy = T.COMPUTE_GRID_MAIN
rows = []
for kt in a.kt:
    s = kt * 32
    g = torch.Generator().manual_seed(2000 + kt)
    A = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
    B = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
    ref = torch.matmul(A.double(), B.double().transpose(-1, -2))
    tol = 16 * (ref.bfloat16().double() - ref).abs().clamp(min=2.0 ** -8) + 0.5
    gross_thr = 0.05 * ref.abs().max()
    ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
    M, N = -(-kt // gy), -(-kt // gx)
    w0 = T._trimul_in0_block_w(kt)
    base = dict(w=w0, sb=(1, 1), l1acc=1, full_sync=0, fid="HiFi4")
    arms = [{**base, "w": w} for w in (1, 2, 4, 8, 16) if kt % w == 0 and w != w0]
    arms = [base] + arms + [{**base, "sb": (2, 2)}, {**base, "l1acc": 0}, {**base, "full_sync": 1},
                            {**base, "fid": "HiFi3"}, {**base, "fid": "HiFi2"}]
    for arm in arms:
        if M % arm["sb"][0] or N % arm["sb"][1]:
            continue
        ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, arm["fid"]), math_approx_mode=False,
                   fp32_dest_acc_en=True, packer_l1_acc=bool(arm["l1acc"]))
        ckc.dst_full_sync_en = bool(arm["full_sync"])
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=arm["w"], out_subblock_h=arm["sb"][0],
            out_subblock_w=arm["sb"][1], out_block_h=M, out_block_w=N, per_core_M=M, per_core_N=N,
            transpose_mcast=False, fused_activation=None, fuse_batch=False)
        r = {"kt": kt, "elems": ref.numel(), **arm}
        try:
            out = ttnn.matmul(ta, tb, compute_kernel_config=ckc, program_config=pc, dtype=ttnn.float32,
                              transpose_b=True)
        except Exception as ex:
            r["error"] = str(ex).splitlines()[0][:160]
            rows.append(r)
            print(json.dumps(r), flush=True)
            continue
        e = ttnn.to_torch(out).double() - ref
        ttnn.deallocate(out)
        big = torch.nonzero(e.abs() > gross_thr)
        r.update(wrong=int((e.abs() > tol).sum()), gross=int(big.shape[0]),
                 rel_l2=float(e.norm() / ref.norm()),
                 gross_at=[[int(c), int(i), int(j), round(float(ref[0, c, i, j]), 2), round(float(e[0, c, i, j]), 2)]
                           for _, c, i, j in big[:8].tolist()])
        rows.append(r)
        print(json.dumps(r), flush=True)
    ttnn.deallocate(ta)
    ttnn.deallocate(tb)
if a.out:
    Path(a.out).write_text(json.dumps({"grid": [gx, gy], "arch": str(dev.arch()), "rows": rows}, indent=1) + "\n")
