#!/usr/bin/env python3
"""Reduce one wrong element of the Wormhole HiFi4 triangle product to a single output tile.

Regenerates the operands of `perf/mgx_affinity/trimul_mm_variants.py` (seed 2000 + Kt, 8 channels,
scale 1), finds the wrong elements of the full product as the trimul runs it, then re-runs the one
32x32 output tile that holds each of them on a 1x1 grid, and asks:

  1. does the tile alone reproduce the miss (same bits, same element)?
  2. which K block of `in0_block_w` tiles carries it (every other block zeroed)?
  3. which K tiles inside that block (drop one tile at a time)?
  4. which rows / columns of the tile matter (zero all but the element's row of A and column of B)?

Every run is scored against a float64 product of the same bf16 operands.
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
ap.add_argument("--kt", type=int, default=16)
ap.add_argument("--channels", type=int, default=8)
ap.add_argument("--max-elems", type=int, default=4)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
           packer_l1_acc=True)


def mm(A, B, w, grid=(1, 1), M=1, N=1):
    """A [.., m, K] x B [.., n, K]^T, trimul's program config shape, fp32 out."""
    ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
        out_block_h=M, out_block_w=N, per_core_M=M, per_core_N=N, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)
    out = ttnn.matmul(ta, tb, compute_kernel_config=ckc, program_config=pc, dtype=ttnn.float32,
                      transpose_b=True)
    o = ttnn.to_torch(out).double()
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    return o


def err(A, B, o):
    return o - torch.matmul(A.double(), B.double().transpose(-1, -2))


kt, s = a.kt, a.kt * 32
g = torch.Generator().manual_seed(2000 + kt)
A = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
B = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
gx, gy = T.COMPUTE_GRID_MAIN
w = T._trimul_in0_block_w(kt)
e = err(A, B, mm(A, B, w, (gx, gy), -(-kt // gy), -(-kt // gx)))
thr = 0.05 * e.new_tensor(float(torch.matmul(A.double(), B.double().transpose(-1, -2)).abs().max()))
bad = torch.nonzero(e.abs() > thr)[:, 1:].tolist()
print(json.dumps({"kt": kt, "w": w, "wrong_full": len(bad)}), flush=True)
rows = []
for c, i, j in bad[:a.max_elems]:
    ti, tj = i // 32, j // 32
    At = A[:, c:c + 1, ti * 32:ti * 32 + 32, :].clone()
    Bt = B[:, c:c + 1, tj * 32:tj * 32 + 32, :].clone()
    li, lj = i - ti * 32, j - tj * 32
    r = {"c": c, "i": i, "j": j, "err_full": round(float(e[0, c, i, j]), 3)}
    et = err(At, Bt, mm(At, Bt, w))
    r["tile_alone"] = round(float(et[0, 0, li, lj]), 3)
    r["tile_other_bad"] = int((et.abs() > thr).sum()) - int(abs(et[0, 0, li, lj]) > thr)
    # 2. which K block
    blocks = []
    for b in range(kt // w):
        m = torch.zeros(s, dtype=torch.bool)
        m[b * w * 32:(b + 1) * w * 32] = True
        Ab, Bb = At * m, Bt * m
        blocks.append(round(float(err(Ab, Bb, mm(Ab, Bb, w))[0, 0, li, lj]), 3))
    r["block_alone"] = blocks
    # 3. inside the worst block, drop one K tile at a time
    b = max(range(len(blocks)), key=lambda x: abs(blocks[x]))
    drops = []
    for t in range(w):
        m = torch.zeros(s, dtype=torch.bool)
        m[b * w * 32:(b + 1) * w * 32] = True
        m[(b * w + t) * 32:(b * w + t + 1) * 32] = False
        Ab, Bb = At * m, Bt * m
        drops.append(round(float(err(Ab, Bb, mm(Ab, Bb, w))[0, 0, li, lj]), 3))
    r["block"], r["drop_tile"] = b, drops
    # 4. only the element's own row of A and column of B, inside that block
    m = torch.zeros(s, dtype=torch.bool)
    m[b * w * 32:(b + 1) * w * 32] = True
    Ar, Br = torch.zeros_like(At), torch.zeros_like(Bt)
    Ar[0, 0, li] = At[0, 0, li] * m
    Br[0, 0, lj] = Bt[0, 0, lj] * m
    r["row_col_only"] = round(float(err(Ar, Br, mm(Ar, Br, w))[0, 0, li, lj]), 3)
    # the element's own block partial sums, tile by tile, in float64
    x = (At[0, 0, li].double() * Bt[0, 0, lj].double())[b * w * 32:(b + 1) * w * 32]
    r["tile_partials"] = [round(float(v), 3) for v in x.view(w, 32).sum(1).cumsum(0)]
    r["ref_block"] = round(float(x.sum()), 3)
    rows.append(r)
    print(json.dumps(r), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "kt": kt, "w": w, "rows": rows}, indent=1) + "\n")
