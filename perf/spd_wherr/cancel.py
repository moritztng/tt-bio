"""Does adding one K tile onto a partial sum held in dest go wrong when the two nearly cancel?

    TT_VISIBLE_DEVICES=<chip> python perf/spd_wherr/cancel.py --out cancel.json [--rows 65536]

Every row of a [rows, 64] @ [64, 32] matmul is one case: K tile 0 sums s + e0 (a[r, 0], a[r, 1]), K tile 1 sums
-s + e1 (a[r, 32], a[r, 33]), w = 1 at those four positions. s is bf16 N(0, 1), e0 and e1 are s * 2^-j * N(0, 1)
in bf16, so the exact result e0 + e1 is ~2^-j smaller than either tile's partial sum: j is the depth of the
cancellation the dest add has to survive. A row is WRONG when the error exceeds |s| / 2: no rounding of these
operands can do that, the erratum (about -2^k) does. Run per fidelity at K block 2 (tile 1 is added onto tile 0
in dest) and K block 1 (packer_l1_acc adds them), fp32 dest acc on.
"""
import argparse, json
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--fid", default="HiFi4,HiFi3,HiFi2,LoFi")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.manual_seed(1)
    R = a.rows
    s = torch.randn(R).bfloat16()
    j = torch.randint(1, 25, (R,))
    scale = s.float() * torch.exp2(-j.float())
    e0, e1 = (scale * torch.randn(R)).bfloat16(), (scale * torch.randn(R)).bfloat16()
    A = torch.zeros(R, 64, dtype=torch.bfloat16)
    A[:, 0], A[:, 1], A[:, 32], A[:, 33] = s, e0, -s, e1
    W = torch.zeros(64, 32, dtype=torch.bfloat16)
    W[0, 0] = W[1, 0] = W[32, 0] = W[33, 0] = 1
    ref = e0.double() + e1.double()
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    res = {"arch": str(dev.arch()), "rows": R, "cells": []}
    try:
        ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        tw = ttnn.from_torch(W, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for fid in a.fid.split(","):
            ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                        math_approx_mode=False, fp32_dest_acc_en=True,
                                                        packer_l1_acc=True)
            for kb in (2, 1):
                cfg = ttnn.MinimalMatmulConfig(M_block_size=4, K_block_size=kb, N_block_size=1, subblock_h=4,
                                               subblock_w=1, compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y))
                y = ttnn.experimental.minimal_matmul(input_tensor=ta, weight_tensor=tw, compute_kernel_config=ck,
                                                     dtype=ttnn.bfloat16, config=cfg)
                Y = ttnn.to_torch(y).double()[:, 0]; ttnn.deallocate(y)
                bad = (Y - ref).abs() > s.double().abs() / 2
                by_j = {int(k): [int(bad[j == k].sum()), int((j == k).sum())] for k in range(1, 25)}
                ex = [[float(s[i]), float(ref[i]), float(Y[i])] for i in bad.nonzero().flatten()[:6].tolist()]
                cell = dict(fid=fid, k_block=kb, wrong=int(bad.sum()), by_j=by_j, examples=ex)
                res["cells"].append(cell)
                print(f"{fid:5s} K{kb}: wrong {cell['wrong']:6d}/{R}  " +
                      " ".join(f"j{k}:{v[0]}" for k, v in by_j.items() if v[0]) + f"  e.g. {ex[:3]}", flush=True)
    finally:
        ttnn.close_device(dev)
        a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
