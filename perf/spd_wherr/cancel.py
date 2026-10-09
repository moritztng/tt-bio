"""Does adding one K tile onto a partial sum held in dest go wrong when the two nearly cancel?

    TT_VISIBLE_DEVICES=<chip> python perf/spd_wherr/cancel.py --out cancel.json [--rows 65536]

Every row of a [rows, 64] @ [64, 32] matmul is one case. K tile 0 holds s (row r: a[r, 0] = s, w[0] = 1), K tile 1
holds -s + d (a[r, 32] = -s, a[r, 33] = d, w[32] = w[33] = 1), so the exact result is d and the device must add
tile 1's sum onto tile 0's in dest. s is a random bf16 N(0, 1) value, d = s * 2^-j with a random sign and j drawn
from 1..24, so j is the depth of the cancellation in bits. The output is compared exactly (it is d, a bf16).
Run per fidelity at K block 2 (both tiles in one dest pass) and K block 1 (packer_l1_acc adds them), fp32 dest acc.
Prints the wrong-row fraction per j: if the in-dest add mis-normalises on deep cancellation, it rises with j at
K block 2 and stays 0 at K block 1.
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
    sign = torch.where(torch.rand(R) < 0.5, -1.0, 1.0)
    d = (s.float() * torch.exp2(-j.float()) * sign).bfloat16()
    A = torch.zeros(R, 64, dtype=torch.bfloat16)
    A[:, 0], A[:, 32], A[:, 33] = s, -s, d
    W = torch.zeros(64, 32, dtype=torch.bfloat16)
    W[0, 0] = W[32, 0] = W[33, 0] = 1
    ref = d.double()
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
                bad = (Y - ref).abs() > ref.abs() * 2.0 ** -6 + 1e-30
                by_j = {int(k): [int(bad[j == k].sum()), int((j == k).sum())] for k in range(1, 25)}
                ex = [[float(s[i]), float(d[i]), float(Y[i])] for i in bad.nonzero().flatten()[:6].tolist()]
                cell = dict(fid=fid, k_block=kb, wrong=int(bad.sum()), by_j=by_j, examples=ex)
                res["cells"].append(cell)
                print(f"{fid:5s} K{kb}: wrong {cell['wrong']:6d}/{R}  " +
                      " ".join(f"j{k}:{v[0]}" for k, v in by_j.items() if v[0]) + f"  e.g. {ex[:3]}", flush=True)
    finally:
        ttnn.close_device(dev)
        a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
