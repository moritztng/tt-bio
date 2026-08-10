#!/usr/bin/env python3
"""Is the row block bit-exact in the row block?

Row blocking splits the pair tensor along rows and runs the same [K] contraction per row, so a
different block size is a different partition of independent row groups, not a different
accumulation order. That predicts torch.equal. Verify it, because E1 and E6 both found configs that
looked order-preserving and were not once `in0_block_w` moved -- here `in0_block_w` is pinned to the
whole of K in every arm, so only the row grouping changes.
"""
import argparse, json
from pathlib import Path

import torch
import ttnn
import tt_bio.tenstorrent as T

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rowblock_ladder import build_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=298)
    ap.add_argument("--c-in", type=int, default=256)
    ap.add_argument("--c-out", type=int, default=768)
    ap.add_argument("--arms", type=int, nargs="+", default=[33, 34, 50])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    num_cores = gx * gy
    k_tiles, n_tiles = -(-a.c_in // 32), -(-a.c_out // 32)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    torch.manual_seed(0)
    xt = torch.randn(a.cols, a.cols, a.c_in, dtype=torch.bfloat16)
    wt = torch.randn(a.c_in, a.c_out, dtype=torch.bfloat16)
    w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    ref = None
    rows = []
    for r in a.arms:
        cfg, pcm, blocks = build_cfg(r, a.cols, k_tiles, n_tiles, num_cores, gx, gy)
        outs = []
        for s in range(0, a.cols, r):
            blk = xt[s:s + r]
            c, _, _ = build_cfg(blk.shape[0], a.cols, k_tiles, n_tiles, num_cores, gx, gy)
            x = ttnn.from_torch(blk.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
            y = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_MEMORY_CONFIG, program_config=c)
            outs.append(ttnn.to_torch(y))
            ttnn.deallocate(y)
            ttnn.deallocate(x)
        got = torch.cat(outs, dim=0)
        if ref is None:
            ref, ref_r = got, r
            rows.append(dict(rows=r, per_core_M=pcm, cores_used=blocks, role="reference"))
        else:
            eq = bool(torch.equal(got, ref))
            d = (got.float() - ref.float()).abs()
            rows.append(dict(rows=r, per_core_M=pcm, cores_used=blocks,
                             torch_equal=eq, max_abs=float(d.max()),
                             n_diff=int((d != 0).sum()), n_elem=int(d.numel()),
                             vs=ref_r))
    out = dict(cols=a.cols, c_in=a.c_in, c_out=a.c_out, arms=a.arms,
               all_bit_exact=all(r.get("torch_equal", True) for r in rows), rows=rows)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    T.cleanup()


if __name__ == "__main__":
    main()
