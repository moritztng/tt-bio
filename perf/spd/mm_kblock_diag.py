#!/usr/bin/env python3
"""Where and when ttnn.experimental.minimal_matmul returns wrong elements (Wormhole).

pv2_shoist_mm_sweep (shm2, .107) at [512,384] x [384,36864], HiFi4 fp32_dest_acc packer_l1_acc: fp32 with
K_block >= 2 and bf16 with K_block 12 (= all of K) put a few elements 4-8 off a float64 reference whose
outputs are ~N(0, 1); K_block 1 matches. Per case: count of elements off by > 0.5, which output tiles, the
same call 5 times (deterministic?), and the same without packer_l1_acc.
"""
import argparse, json
from pathlib import Path

import torch
import ttnn
from tt_bio import tenstorrent as tt

CASES = [  # (dtype, M rows, K, N, M_block, K_block, N_block)
    ("float32", 512, 384, 36864, 2, 2, 16),
    ("float32", 512, 384, 36864, 2, 1, 16),
    ("bfloat16", 512, 384, 36864, 2, 12, 2),
    ("bfloat16", 512, 384, 36864, 2, 4, 16),
    ("bfloat16", 512, 384, 4096, 2, 12, 2),
    ("bfloat16", 512, 128, 4096, 2, 4, 2),
    ("bfloat16", 512, 256, 4096, 2, 8, 4),
    ("float32", 512, 128, 4096, 2, 2, 4),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = tt.get_device()
    gx, gy = tt.COMPUTE_GRID_MAIN
    res = []
    for l1acc in (True, False):
        ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                     fp32_dest_acc_en=True, packer_l1_acc=l1acc)
        for dname, m, k, n, mb, kb, nb in CASES:
            torch.manual_seed(0)
            xt, wt = torch.randn(1, m, k), torch.randn(k, n) / k ** 0.5
            ref = (xt.double() @ wt.double())[0]
            dt = getattr(ttnn, dname)
            x = ttnn.from_torch(xt, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            w = ttnn.from_torch(wt, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            sw = max(v for v in (1, 2, 4) if nb % v == 0)
            cfg = ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=1,
                                           subblock_w=sw, compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
            row = {"dtype": dname, "m": m, "k": k, "n": n, "blk": [mb, kb, nb], "packer_l1_acc": l1acc, "runs": []}
            prev = None
            for _ in range(5):
                o = ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, dtype=dt, config=cfg)
                got = ttnn.to_torch(o).double().reshape(m, n); ttnn.deallocate(o)
                bad = (got - ref).abs() > 0.5
                idx = bad.nonzero()
                tiles = sorted({(int(i) // 32, int(j) // 32) for i, j in idx[:4096]})
                row["runs"].append({"n_bad": int(bad.sum()), "max_abs": float((got - ref).abs().max()),
                                    "rel_l2": float((got - ref).norm() / ref.norm()), "tiles": tiles[:12],
                                    "n_tiles": len(tiles), "rows": sorted({int(i) % 32 for i, _ in idx[:4096]})[:8],
                                    "same_as_prev": None if prev is None else bool(torch.equal(got, prev))})
                prev = got
            r0 = row["runs"][0]
            print(f"{dname:8s} {m}x{k}x{n} blk {mb},{kb},{nb} l1acc={l1acc}: bad {[r['n_bad'] for r in row['runs']]} "
                  f"tiles {r0['n_tiles']} {r0['tiles'][:4]} rows {r0['rows']} max {r0['max_abs']:.3f} "
                  f"det {[r['same_as_prev'] for r in row['runs'][1:]]}", flush=True)
            res.append(row)
            ttnn.deallocate(x); ttnn.deallocate(w)
            Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
