#!/usr/bin/env python3
"""Does minimal_matmul's nondeterministic failure at K 128 reach a config tt-bio ships (Wormhole)?

kbd (mm_kblock_diag, .107 chip 24): bf16 [512,128] x [128,4096], block (2, 4, 2), subblock (1, 2), HiFi4 fp32 dest
put 7-12 K elements 4-6 off float64, a different set on every call, with packer_l1_acc on or off. That is not the
known deterministic fp32-dest cancellation fault (dest_guard.py). Per case: the bad count over 3 calls and whether the
calls agree. "probe" cases vary one field of the failing config; "ship" cases are the K-128 entries of
tenstorrent._MM_BLOCK and pair_mm._BLOCK at pair-sized rows.
"""
import argparse, json
from pathlib import Path

import torch
import ttnn
from tt_bio import tenstorrent as tt

CASES = [  # (kind, rows, K, N, M_block, K_block, N_block, subblock_h, subblock_w, fp32_dest)
    ("probe", 512, 128, 4096, 2, 4, 2, 1, 2, True),
    ("probe", 512, 128, 4096, 2, 4, 2, 1, 2, False),
    ("probe", 512, 128, 4096, 2, 4, 2, 1, 1, True),
    ("probe", 512, 128, 4096, 2, 4, 4, 1, 2, True),
    ("probe", 512, 128, 4096, 4, 4, 2, 1, 2, True),
    ("probe", 512, 128, 4096, 2, 1, 2, 1, 2, True),
    ("probe", 512, 128, 4096, 2, 2, 2, 1, 2, True),
    ("probe", 512, 128, 1024, 2, 4, 2, 1, 2, True),
    ("probe", 4096, 128, 4096, 2, 4, 2, 1, 2, True),
    ("ship", 16384, 128, 384, 4, 4, 1, 4, 1, True),     # _MM_BLOCK (4, 12): boltz2 / openfold3 qkv
    ("ship", 16384, 128, 128, 4, 4, 1, 4, 1, True),     # _MM_BLOCK (4, 4): gate
    ("ship", 16384, 128, 128, 8, 4, 1, 4, 1, True),     # pair_mm (4, 4)
    ("ship", 16384, 128, 384, 4, 4, 2, 2, 2, True),     # pair_mm (4, 12)
    ("ship", 16384, 128, 512, 4, 4, 4, 1, 4, True),     # pair_mm (4, 16)
    ("ship", 16384, 128, 1024, 8, 4, 4, 1, 4, True),    # pair_mm (4, 32)
    ("ship", 16384, 128, 384, 4, 4, 2, 2, 2, False),
    ("ship", 16384, 128, 1024, 8, 4, 4, 1, 4, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    dev = tt.get_device()
    gx, gy = tt.COMPUTE_GRID_MAIN
    res = []
    for kind, m, k, n, mb, kb, nb, sh, sw, f32 in CASES:
        ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                     fp32_dest_acc_en=f32, packer_l1_acc=True)
        torch.manual_seed(0)
        xt, wt = torch.randn(1, m, k).bfloat16(), (torch.randn(k, n) / k ** 0.5).bfloat16()
        ref = (xt.double() @ wt.double())[0]
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cfg = ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh,
                                       subblock_w=sw, compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
        row = {"kind": kind, "m": m, "k": k, "n": n, "blk": [mb, kb, nb, sh, sw], "fp32_dest": f32, "bad": [],
               "max_abs": [], "same": []}
        prev = None
        try:
            for _ in range(a.runs):
                o = ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, config=cfg)
                got = ttnn.to_torch(o).double().reshape(m, n); ttnn.deallocate(o)
                err = (got - ref).abs()
                row["bad"].append(int((err > 0.5).sum())); row["max_abs"].append(round(float(err.max()), 4))
                if prev is not None:
                    row["same"].append(bool(torch.equal(got, prev)))
                prev = got
        except Exception as e:  # noqa: BLE001 -- an illegal config is a result, not a crash
            row["error"] = str(e).splitlines()[0][:200]
        print(f"{kind:5s} {m}x{k}x{n} blk {mb},{kb},{nb} sub {sh},{sw} f32={f32}: bad {row['bad']} "
              f"max {row['max_abs']} same {row['same']} {row.get('error', '')}", flush=True)
        res.append(row)
        ttnn.deallocate(x); ttnn.deallocate(w)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
