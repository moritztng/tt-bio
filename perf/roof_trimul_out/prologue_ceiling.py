"""ROOF phase A, link 2 (PROLOGUE): what `layer_norm` -> `_pair_proj_linear` can possibly return.

FUSION_PAIRS rank 2, 268.4 MB of one Pairformer block: the trimul's out-norm writes a 67.11 MB
pair tensor to DRAM and the out-projection reads it straight back, nothing else touching it.
Fusing means the matmul's reader normalises the tile it just loaded, which is available here --
the normalised axis IS the contraction axis (c_z = 128, kt = 4), so the reader already holds a
whole normalisation row in one K block.

Building that is a compute-kernel prologue. Before building it, this measures its CEILING. A
prologue fusion deletes the layer norm's DRAM write, the matmul's read of it and one dispatch; it
keeps the arithmetic. So it can never return more than the whole `layer_norm` program costs, which
is exactly `t(layer_norm + proj) - t(proj)`. If that ceiling is under what a fused kernel at this
site is measured to cost, the link is dead without writing the kernel.

The same shapes and the same interleaving as `op_ab.py`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn

from tt_bio import tenstorrent as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--cz", type=int, default=128)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ckc = T.trunk_compute_kernel_config(compute_kernel_config())
    N, C = a.n, a.cz
    torch.manual_seed(0)
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = tt(torch.randn(1, N, N, C) * 0.5)
    xn = tt(torch.randn(1, N, N, C) * 0.5)          # a pre-normalised stand-in, same shape
    wp = tt(torch.randn(C, C) * (C ** -0.5))
    gam, beta = tt(torch.ones(32, C)), tt(torch.zeros(32, C))

    def norm(t):
        return ttnn.layer_norm(t, weight=gam, bias=beta, epsilon=1e-5,
                               compute_kernel_config=ckc)

    def arm_both():
        n = norm(x)
        p = T._trimul_out_proj(n, wp, ckc)
        ttnn.deallocate(n)
        return p

    def arm_proj():
        return T._trimul_out_proj(xn, wp, ckc)

    def arm_norm():
        return norm(x)

    def arm_copy():
        # The same 67.11 MB read and 67.11 MB write with no arithmetic at all. This is what the
        # round trip the fusion deletes actually costs at this site, and 134.2 MB over it is the
        # achieved stream rate the deleted bytes would come back at.
        return ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    fns = {"both": arm_both, "proj": arm_proj, "norm": arm_norm, "copy": arm_copy,
           "both2": arm_both}

    def once(fn):
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        ttnn.deallocate(r)
        return dt

    for k in fns:
        once(fns[k])
    ttnn.synchronize_device(dev)
    s = {k: [] for k in fns}
    for _ in range(a.reps):
        for k in fns:
            s[k].append(once(fns[k]))
    med = {k: statistics.median(v) * 1e3 for k, v in s.items()}
    ceiling = med["both"] - med["proj"]
    res = {"n": N, "cz": C, "arch": str(dev.arch()), "grid": list(T.COMPUTE_GRID_MAIN),
           "reps": a.reps, "median_ms": {k: round(v, 4) for k, v in med.items()},
           "ms": {k: [round(x * 1e3, 4) for x in v] for k, v in s.items()},
           "prologue_ceiling_ms": round(ceiling, 4),
           "aa_floor_pct": round(100 * (med["both2"] - med["both"]) / med["both"], 3),
           "ceiling_ratio_on_pair": round(med["both"] / (med["both"] - ceiling), 4),
           "roundtrip_MB": round(2 * 2 * N * N * C / 1e6, 3),
           "achieved_GB_s": round(2 * 2 * N * N * C / 1e9 / (med["copy"] / 1e3), 2)}
    print(json.dumps({k: res[k] for k in ("median_ms", "prologue_ceiling_ms", "aa_floor_pct",
                                          "ceiling_ratio_on_pair", "roundtrip_MB",
                                          "achieved_GB_s")}, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
