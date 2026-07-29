"""Is the 13x10 grid bit-exact against the 11x10 grid rfd3.py actually pins?

p28 §2: `tt_bio/rfd3.py` snapshots `CORE_GRID_MAIN` at import time, but
`_configure_active_compute_grid` rewrites that global to 13x10 when a Blackhole device opens.
So every explicit `core_grid=CORE_GRID_MAIN` in rfd3.py -- including the pair-bias projection
p20 measured as the biggest grid win in the DiT -- runs on 110 of the 130 available cores.

Whether fixing that is free or release-gated turns on ONE question this script answers: is the
13x10 result BITWISE equal to the 11x10 result on those shapes? Only `in0_block_w` regroups the
fp32 accumulation (rfd3.py documents the derivation from tt-metal's config builders), and its
branch predicate depends on the grid -- so the two grids may or may not land on the same value.
Measured per shape, not argued.

Shapes are the real pinned call sites, taken from the shipped module dimensions:
  * the DiT/encoder pair-bias projection   [B, I*I or L*K, C_pair] @ [C_pair, 16]
  * DiffusionTokenEncoder.process_z         [B, I*I, 258] @ [258, 128]
  * process_a / process_r / to_r_update     [B, L, 3 or 128] @ [., 128 or 3]
  * _grid_if_single_k_tile's K<=32 sites    [B*I, n_head, n_query, head_dim]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPS = 20


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=250)
    ap.add_argument("--atoms", type=int, default=3359)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    return ap.parse_args()


def main():
    args = parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import rfd3

    dev = tt.get_device()
    dx = int(dev.compute_with_storage_grid_size().x)
    dy = int(dev.compute_with_storage_grid_size().y)
    print(json.dumps({"device_grid": [dx, dy],
                      "rfd3.CORE_GRID_MAIN": [rfd3.CORE_GRID_MAIN.x, rfd3.CORE_GRID_MAIN.y],
                      "tenstorrent.CORE_GRID_MAIN": [tt.CORE_GRID_MAIN.x, tt.CORE_GRID_MAIN.y]}),
          flush=True)
    if dx < 13:
        print("SKIP device grid narrower than 13 -- nothing to compare", flush=True)
        return

    ckc = rfd3._default_compute_kernel_config()
    dt = ttnn.bfloat16
    I, L = args.tokens, args.atoms
    C_PAIR = rfd3.LocalTokenTransformer.C_PAIR          # 128
    C_ATOM = rfd3.RFD3DiffusionModule.C_ATOM            # 128
    C_Z = rfd3.RFD3DiffusionModule.C_Z                  # 128
    N_BINS = rfd3.DiffusionTokenEncoder.N_BINS
    DIT_KEYS = rfd3.RFD3DiffusionModule.DIT_KEYS        # 32
    N_ATTN_KEYS = rfd3.RFD3DiffusionModule.N_ATTN_KEYS  # 128

    # (name, rows, K, N, calls per step at n_recycle=2)
    sites = [
        ("dit.pair_bias",      I * I,      C_PAIR, 16, 18 * 2),
        ("enc.pair_bias",      L * N_ATTN_KEYS, 16, 16, 3),
        ("token_enc.process_z", I * I, C_Z + 2 * N_BINS, C_Z, 2),
        ("dm.process_a",       L, 3, C_ATOM, 1),
        ("dm.process_r",       L, 3, C_ATOM, 1),
        ("dm.to_r_update",     L, C_ATOM, 3, 2),
        ("attn.single_k_tile", L * 4, 32, 32, 6),
    ]
    grids = {"g11x10": ttnn.CoreGrid(y=10, x=11), "g13x10": ttnn.CoreGrid(y=10, x=13)}

    saved = {1: 0.0, 8: 0.0}
    exact_all = True
    for B in args.batches:
        for name, M, K, N, per_step in sites:
            try:
                x = ttnn.from_torch(torch.randn(B, M, K, dtype=torch.bfloat16),
                                    layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
                w = ttnn.from_torch(torch.randn(K, N, dtype=torch.bfloat16),
                                    layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
            except Exception as exc:
                print("ROW " + json.dumps({"site": name, "D": B, "alloc_err": str(exc)[:120]}),
                      flush=True)
                continue
            row = {"site": name, "D": B, "M": M, "K": K, "N": N, "calls_per_step": per_step}
            outs = {}
            for label, grid in grids.items():
                try:
                    o = ttnn.linear(x, w, core_grid=grid, compute_kernel_config=ckc, dtype=dt)
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(REPS):
                        tmp = ttnn.linear(x, w, core_grid=grid,
                                          compute_kernel_config=ckc, dtype=dt)
                        ttnn.deallocate(tmp)
                    ttnn.synchronize_device(dev)
                    row[label] = round((time.perf_counter() - t0) / REPS * 1000, 4)
                    outs[label] = o
                except Exception as exc:
                    row[label] = None
                    row[label + "_err"] = str(exc)[:120]
            if len(outs) == 2:
                ta = ttnn.to_torch(outs["g11x10"]).float()
                tb = ttnn.to_torch(outs["g13x10"]).float()
                row["exact_13_vs_11"] = bool(torch.equal(ta, tb))
                row["maxabs_13_vs_11"] = float((ta.double() - tb.double()).abs().max())
                exact_all = exact_all and row["exact_13_vs_11"]
                if row["g11x10"] and row["g13x10"]:
                    row["x_13_over_11"] = round(row["g11x10"] / row["g13x10"], 3)
                    row["saved_ms_step"] = round((row["g11x10"] - row["g13x10"]) * per_step, 3)
                    saved[B] = saved.get(B, 0.0) + row["saved_ms_step"]
            for o in outs.values():
                ttnn.deallocate(o)
            ttnn.deallocate(x)
            ttnn.deallocate(w)
            print("ROW " + json.dumps(row), flush=True)

    print("SUMMARY " + json.dumps({
        "all_shapes_bit_exact_13_vs_11": exact_all,
        "saved_ms_per_step": {f"D{k}": round(v, 3) for k, v in saved.items()},
    }), flush=True)


if __name__ == "__main__":
    main()
