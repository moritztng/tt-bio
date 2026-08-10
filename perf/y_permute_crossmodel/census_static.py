#!/usr/bin/env python3
"""Deliverable 1a — the shape-window census, read off origin/main, on qb2's own 11x10 grid.

`_channel_move` (tt_bio/tenstorrent.py:1474) is reached from exactly two call sites, both inside
`TriangleMultiplication._transform_chunk` (1561, 1564). Every model that reaches the kernel does so
through that one class, so eligibility is a pure function of (pair-track N, compute grid, fast mode)
plus the trimul chunk width the grid picks. This script evaluates the SHIPPED gate against REAL
tensors at every N a production input can land on, so the layout/dtype/interleaved clauses are
exercised and not merely reasoned about.

    TT_VISIBLE_DEVICES=2 python3 perf/y_permute_crossmodel/census_static.py
"""
from __future__ import annotations

import importlib.metadata as md
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

OUT = Path(__file__).resolve().parent


def main() -> int:
    import tt_bio.tenstorrent as T
    from tt_bio import reblock_permute as RP

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    RP.set_enabled(True)

    R = {
        "wheel": md.version("ttnn"), "host": "qb2", "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": [g.x, g.y], "cores": g.x * g.y,
        "COMPUTE_GRID_MAIN": list(T.COMPUTE_GRID_MAIN),
        "TRIANGLE_MULT_L1_MAX_SEQ": T.TRIANGLE_MULT_L1_MAX_SEQ,
        "TRIANGLE_MULT_L1_MAX_SEQ_FAST": T.TRIANGLE_MULT_L1_MAX_SEQ_FAST,
        "TRIANGLE_MULT_L1_MAX_SEQ_FAST_13X10": T.TRIANGLE_MULT_L1_MAX_SEQ_FAST_13X10,
        "TRIANGLE_MULT_CHUNK_SIZE": T.TRIANGLE_MULT_CHUNK_SIZE,
        "TRIANGLE_MULT_L1_CHUNK_BUDGET": T.TRIANGLE_MULT_L1_CHUNK_BUDGET,
        "rows": [],
    }
    print(json.dumps({k: v for k, v in R.items() if k != "rows"}, indent=1), flush=True)

    HIDDEN = 128  # c_z=256 trimul hidden; captured live per model in the fold runs
    NS = [64, 117, 128, 222, 256, 287, 288, 298, 320, 352, 353, 384, 448, 512, 566, 640, 704,
          768, 891, 1024, 1095, 1690]

    for fast in (False, True):
        T._FAST_MODE = fast
        for N in NS:
            l1max = T._trimul_l1_max_seq()
            mc = T._triangle_mul_memory_config(N)
            C = T._trimul_chunk_size(N, HIDDEN)
            bt = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
            nbytes = N * N * C * 2
            probe_mc = ttnn.DRAM_MEMORY_CONFIG if (bt == "L1" and nbytes > 40e6) else mc
            note = "" if probe_mc is mc else "in staged DRAM (too big for L1 probe)"
            try:
                x = ttnn.from_torch(torch.zeros(1, N, N, C, dtype=torch.bfloat16),
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=probe_mc)
                elig = bool(RP.eligible(x, mc))
                ttnn.deallocate(x)
            except Exception as e:
                elig, note = None, "alloc failed: " + type(e).__name__
            row = {
                "fast": fast, "N": N, "l1max": l1max, "out_buffer": bt, "chunk_C": C,
                "n_pairs": HIDDEN // C // 2,
                "channel_moves_per_chunk_pair": 2 if bt == "DRAM" else 1,
                "eligible": elig, "note": note,
            }
            R["rows"].append(row)
            print("fast=%d N=%5d l1max=%4d out=%-4s C=%3d n_pairs=%2d moves/pair=%d eligible=%s %s"
                  % (int(fast), N, l1max, bt, C, row["n_pairs"],
                     row["channel_moves_per_chunk_pair"], elig, note), flush=True)
    T._FAST_MODE = False
    (OUT / "census_static.json").write_text(json.dumps(R, indent=1))
    print("wrote", OUT / "census_static.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
