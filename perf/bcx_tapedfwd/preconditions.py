#!/usr/bin/env python3
"""Why the fused triangle-attention route declines at 288, the length BindCraft 2 actually runs.

Host only, no device. Replays the shipped precondition set -- `sdpa_generic.plan`, which is the
same function `triatt_sdpa.sdpa` calls -- over the shipped chunk ladders at each length, and
prints which of the hoisted fill's six preconditions fails.

The answer is one line of arithmetic: at 288 the shipped q_chunk is 64 and 288 % 64 = 32. A
q_chunk that does not divide the padded sequence sets `use_padded_mask`, and that is a
precondition of the hoisted fill, so the fused path declines outright. 288 = 2^5 * 9 and 64 = 2^6,
so 64 cannot divide it; 256, 320 and 384 are all divided by their shipped q_chunk exactly.

    preconditions.py --ns 192,224,256,288,320,384 --grids 11x10,13x10,8x8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_bio import sdpa_generic as SG, tenstorrent as T, triatt_sdpa as TS

HEADS, HEAD_DIM = 4, 32
NAMES = ("nh_per_core", "q_per_core", "bcast_batch", "use_padded_mask", "NKH", "NVH")


def failures(S, qc, kc, grid):
    """The hoisted fill's preconditions that FAIL for this config, by name."""
    cores = grid[0] * grid[1]
    qpf = TS.q_parallel_factor(S, HEADS, qc, cores)
    split = (cores // (HEADS * qpf), HEADS, qpf)
    if split[0] < 1:
        return ["split_invalid"], None
    p = SG.plan_for_shape(S, HEADS, HEAD_DIM, qc, kc, grid=grid, split=split)
    bad = [n for n, ok in (("nh_per_core", p["nh_per_core"] == 1),
                           ("q_per_core", p["q_per_core"] == 1),
                           ("bcast_batch", p["bcast_batch"]),
                           ("use_padded_mask", not p["use_padded_mask"]),
                           ("NKH", p["NKH"] == HEADS), ("NVH", p["NVH"] == HEADS)) if not ok]
    return bad, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="192,224,256,288,320,384")
    ap.add_argument("--grids", default="11x10,13x10,8x8")
    ap.add_argument("--out", default="out/preconditions.json")
    args = ap.parse_args()

    grids = [tuple(int(x) for x in g.split("x")) for g in args.grids.split(",")]
    rows = []
    for S in [int(x) for x in args.ns.split(",")]:
        sq, sk = T._sdpa_chunks_shipped(S, S)[:2]
        row = {"n": S, "shipped_q_chunk": sq, "shipped_k_chunk": sk,
               "shipped_q_divides": S % sq == 0,
               "q_chunks": list(T._tri_att_q_chunks(S, S)),
               "k_chunks": list(T._tri_att_k_chunks(S, S)), "grids": {}}
        for g in grids:
            ok, bad = [], {}
            for kc in row["k_chunks"]:
                for qc in row["q_chunks"]:
                    f, _ = failures(S, qc, kc, g)
                    (ok.append([qc, kc]) if not f
                     else bad.setdefault(",".join(f), []).append([qc, kc]))
            row["grids"][f"{g[0]}x{g[1]}"] = {"fused_ok": ok, "declined": bad}
        rows.append(row)
        gk = f"{grids[0][0]}x{grids[0][1]}"
        print(f"n={S:4d} shipped q={sq:4d} k={sk:4d} divides={row['shipped_q_divides']!s:5} "
              f"fused-ok on {gk}: {len(row['grids'][gk]['fused_ok'])}")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"heads": HEADS, "head_dim": HEAD_DIM, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
