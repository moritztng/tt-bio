"""Why `triatt_sdpa.fused_pairs` is empty at OpenDDE's 1536-residue refiner width, by arithmetic.

No device: the plan and CB model are the ones `fused_pairs` itself uses. For each 32-aligned
(q_chunk, k_chunk) at padded S, records the gate that refuses it and, for the L1 gate, the CB total
against the budget and the persistent mask's share.

    python perf/mgx_wide_seq/fused_reach_3008.py [S ...]
"""
import json
import sys

import ttnn

from tt_bio import sdpa_generic as SG
from tt_bio import triatt_sdpa as TS

CORES, H, D = 72, 12, 32            # whglx 8x9 grid; OpenDDE refiner: c_z 384 / 12 heads


def reach(seq):
    rows = []
    for kc in SG.chunk_divisors(seq):
        for qc in SG.chunk_divisors(seq):
            q_pf = TS.q_parallel_factor(seq, H, qc, CORES, cap=0)
            p = SG.plan_for_shape(seq, H, D, qc, kc, grid=(CORES, 1), split=(
                max(CORES // (H * q_pf), 1), H, q_pf), dtype=ttnn.bfloat16)
            r = {"q": qc, "k": kc, "q_per_core": p["q_per_core"]}
            if p["q_per_core"] != 1 or p["nh_per_core"] != 1 or p["use_padded_mask"]:
                r["why"] = "fill_preconditions"
            else:
                pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
                tot = SG.cb_bytes(p, mask_cb_tiles=pers)
                r.update(why="fits" if SG.cb_fits_l1(p, mask_cb_tiles=pers) else "l1",
                         cb_bytes=tot, mask_bytes=pers * 2048, budget=SG.L1_PER_CORE)
            rows.append(r)
    return rows


for s in [int(x) for x in sys.argv[1:]] or [3008]:
    rows = reach(s)
    print(json.dumps({"S": s, "divisors": SG.chunk_divisors(s),
                      "fused_pairs": TS.fused_pairs(s, H, D, CORES, ttnn.bfloat16),
                      "rows": rows}))
