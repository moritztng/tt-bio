#!/usr/bin/env python3
"""What the shape-keyed dispatch picks at padded width 64 and at 384, read off the shipped
functions themselves rather than inferred from a fold.

Pure lookup: every function here is an `lru_cache`d pure function of shapes, so this records the
SAME pick the fold will make, before any device arm runs. It does open a device, because
`_fp32_softmax_core_budget` and `COMPUTE_GRID_MAIN` read the real core grid.

Shapes come from `openfold3_trunk.py` `_PF_DIMS = (32, 4, 24, 16)`:
  pair track   TriangleAttention   head_dim 32, 4 heads,  q [N, 4, N, 32] (batch = seq)
  single track AttentionPairBias   head_dim 24, 16 heads, q [1, 16, N, 24]
"""
import json
import sys

import tt_bio.tenstorrent as T


def probe(track, rows, heads, N, d, dv):
    q_len = k_len = N
    hpr = heads * N
    per_row = hpr * k_len * 4
    blk = max(32, int(T._FP32_SOFTMAX_BLOCK_BYTES // per_row) // 32 * 32) if per_row else rows
    bmm = (heads, -(-q_len // 32), -(-d // 32), -(-k_len // 32), -(-dv // 32))
    cap = T._FP32_SOFTMAX_L1_ROW_CAP.get((hpr, k_len))
    free_cap = T._FP32_SOFTMAX_L1_FREE_ROW_CAP.get((hpr, k_len))
    tuned = T._fp32_softmax_l1_rows(per_row, hpr, cap)
    rows_p, cores_p = T._fp32_softmax_l1_plan(per_row, hpr, k_len, cap, bmm, free_cap)
    fin = min(blk, rows_p) if rows_p else blk
    return {
        "track": track, "N": N, "leading_rows": rows, "heads": heads, "head_dim": d,
        "height_per_row": hpr, "per_row_bytes": per_row, "block_byte_budget_rows": blk,
        "row_cap": cap, "free_row_cap": free_cap, "tuned_rows": tuned,
        "plan_rows": rows_p, "plan_cores": cores_p, "block_rows_final": fin,
        "blocks_per_call": -(-rows // fin),
        "shard_buildable": T._fp32_softmax_shard(fin, hpr, k_len, cores_p) is not None,
        "l1_resident": bool(rows_p),
    }


def main():
    out = {
        "what": "the shape-keyed picks at padded width 64 and 384, read from the shipped "
                "functions before any device arm",
        "constants": {
            "TRIATT_FUSED_HIFI_MIN_S": T._TRIATT_FUSED_HIFI_MIN_S,
            "TRIATT_HIFI_MIN_S_PADDED": T._TRIATT_HIFI_MIN_S_PADDED,
            "FP32_SOFTMAX_L1_GRID": list(T._FP32_SOFTMAX_L1_GRID),
            "FP32_SOFTMAX_L1_BYTES_PER_CORE": T._FP32_SOFTMAX_L1_BYTES_PER_CORE,
            "FP32_SOFTMAX_BLOCK_BYTES": T._FP32_SOFTMAX_BLOCK_BYTES,
            "FP32_SOFTMAX_L1_ANY_CORES": T._FP32_SOFTMAX_L1_ANY_CORES,
            "FP32_SOFTMAX_L1_FLOAT_CORES": T._FP32_SOFTMAX_L1_FLOAT_CORES,
            "fp32_softmax_core_budget": T._fp32_softmax_core_budget(),
            "COMPUTE_GRID_MAIN": list(T.COMPUTE_GRID_MAIN),
            "SDPA_BAND_DIV_K": T._SDPA_BAND_DIV_K,
        },
        "single_track": [probe("single/AttentionPairBias", 1, 16, n, 24, 24) for n in (64, 384)],
        "pair_track_materialised": [probe("pair/TriangleAttention", n, 4, n, 32, 32)
                                    for n in (64, 384)],
        "pair_track_fused": [{
            "N": n,
            "site_flag_openfold3_trunk": T.triatt_sdpa_hifi_site("openfold3.trunk", True),
            "fused_hifi_on": T._fused_hifi_on(True),
            "min_s_gate_declines": min(n, n) < T._TRIATT_FUSED_HIFI_MIN_S,
            "padded_len": T._padded_sdpa_len(n),
            "chunks_shipped_q_k": list(T._sdpa_chunks_shipped(n, n)),
            "q_chunk_ladder": list(T._tri_att_q_chunks(n, n)),
        } for n in (64, 384)],
    }
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
