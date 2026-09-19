#!/usr/bin/env python3
"""How many configs does fused_pairs' bf16 pricing of q/k/v/out cost bfp8?

fused_pairs calls cb_fits_l1(p, mask_cb_tiles=pers, mask_dtype=dt) and lets q/k/v/out fall back
to cb_bytes' bfloat16 defaults, so under uniform bfp8 four of the five operand CBs are priced at
twice their real page size. This compares the three pricings.
"""
import json

import ttnn

import tt_bio.sdpa_generic as SG
import tt_bio.triatt_sdpa as TS

CORES, HEADS, HEAD_DIM = 110, 4, 32


def count(seq, dt, all_operands):
    n = 0
    for qc in SG.chunk_divisors(seq):
        for kc in SG.chunk_divisors(seq):
            q_pf = TS.q_parallel_factor(seq, HEADS, qc, CORES, cap=0)
            p = SG.plan_for_shape(seq, HEADS, HEAD_DIM, qc, kc, grid=(CORES, 1), split=(
                max(CORES // (HEADS * q_pf), 1), HEADS, q_pf), dtype=dt)
            if p["q_per_core"] != 1 or p["nh_per_core"] != 1 or p["use_padded_mask"]:
                continue
            pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
            kw = {"mask_dtype": dt}
            if all_operands:
                kw.update(q_dtype=dt, k_dtype=dt, v_dtype=dt, out_dtype=dt)
            if SG.cb_fits_l1(p, mask_cb_tiles=pers, **kw):
                n += 1
    return n


def main():
    res = {}
    print(" seq |  bf16 | bfp8 as fused_pairs prices it | bfp8 priced correctly | configs lost")
    for seq in (512, 640, 768, 896, 1024, 1280, 1536, 2048):
        b16 = count(seq, ttnn.bfloat16, True)
        buggy = count(seq, ttnn.bfloat8_b, False)
        fixed = count(seq, ttnn.bfloat8_b, True)
        res[seq] = {"bf16": b16, "bfp8_buggy": buggy, "bfp8_fixed": fixed,
                    "lost": fixed - buggy}
        print(f"{seq:5d} | {b16:5d} | {buggy:29d} | {fixed:21d} | {fixed - buggy:12d}")
    with open("perf/bfp8_l1chunk/fused_pairs_misprice.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
