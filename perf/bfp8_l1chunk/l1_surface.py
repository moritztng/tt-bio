"""Which (q_chunk, k_chunk) the persistent-mask SDPA fits in L1, bf16 against uniform bfp8.

Host only: `cb_bytes` is arithmetic over `plan`, so the whole surface is priced without a device.
The question this answers is the row s premise -- bfp8 halves the operand tile, so it changes WHAT
FITS, not just how many bytes move. Output feeds the CHUNK-LADDER line of the state doc.
"""
import json
import sys

import ttnn

import tt_bio.sdpa_generic as SG
import tt_bio.triatt_sdpa as TS

CORES = 110                      # qb2 p300c, grid 11x10
HEADS, HEAD_DIM = 4, 32          # boltz-2 triangle attention


def surface(seq, dtype, cores=CORES, heads=HEADS, head_dim=HEAD_DIM):
    """Every dividing (q_chunk, k_chunk) with its per-core CB bytes and whether it fits."""
    rows = []
    for qc in SG.chunk_divisors(seq):
        for kc in SG.chunk_divisors(seq):
            q_pf = TS.q_parallel_factor(seq, heads, qc, cores, cap=0)
            p = SG.plan_for_shape(seq, heads, head_dim, qc, kc, grid=(cores, 1), split=(
                max(cores // (heads * q_pf), 1), heads, q_pf), dtype=dtype)
            pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
            # every operand CB priced at the real dtype, which is what `fused_pairs` does NOT do
            b = SG.cb_bytes(p, q_dtype=dtype, k_dtype=dtype, v_dtype=dtype,
                            mask_dtype=dtype, out_dtype=dtype, mask_cb_tiles=pers)
            rows.append({
                "q_chunk": qc, "k_chunk": kc, "cb_bytes": b,
                "fits": b + SG.PROGRAM_RESERVE <= SG.L1_PER_CORE,
                "preconds_ok": (p["q_per_core"] == 1 and p["nh_per_core"] == 1
                                and not p["use_padded_mask"]),
                "cost": TS.per_core_cost(p, qc, seq),
            })
    return rows


def servable(rows):
    return sorted([r for r in rows if r["fits"] and r["preconds_ok"]], key=lambda r: r["cost"])


def main():
    seqs = [int(x) for x in sys.argv[1:]] or [512, 640, 768, 896, 1024, 1280, 1536]
    out = {"cores": CORES, "heads": HEADS, "head_dim": HEAD_DIM,
           "L1_PER_CORE": SG.L1_PER_CORE, "PROGRAM_RESERVE": SG.PROGRAM_RESERVE, "seqs": {}}
    for seq in seqs:
        b16 = surface(seq, ttnn.bfloat16)
        b8 = surface(seq, ttnn.bfloat8_b)
        s16, s8 = servable(b16), servable(b8)
        by16 = {(r["q_chunk"], r["k_chunk"]): r for r in b16}
        gained = [(r["q_chunk"], r["k_chunk"]) for r in s8
                  if not by16[(r["q_chunk"], r["k_chunk"])]["fits"]]
        out["seqs"][seq] = {
            "n_servable_bf16": len(s16), "n_servable_bfp8": len(s8),
            "best_bf16": (s16[0]["q_chunk"], s16[0]["k_chunk"]) if s16 else None,
            "best_bfp8": (s8[0]["q_chunk"], s8[0]["k_chunk"]) if s8 else None,
            "cost_bf16": s16[0]["cost"] if s16 else None,
            "cost_bfp8": s8[0]["cost"] if s8 else None,
            "newly_fitting": gained,
            "rows_bf16": b16, "rows_bfp8": b8,
        }
        r = out["seqs"][seq]
        ratio = (r["cost_bf16"] / r["cost_bfp8"]) if (r["cost_bf16"] and r["cost_bfp8"]) else None
        print(f"seq {seq:5d}  servable bf16 {len(s16):3d} -> bfp8 {len(s8):3d}   "
              f"best bf16 {r['best_bf16']} bfp8 {r['best_bfp8']}   "
              f"cost ratio {ratio if ratio is None else round(ratio, 4)}   "
              f"newly fitting {len(gained)}")
    with open("perf/bfp8_l1chunk/l1_surface.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
