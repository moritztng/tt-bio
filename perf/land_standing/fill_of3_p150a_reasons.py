#!/usr/bin/env python3
"""Write the 22 dark-lever exemption reasons the openfold3 p150a re-record left as TODOs.

Every reason below is derived from source or from a counter in the same baseline, never from a
story. The derivations:

  SDPA_WIDE_K          `_dividing_k_chunks(S, S)` returns a single entry at all ten rungs, so
                       `_tri_att_sdpa_at`'s `if len(k_chunks) > 1` block, which holds both counter
                       sites, is never entered. Computed this pass against the shipped picks.
  TRANSPOSE_L1_RESIDENT  the 1.25 x bytes <= 110 x 1532416 gate, evaluated on the two pair shapes
                       openfold3 presents ([S,S,128] x424 and [S,S,64] x16), reproduces the served
                       count at all ten rungs.
  TRIATT_SDPA_HIFI     the ladder in `_tri_att_sdpa_hifi_inner` is exhausted; the clauses are in
                       `triatt_sdpa.REJECTS` on the same run, reported under TRIATT_PERSISTENT_MASK.
  TRIMUL_TAIL_F1       the k_tiles clause, already written at the lower rungs, restated with this
                       rung's own counts.
"""
import json
import pathlib

FRAG = pathlib.Path(__file__).resolve().parents[2] / "docs" / "size_ladder_baseline.d" / "openfold3.json"

SHIPPED_K = {256: 256, 512: 256, 640: 160, 768: 256, 896: 224,
             1024: 256, 1152: 192, 1280: 256, 1408: 128, 1536: 256}

# rung -> (persistent-mask decline clauses, fp32-softmax L1 grid serves) at that rung
HIFI = {1152: ("fill_preconditions x10, l1_budget x3", 168960),
        1280: ("fill_preconditions x9, l1_budget x3", 561921),
        1408: ("fill_preconditions x7, l1_budget x3", 618113),
        1536: ("fill_preconditions x12, l1_budget x3", 675840)}

# rung -> the transpose class that leaves L1 there, and the bytes it asks for
TRANSPOSE = ("L1 capacity, computed rather than argued: the gate is 1.25 x bytes <= 110 x 1532416 "
             "= 168.57 MB, and openfold3 presents two pair shapes, [S,S,128] on 424 calls and "
             "[S,S,64] on 16 (the split PAIR_TRANSPOSE_VIA_ROW_MAJOR's own reject dict records at "
             "512 aa). The 128-wide class leaves L1 at 768 aa asking 188.7 MB; the 64-wide class "
             "leaves it here, asking %.1f MB against 167.8 MB at 1024 aa, which fits with 0.5 %% "
             "to spare. That model reproduces the served count at all ten rungs "
             "(440/440/440/16/16/16/0/0/0/0), so every call at this size is on the DRAM route by "
             "capacity, and PAIR_TRANSPOSE_VIA_ROW_MAJOR picks up all 440 at the same rung.")

TRIMUL = ("declines all %d calls on %s: F1_BLOCK_KEYS allow-lists only (8, 8) and openfold3's "
          "trimul tails resolve 4 and 2 K tiles, which are properties of c_hidden and not of N, "
          "so F1 is inert on this model at every size and not dark at this one.")

WIDE_K = ("structural, not size-specific: both of this lever's counter sites live inside "
          "`_tri_att_sdpa_at`'s `if len(k_chunks) > 1` block, and `_dividing_k_chunks` returns a "
          "single entry here because the shipped k of %d divides the padded length of %d, so K5 "
          "has no pick to change. The one other serve site is the above-cap fused route, which "
          "SDPA_FUSED_LARGE_S records as 0 served / 0 declined at the same rung. OpenFold3 barely "
          "reaches this function at all: all four of its tri-att sites pass fp32_softmax=True, and "
          "that branch of `_attend_heads` routes to `_tri_att_sdpa_hifi` or to "
          "`_fp32_softmax_attention`.")

HIFI_REASON = ("the fused route's L1 ceiling, which is per-core and not N. "
               "`_tri_att_sdpa_hifi_inner` exhausts its ladder and returns None, so all 384 calls "
               "fall back to `_fp32_softmax_attention`, which is why FP32_SOFTMAX_L1_GRID reads "
               "%d serves at this rung against 19096 at 1024 aa. The clauses are in "
               "`triatt_sdpa.REJECTS` on this same run, reported under TRIATT_PERSISTENT_MASK: "
               "%s. `l1_budget` is the device's own allocator throw absorbed at "
               "triatt_sdpa.py:451, and once those configs are retired into `_TRIATT_HIFI_OVER_L1` "
               "the remaining calls never re-enter. The boundary is measured, not modelled: 384 "
               "served at 1024 aa and every call declined at 1088 aa on a p300c (56ad6c0e0), and "
               "this ladder reproduces the pre-HiFi baseline within 3 %% at 1152-1536 while "
               "beating it 2.33x at 1024.")


def main():
    frag = json.loads(FRAG.read_text())
    levers = frag["cards"]["p150a"]["models"]["openfold3"]["levers"]
    written = 0
    for rung_s, block in levers.items():
        rung = int(rung_s)
        for flag, entry in block.items():
            reason = entry.get("reason", "")
            if not isinstance(reason, str) or not reason.startswith("TODO"):
                continue
            if flag == "SDPA_WIDE_K":
                entry["reason"] = WIDE_K % (SHIPPED_K[rung], rung)
            elif flag == "TRANSPOSE_L1_RESIDENT":
                entry["reason"] = TRANSPOSE % (rung * rung * 64 * 2 * 1.25 / 1e6)
            elif flag == "TRIATT_SDPA_HIFI":
                clauses, grid = HIFI[rung]
                entry["reason"] = HIFI_REASON % (grid, clauses)
            elif flag == "TRIMUL_TAIL_F1":
                rej = entry.get("rejects") or {}
                shape = ", ".join(f"{k} x{v}" for k, v in sorted(rej.items()))
                entry["reason"] = TRIMUL % (entry["declined"], shape)
            else:
                raise SystemExit(f"unhandled dark lever {flag} at rung {rung}")
            written += 1
    FRAG.write_text(json.dumps(frag, indent=2) + "\n")
    print(f"wrote {written} exemption reasons to {FRAG}")


if __name__ == "__main__":
    main()
