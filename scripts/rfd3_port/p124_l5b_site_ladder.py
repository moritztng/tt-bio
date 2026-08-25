#!/usr/bin/env python3
"""p124 -- where L5b has a site at all, and where it can be bit-exact. No device needed.

L5b fuses the PV matmul into the landed softmax kernel. Two separate size conditions decide
whether that is even possible at a given atom count, and neither was in the plan:

1. **The kernel has to engage.** `softmax_generic.eligible` returns the large-kernel plan only
   when the CB estimate trips `use_large`. Below that width the fold runs the shipped
   `softmax` + `typecast` pair and there is no fused kernel to extend, so L5b has no site.

2. **The two K-groupings have to agree.** The softmax kernel's normalize pass walks the row in
   `blk = find_max_divisor(Wt, 4)` tiles. The shipped PV matmul accumulates K in
   `in0_block_w = 2 if k_tiles % 2 == 0 else 1`. K-blocking is the only thing that regroups the
   fp32 accumulation (`_attn_value_program_config`), so the fused kernel reproduces the shipped
   arithmetic only where `blk == in0_block_w`. `state/rfd3-fusion-programme.md` §9.2 found these
   equal at the R4 fixture and called it a coincidence rather than a design. This enumerates the
   coincidence.

Everything here is the arithmetic of the shipped functions, imported where it is importable and
transcribed where importing it would need a device. No measurement, so nothing here is a screen
result and nothing here needs a card.
"""
import json
import pathlib
import sys

TILE = 32
L1_PER_CORE = 1499136          # softmax_generic.L1_PER_CORE
TILE_BYTES_FP32 = 4096         # softmax_generic._TILE_BYTES[float32]


def find_max_divisor(val, start_max_div):
    """tt_metal::find_max_divisor. It skips 7 and 5. (softmax_generic.find_max_divisor)"""
    for d in range(start_max_div, 0, -1):
        if d in (7, 5):
            continue
        if val % d == 0:
            return d
    return 1


def use_large(Wt):
    """softmax_generic.plan()'s `use_large` for a rank-4 fp32 input with fp32_dest_acc and
    numeric_stable, both of which the RFD3 atom site passes. cb_sum reduces to a function of Wt
    alone because in0_t == im0_t == im4_t == Wt there."""
    bs = find_max_divisor(Wt, 4)
    cb_sum = TILE_BYTES_FP32 * (3 * Wt + 2 * bs + 5)
    return (L1_PER_CORE * 0.9) < cb_sum, cb_sum, bs


def in0_block_w(k_tiles):
    """tenstorrent._attn_value_program_config's K-blocking."""
    return 2 if k_tiles % 2 == 0 else 1


def pv_config_exists(m_tiles, n_tiles):
    """_attn_value_program_config returns None outside its 1D-mirror branch, and then the PV
    matmul runs on ttnn's heuristic with no pinned blocking for a kernel to mirror. The atom
    site's head_dim is 32, so n_tiles is 1 and this only fails for very small m."""
    return not (n_tiles > 8 or m_tiles <= 8 or n_tiles >= m_tiles)


def classify(atoms):
    """`atoms` may be a raw count or an already-padded axis; padding is idempotent either way."""
    Wt = (-(-atoms // TILE) * TILE) // TILE          # _align_tile(atoms) / 32
    large, cb_sum, blk = use_large(Wt)
    ibw = in0_block_w(Wt)                            # PV's K axis IS the padded atom key axis
    m_tiles = -(-atoms // TILE)
    pinned = pv_config_exists(m_tiles, 1)            # head_dim 32 -> n_tiles 1
    return {"atoms": atoms, "key_width": Wt * TILE, "Wt": Wt,
            "cb_sum_kb": round(cb_sum / 1024), "kernel_engages": large,
            "pv_config_pinned": pinned,
            "blk": blk, "in0_block_w": ibw,
            "l5b_bit_exact": bool(large and pinned and blk == ibw)}


def main():
    # The ds-fix ladder. Two rungs have a MEASURED padded atom axis and are used as such: R4 is
    # 6080 (the census fixture) and R3 is 4576 (p122 on pc, perf/p122/atom_softmax_wall_R3.json).
    # The rest are scaled from R4's 685 tokens / 6051 atoms and are estimates, which matters
    # because the verdict turns on Wt's divisibility and scaling can land on the wrong side of it.
    ladder = [("R0", 217, None), ("R1", 296, None), ("R2", 418, None),
              ("R3", 514, 4576), ("R4", 685, 6080)]
    rungs = []
    for name, tok, measured_w in ladder:
        r = classify(measured_w if measured_w else round(6051 * tok / 685))
        r["rung"], r["tokens"] = name, tok
        r["width_source"] = "measured" if measured_w else "scaled from R4"
        rungs.append(r)

    smallest = next((Wt for Wt in range(1, 4096) if use_large(Wt)[0]), None)

    # How much of the size axis L5b can serve, over every width where the kernel engages at all.
    hi = 4096
    eng = [Wt for Wt in range(smallest, hi) if use_large(Wt)[0]]
    # Every engaged width has m_tiles >= 108 and n_tiles == 1, so the PV config is pinned
    # everywhere the kernel engages; asserted rather than assumed.
    assert all(pv_config_exists(Wt, 1) for Wt in eng)
    ok = [Wt for Wt in eng if find_max_divisor(Wt, 4) == in0_block_w(Wt)]
    declines = {}
    for Wt in eng:
        if find_max_divisor(Wt, 4) != in0_block_w(Wt):
            declines.setdefault("blk=%d vs in0_block_w=%d"
                                % (find_max_divisor(Wt, 4), in0_block_w(Wt)), 0)
            declines["blk=%d vs in0_block_w=%d"
                     % (find_max_divisor(Wt, 4), in0_block_w(Wt))] += 1

    res = {"ladder": rungs,
           "kernel_engages_above_key_width": smallest * TILE,
           "kernel_engages_above_atoms": (smallest - 1) * TILE + 1,
           "widths_examined": len(eng),
           "widths_l5b_bit_exact": len(ok),
           "l5b_bit_exact_fraction": round(len(ok) / len(eng), 4),
           "decline_reasons": declines,
           "rule": "blk == in0_block_w iff Wt is not divisible by 4 and not divisible by 3"}

    print("%-5s %-7s %-6s %-5s %-9s %-5s %-11s %-11s %s"
          % ("rung", "tokens", "keyW", "Wt", "kernel?", "blk", "in0_block_w", "L5b exact?", "width"))
    for r in rungs:
        print("%-5s %-7d %-6d %-5d %-9s %-5d %-11d %-11s %s"
              % (r["rung"], r["tokens"], r["key_width"], r["Wt"],
                 r["kernel_engages"], r["blk"], r["in0_block_w"], r["l5b_bit_exact"],
                 r["width_source"]))
    print("\nthe fused softmax kernel engages only above key width %d (about %d atoms); below it "
          "the fold runs the shipped softmax+typecast pair and L5b has NO SITE."
          % (res["kernel_engages_above_key_width"], res["kernel_engages_above_atoms"]))
    print("over the %d engaged widths up to %d, L5b can be bit-exact on %d of them (%.1f %%)."
          % (len(eng), hi * TILE, len(ok), 100 * res["l5b_bit_exact_fraction"]))
    for k, v in sorted(declines.items(), key=lambda kv: -kv[1]):
        print("  declines, %-24s %5d widths" % (k, v))
    print("\nrule: %s" % res["rule"])

    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p124/l5b_site_ladder.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
