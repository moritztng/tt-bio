#!/usr/bin/env python3
"""C12 pass 23: what kblock's witnessed `mt_total` 752 and 672 actually are.

Pass 18 concluded the pair track is row-blocked into "ten chunks of 47 rows plus one of 42"
at 512 aa, with heights 64 at 298 aa and 32 at 768. Pass 21 disputed it: evaluating the shipped
`tenstorrent.py:8336` Transition expression gives `h_chunk` 16 and `mt_total` 256 at 512 aa,
the census label to the digit, so it marked the finding DISPUTED and made settling it a required
report from `c12-profiled-fold`. That row concluded WITHOUT the report, so the dispute is still
open and this closes as much of it as source arithmetic can.

The evidence in tension:
  kblock `_linear_block_cfg` witness   mt_total 752 and 672, "no call in the fold has 256"
  silu instrumented fold              3,762 fused-silu calls where the census books 8,960
  pass 21 source evaluation           h_chunk 16 -> mt_total 256

Run: python3 witness_arithmetic.py
"""
import math

W_BY_AA = {298: 320, 512: 512, 768: 768}
TILE = 32
GRID_X, GRID_Y = 11, 10          # p300c compute grid; 110 cores, confirmed independently by
                                 # c12-genericop-rate reading 110/110 on all 336 generic_op programs
WITNESSED_MT = (752, 672)        # c12-kblock-unlock's _linear_block_cfg witness
CENSUS_MT = 256                  # the census key label, and pass 21's source evaluation
PASS18_HEIGHTS = {298: 64, 512: 47, 768: 32}
# tt_bio/reblock_permute.py docstring: "298 aa folds with C = 64 on a 13x10 grid and with C = 32
# on an 11x10 one" -- the CHANNEL count, not a row-block height.
REBLOCK_DOC_C = (64, 32)


def split(w, parts):
    """`parts`-way split of `w` rows: chunk size, count, and the ragged last chunk."""
    n = math.ceil(w / parts)
    count = math.ceil(w / n)
    return n, count, w - (count - 1) * n


def main():
    print("1. WHAT 752 AND 672 ARE, IF mt_total IS A FLATTENED h x W IN TILES")
    for mt in WITNESSED_MT:
        rows = mt * TILE
        h = rows / W_BY_AA[512]
        print(f"   mt_total {mt:4d} -> {rows:6d} rows -> h = {rows}/{W_BY_AA[512]} = {h:g}"
              f"{'  INTEGER' if h == int(h) else ''}")
    n, count, last = split(W_BY_AA[512], GRID_X)
    print(f"\n   an {GRID_X}-way split of W={W_BY_AA[512]}: chunk {n}, count {count}, last {last}")
    print(f"   -> {count - 1} x {n} + {last} = {(count-1)*n + last} rows, and W = {W_BY_AA[512]}"
          f"  {'EXACT PARTITION' if (count-1)*n + last == W_BY_AA[512] else 'MISMATCH'}")
    print(f"   in tiles: {n} x {W_BY_AA[512]} / {TILE} = {n*W_BY_AA[512]//TILE}, "
          f"{last} x {W_BY_AA[512]} / {TILE} = {last*W_BY_AA[512]//TILE}")
    hit = (n*W_BY_AA[512]//TILE, last*W_BY_AA[512]//TILE) == WITNESSED_MT
    print(f"   witnessed {WITNESSED_MT}  -> {'MATCHES TO THE DIGIT' if hit else 'does not match'}")

    print(f"\n2. DOES THE SAME MECHANISM REPRODUCE PASS 18'S OTHER TWO SIZES?")
    print(f"   {'aa':>5} {'W':>5} {'11-way':>7} {'13-way':>7} {'pass18':>7}  verdict")
    for aa, w in W_BY_AA.items():
        n11 = split(w, 11)[0]
        n13 = split(w, 13)[0]
        p18 = PASS18_HEIGHTS[aa]
        ok = "MATCHES 11-way" if n11 == p18 else ("matches 13-way" if n13 == p18 else "NEITHER")
        print(f"   {aa:5d} {w:5d} {n11:7d} {n13:7d} {p18:7d}  {ok}")
    print(f"   pass 18's 298/768 heights are {PASS18_HEIGHTS[298]} and {PASS18_HEIGHTS[768]};")
    print(f"   reblock_permute's docstring documents C = {REBLOCK_DOC_C[0]} and "
          f"{REBLOCK_DOC_C[1]} as CHANNEL counts at 298 aa.")
    print(f"   Those are the same two numbers, so a height/channel conflation is the likely source.")

    print(f"\n3. WHAT row_block() CANNOT DO")
    print(f"   tt_bio/tenstorrent.py:5209 `row_block()` returns `(budget // per_row) // 32 * 32`,")
    print(f"   floored at 32 -- 32-ALIGNED BY CONSTRUCTION. 47 and 42 are not multiples of 32,")
    print(f"   so whatever produced the witnessed values, it is not row_block().")

    print(f"\n4. WHAT THIS SETTLES AND WHAT IT DOES NOT")
    print(f"   SETTLED: the witnessed mt_total pair is an exact {GRID_X}-way row partition of")
    print(f"            W=512 (47 x10 + 42), matching BOTH witnessed values to the digit. A")
    print(f"            16-row chunk divides 512 evenly into 32 and can produce no ragged 42,")
    print(f"            so pass 21's h_chunk=16 expression is NOT the site kblock witnessed and")
    print(f"            its evaluation therefore did not refute pass 18's 512 aa claim.")
    print(f"   SETTLED: pass 18's 298 aa and 768 aa heights are unsupported by this mechanism.")
    print(f"   OPEN:    which source line performs the {GRID_X}-way split. It is not row_block()")
    print(f"            (32-aligned) and not _triangle_mul_program_config (divides TILES by gx/gy,")
    print(f"            giving per_core_N = ceil(16/11) = 2 at 512 aa, not 47). Both the census")
    print(f"            256 site and a 47-row site can exist; pass 21's ranked hypothesis (1),")
    print(f"            'a different site', is the one this arithmetic supports.")


if __name__ == "__main__":
    main()
