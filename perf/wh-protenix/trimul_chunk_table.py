#!/usr/bin/env python3
"""Which shapes does the tile-padded trimul chunk move, and by how much.

The companion to transition_cap_table.py, for the other op in the same crash class.
Pure arithmetic, no device: it replays `_trimul_chunk_size` for both the shipped logical
`seq_len` and the tile-padded one, so the blast radius of the fix is a table rather than a
claim. Run it before and after any edit to that function.

The chunk tensors are [batch, chunk, seq, seq] TILE tensors, so both seq dims round up to
32. Pricing them on the logical seq understates the footprint by (tile(seq)/seq)^2, which
at 225 aa is 29% -- exactly one doubling of the chunk. Live evidence on the 8x9 Galaxy
(72 cores), protenix-v2 confidence Pairformer:

    205 aa  padded 224  chunk 64  footprint 3,211,264  <= budget 3,629,908   folds
    225 aa  padded 256  chunk 64  footprint 4,194,304  >  budget             THREW
    245 aa  padded 256  chunk 32                                             folds

    python3 perf/wh-protenix/trimul_chunk_table.py           # rows the fix moves
    python3 perf/wh-protenix/trimul_chunk_table.py --all     # every row
"""
import argparse

BUDGET_BASE = 64 * 320 * 320    # TRIANGLE_MULT_L1_CHUNK_BUDGET
CHUNK0 = 32                     # TRIANGLE_MULT_CHUNK_SIZE
GRID_DIVISOR = 13 * 10          # COMPUTE_GRID_X_13 * 10, the Blackhole calibration grid
# Above _trimul_l1_max_seq the chunks live in DRAM and the function returns CHUNK0 before
# reaching the loop, so the fix cannot touch those. Non-fast (the confidence head) is the
# tighter of the two and is what the live crash ran on.
L1_MAX_SEQ = {"fast": 320, "non-fast": 288}
# self._hidden for the trimuls a served protenix-v2 fold reaches, read off a live 225 aa
# fold on UF-EV-A13-GWH02 ([TRIMUL] probe in whbase/pxtail).
HIDDENS = (64, 128, 256)


def chunk(seq, hidden, batch, cores, pad, max_seq):
    if seq > max_seq:
        return CHUNK0
    budget = BUDGET_BASE * cores / GRID_DIVISOR
    sq = (-(-int(seq) // 32) * 32) if pad else seq
    c = CHUNK0
    while hidden % (c * 2) == 0 and batch * (c * 2) * sq * sq <= budget:
        c *= 2
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cores", type=int, default=72, help="8x9 Galaxy")
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()

    print(f"{'mode':<9} {'hidden':>6} {'N':>5} {'chunk now':>10} {'chunk fixed':>12}  verdict")
    moved = 0
    for mode, max_seq in L1_MAX_SEQ.items():
        for hidden in HIDDENS:
            for n in range(16, 1025):
                old = chunk(n, hidden, a.batch, a.cores, False, max_seq)
                new = chunk(n, hidden, a.batch, a.cores, True, max_seq)
                if old != new:
                    moved += 1
                if a.all or old != new:
                    v = "narrowed (bit-exact)" if old != new else "unchanged"
                    print(f"{mode:<9} {hidden:>6} {n:>5} {old:>10} {new:>12}  {v}")
    print(f"\n{moved} (mode, hidden, N) rows narrow; none widen, and narrowing the trimul "
          f"chunk is bit-exact.")
    print("Blackhole is untouched by construction: the fix is inside _IS_SMALL_GRID, "
          "which is False at 110 and 130 cores.")


if __name__ == "__main__":
    main()
