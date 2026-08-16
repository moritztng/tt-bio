#!/usr/bin/env python3
"""Which shapes does the tile-padded Transition cap move, and by how much.

Pure arithmetic, no device: it replays `Transition.__call__`'s small-grid row-chunk sizing
for both caps (logical `w_eff` as shipped, tile-padded `w_eff` as fixed) over every token
count a served target can take, so the blast radius of the fix is a table rather than a
claim. Run it before and after any edit to that block.

The measured law it is checked against (perf/wh-protenix/wh_transition_h.py, 14 points over
6 widths on UF-EV-A13-GWH02, 8x9 = 72 cores): the fc1/fc2 pair fits at <= 393,216 B/core and
throws at >= 409,600. Bytes are counted TILE-PADDED, which is the whole point -- 298 aa's
W=298 pads to 320 and lands on 409,600 exactly.

    python3 perf/wh-protenix/transition_cap_table.py            # rows the fix moves
    python3 perf/wh-protenix/transition_cap_table.py --all      # every row
"""
import argparse

CORES = 72                      # 8x9 Galaxy compute grid
BUDGET = 393_216                # TRANSITION_L1_CHUNK_BYTES_PER_CORE at _l1 >= 1,466,080
THROW = 409_600                 # measured first-throw point, B/core
# The cap counts 2 B/element unconditionally, which is right for bf16 and over-counts
# --fast: a bf8_b tile is 1024 B of mantissa plus a 64 B exponent section, 1.0625 B/element.
# The measured throw edge was taken on bf16, so only a bf16 Transition can be read against
# it -- a --fast row over the edge on the cap's own arithmetic is at ~0.53x that in reality.
BPE = {True: 1.0625, False: 2.0}
W_CHUNKING_THRESHOLD = 640      # TRANSITION_W_CHUNKING_THRESHOLD, small grid at full L1
W_CHUNK_SIZE = 512              # TRANSITION_W_CHUNK_SIZE, same
H_CHUNK_FAST, H_CHUNK, H_CHUNK_BIG, BIG_MAX_W = 32, 16, 32, 384

# (label, pair channel, fc1 hidden, fast) for every Transition a served fold reaches at
# pair-tensor rank 4. Read off a live 298 aa protenix-v2 --fast fold on the Galaxy.
SHAPES = [
    ("protenix trunk pair",      256, 1024, True),
    ("protenix trunk pair",      256, 1024, False),
    ("protenix msa (c=128)",     128,  512, True),
    ("protenix msa (c=64)",       64,  128, True),
    ("protenix diffusion cond",  256,  512, False),
    ("protenix confidence pair", 256, 1024, False),
    ("boltz2 / esmfold2 pair",   128,  512, True),
    ("boltz2 / esmfold2 pair",   128,  512, False),
]


def tile(v):
    return -(-int(v) // 32) * 32


def h_chunk(w, c, hid, fast, pad):
    """The shipped sizing. `pad` selects the fixed (tile-padded) cap."""
    h = H_CHUNK_FAST if fast else H_CHUNK
    if not fast and w <= BIG_MAX_W and c <= 256:
        h = H_CHUNK_BIG
    w_eff = min(w, W_CHUNK_SIZE) if w > W_CHUNKING_THRESHOLD else w
    ref = 1024 * 128 * 128 // max(128, c)
    h = max(1, int(h * min(1.0, ref / (w_eff * c))))
    e = (tile if pad else int)
    cap = max(1, int(BUDGET * CORES // (2 * e(w_eff) * (e(c) + 2 * e(hid)))))
    h = min(h, cap)
    real = BPE[fast] * tile(w_eff) * (tile(c) + 2 * tile(hid)) * h / CORES
    return h, cap, real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-n", type=int, default=1024)
    a = ap.parse_args()
    print(f"{'shape':<26} {'N':>5} {'h_old':>5} {'h_new':>5} {'B/core old':>11} "
          f"{'B/core new':>11}  verdict")
    moved = throws = 0
    for label, c, hid, fast in SHAPES:
        for n in range(16, a.max_n + 1):
            ho, _, bo = h_chunk(n, c, hid, fast, pad=False)
            hn, _, bn = h_chunk(n, c, hid, fast, pad=True)
            if ho != hn:
                moved += 1
            if bo >= THROW:
                throws += 1
            if a.all or ho != hn:
                v = ("was THROWING, now fits" if bo >= THROW and bn < THROW else
                     "unchanged" if ho == hn else "shrunk")
                print(f"{label:<26} {n:>5} {ho:>5} {hn:>5} {bo:>11,.0f} {bn:>11,.0f}  {v}")
    print(f"\n{moved} (shape, N) pairs move; {throws} of them were over the "
          f"{THROW:,} B/core measured throw edge before the fix.")


if __name__ == "__main__":
    main()
