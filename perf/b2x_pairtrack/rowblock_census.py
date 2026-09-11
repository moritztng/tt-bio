#!/usr/bin/env python3
"""Full-grid Transition row block, shipped vs derived-and-snapped, every shipped shape. No device.

The device A/B affords two shapes. This walks the whole (channel x size) space the engine can
reach on a Blackhole full grid and prints every shape whose row block MOVES, plus the per-core
L1 the new block asks for -- which is the number the change is bounded by.

Replicates the arithmetic in Transition.__call__ (tt_bio/tenstorrent.py) for the `not
_IS_SMALL_GRID` branch only; the small-grid path is untouched by the change under gate.
"""
tile = lambda v: -(-int(v) // 32) * 32

BYTES_PER_CORE = 393216   # TRANSITION_L1_CHUNK_BYTES_PER_CORE, full grid (L1 ratio clamps to 1.0)
H_BIG, H_STD = 32, 16     # TRANSITION_H_CHUNK_SIZE_BIG / _SIZE
BIG_MAX_W = 384           # TRANSITION_H_CHUNK_BIG_MAX_W
W_CHUNK = 1024            # TRANSITION_W_CHUNK_SIZE
REF = 1024 * 128


def thr(gx):              # transition_w_chunking_threshold
    return 1536 if gx == 13 else 1024


def base_h(W, c):
    return H_BIG if (W <= BIG_MAX_W and c <= 256) else H_STD


def l1_rows_at(w, c, hid, gx, gy):
    return BYTES_PER_CORE * gx * gy / (2 * tile(w) * (tile(c) + 2 * tile(hid)))


def bytes_per_core(h, w, c, hid, gx, gy):
    return 2 * h * tile(w) * (tile(c) + 2 * tile(hid)) / (gx * gy)


def heights(H, W, c, hid, gx, gy):
    bh = base_h(W, c)
    w_eff = min(W, W_CHUNK) if W > thr(gx) else W
    old = max(1, int(bh * min(1.0, REF / (w_eff * c))))
    l1_h = max(1, int(l1_rows_at(w_eff, c, hid, gx, gy)))
    new = max(old, min(H, 1 << (l1_h.bit_length() - 1)))
    return w_eff, old, new, l1_h


# (label, channel, swiglu hidden, kind) for every 4D Transition the engine ships.
# 3D transitions (single/token track) return before this code.
PARTS = [
    ("boltz2/esmfold2 pair", 128, 512, "pair"),
    ("boltz2 MSA",            64, 256, "msa"),
    ("openfold3 pair",       128, 512, "pair"),
    ("rf3 pair",             192, 768, "pair"),
    ("protenix-v2 pair",     256, 1024, "pair"),
    ("protenix-v2 MSA",       64, 256, "msa"),
    ("opendde pair",         384, 1536, "pair"),
]
# size-ladder rungs + the capacity bar + the sizes the shipped models actually serve
RUNGS = [128, 256, 298, 320, 384, 448, 512, 576, 640, 768, 896, 1024, 1088, 1280, 1536]
MSA_DEPTHS = [1024, 4096, 8192]

for gx, gy in ((11, 10), (13, 10)):
    print("=" * 96)
    print("grid %dx%d = %d cores   (W-chunk threshold %d)" % (gx, gy, gx * gy, thr(gx)))
    print("=" * 96)
    print("%-22s %5s %6s | %6s %5s %5s %5s | %10s %10s" % (
        "part", "W", "H", "w_eff", "raw", "old", "new", "B/core old", "B/core new"))
    moved = 0
    for label, c, hid, kind in PARTS:
        for W in RUNGS:
            Hs = [W] if kind == "pair" else MSA_DEPTHS
            for H in Hs:
                w_eff, old, new, l1_h = heights(H, W, c, hid, gx, gy)
                if new == old:
                    continue
                moved += 1
                print("%-22s %5d %6d | %6d %5d %5d %5d | %10.0f %10.0f" % (
                    label, W, H, w_eff, l1_h, old, new,
                    bytes_per_core(old, w_eff, c, hid, gx, gy),
                    bytes_per_core(new, w_eff, c, hid, gx, gy)))
    print("shapes whose row block moves: %d" % moved)
    over_new, over_old, worst_moved = [], [], 0.0
    for label, c, hid, kind in PARTS:
        for W in RUNGS:
            Hs = [W] if kind == "pair" else MSA_DEPTHS
            for H in Hs:
                w_eff, old, new, l1_h = heights(H, W, c, hid, gx, gy)
                bo = bytes_per_core(old, w_eff, c, hid, gx, gy)
                bn = bytes_per_core(new, w_eff, c, hid, gx, gy)
                if new != old:
                    worst_moved = max(worst_moved, bn)
                    if bn > BYTES_PER_CORE:
                        over_new.append((label, W, H, old, new, bn))
                elif bo > BYTES_PER_CORE:
                    over_old.append((label, W, H, old, bo))
    print("worst per-core L1 among the shapes that MOVED: %.0f B (budget %d) -- %s"
          % (worst_moved, BYTES_PER_CORE,
             "inside" if worst_moved <= BYTES_PER_CORE else "OVER"))
    print("shapes the change pushes OVER the budget: %d %s"
          % (len(over_new), over_new if over_new else "(none -- the snap bounds it by construction)"))
    print("shapes ALREADY over the budget today, left untouched by raise-only: %d" % len(over_old))
    for row in sorted(set(over_old), key=lambda r: -r[4])[:4]:
        print("    %-22s W=%-5d H=%-5d h=%-3d %.0f B/core  (ships today)" % row)
