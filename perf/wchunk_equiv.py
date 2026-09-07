#!/usr/bin/env python3
"""Old vs new W-chunk gate, every shipped shape, no device.

Replicates both versions of the Transition row-height arithmetic with the constants MEASURED on
GWH02 by perf/wchunk_probe.py, and reports every (model, W) where the two disagree. The device
A/B can only afford a handful of rungs; this covers the whole space, which is what makes "no
small-size cost" a claim about the code rather than about two folds.
"""
# MEASURED on GWH02 after a real device open (perf/wchunk_probe.py).
GX, GY = 8, 9
THR = 608                 # TRANSITION_W_CHUNKING_THRESHOLD, snapped
CHUNK = 480               # TRANSITION_W_CHUNK_SIZE, snapped
BYTES_PER_CORE = 393216   # TRANSITION_L1_CHUNK_BYTES_PER_CORE
SEQ_MORE = 1088           # SEQ_LEN_MORE_CHUNKING
MAX_C = 384               # SMALL_GRID_TRANSITION_MAX_C
H_BIG, H_STD = 32, 16     # TRANSITION_H_CHUNK_SIZE_BIG / _SIZE
BIG_MAX_W = 384          # TRANSITION_H_CHUNK_BIG_MAX_W

import sys

ELEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 1179648  # SMALL_GRID_TRANSITION_ELEMS

tile = lambda v: -(-int(v) // 32) * 32


def base_h(W, c):
    return H_BIG if (W <= BIG_MAX_W and c <= 256) else H_STD


def ref_of(c):
    return (1024 * 128) * 128 // max(128, c)


def l1_rows_at(w, c, hid):
    return BYTES_PER_CORE * GX * GY / (2 * tile(w) * (tile(c) + 2 * tile(hid)))


def old(W, H, c, hid):
    bh, ref = base_h(W, c), ref_of(c)
    w_eff = min(W, CHUNK) if W > THR else W
    h = max(1, int(bh * min(1.0, ref / (w_eff * c))))
    if ELEMS and 256 < c <= MAX_C and W <= THR and H <= SEQ_MORE:
        h = max(h, min(bh, ELEMS // (w_eff * c)))
    cap = max(1, int(BYTES_PER_CORE * GX * GY // (2 * tile(w_eff) * (tile(c) + 2 * tile(hid)))))
    return (W > THR), w_eff, min(h, cap)


def new(W, H, c, hid):
    bh, ref = base_h(W, c), ref_of(c)
    rows_at_W = min(bh * min(1.0, ref / (W * c)), l1_rows_at(W, c, hid))
    w_chunked = rows_at_W < 1.0
    w_eff = min(W, CHUNK) if w_chunked else W
    h = max(1, int(bh * min(1.0, ref / (w_eff * c))))
    if ELEMS and 256 < c <= MAX_C and W <= THR and H <= SEQ_MORE:
        h = max(h, min(bh, ELEMS // (w_eff * c)))
    return w_chunked, w_eff, min(h, max(1, int(l1_rows_at(w_eff, c, hid))))


# (label, pair channel, swiglu hidden) -- the shipped Transition shapes on this engine.
MODELS = [("boltz2/esmfold2 c=128", 128, 512), ("protenix-v2 c=256", 256, 1024),
          ("rf3 c=192", 192, 768), ("opendde c=384", 384, 1536)]
LADDER = [128, 256, 298, 320, 384, 448, 512, 576, 608, 640, 672, 768, 896, 960, 1024, 1088]

hdr = "%-24s %5s | %9s %6s %3s | %9s %6s %3s | same?" % (
    "model", "W", "old chunk", "w_eff", "h", "new chunk", "w_eff", "h")
print("SMALL_GRID_TRANSITION_ELEMS = %d" % ELEMS)
print(hdr)
ndiff_below = 0
diffs = []
for label, c, hid in MODELS:
    for W in LADDER:
        o, n = old(W, W, c, hid), new(W, W, c, hid)
        same = "yes" if o == n else "NO"
        if o != n:
            diffs.append((label, W, o, n))
            if W <= THR:
                ndiff_below += 1
        print("%-24s %5d | %9s %6d %3d | %9s %6d %3d | %s"
              % (label, W, o[0], o[1], o[2], n[0], n[1], n[2], same))
    w = 32
    while w <= 262144 and not new(w, w, c, hid)[0]:
        w += 32
    print("%-24s -> new gate first fires at W=%s\n" % (label, w if w <= 262144 else None))

print("rungs that differ at all: %d" % len(diffs))
print("rungs that differ AT OR BELOW the old %d-token gate: %d" % (THR, ndiff_below))
for label, W, o, n in diffs:
    print("  differs: %-24s W=%-5d old=%s new=%s" % (label, W, o, n))
