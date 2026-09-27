#!/usr/bin/env python3
"""Which lengths below `_Q_SPLIT_MAX_S` are in 832's position: no fused pair, silently.

832 is not special. `_tri_att_fused_large_s` is the route that pairs a NARROW q with a WIDE k,
it is default ON, and it is gated to `q_len > _Q_SPLIT_MAX_S` (1024). Below the cap the only
route is the ladder, and the ladder can only offer the shipped k unless a lever is on. Where the
shipped k does not divide the padded length, `plan` sets `use_padded_mask`, `fill_preconditions`
declines every rung, and the call falls to the materialised fp32 softmax. Nothing raises and no
counter outside the kernel moves.

The cap's justification is that below it "the ladder already lands on a fused pair", verified at
512 and 1024. This enumerates every tile-aligned length instead of sampling two.

It is a MODEL, so it is validated before it is believed. Six lengths have real device outcomes
from this row's own runs; the model has to reproduce all six or the sweep below it means nothing.
"""
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing")
import ttnn                                                        # noqa: E402
from tt_bio import tenstorrent as T                                # noqa: E402
from tt_bio import sdpa_generic as SG                              # noqa: E402
from tt_bio import triatt_sdpa as TS                               # noqa: E402

HEADS, HEAD_DIM, DTYPE = 4, 32, ttnn.bfloat16      # OpenFold3 trunk triangle attention
CORES = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
TILE = T.SDPA_CHUNK_TILE
CAP = TS._Q_SPLIT_MAX_S


def fits(padded, qc, kc):
    """The engine's own budget model: plan preconditions, then the CB fit."""
    try:
        q_pf = TS.q_parallel_factor(padded, HEADS, qc, CORES, cap=0)
        p = SG.plan_for_shape(padded, HEADS, HEAD_DIM, qc, kc, grid=(CORES, 1),
                              split=(max(CORES // (HEADS * q_pf), 1), HEADS, q_pf), dtype=DTYPE)
    except Exception:                                              # noqa: BLE001
        return False
    if p["q_per_core"] != 1 or p["nh_per_core"] != 1 or p["use_padded_mask"]:
        return False
    pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    return bool(SG.cb_fits_l1(p, mask_cb_tiles=pers,
                              **{"%s_dtype" % o: DTYPE for o in ("q", "k", "v", "mask", "out")}))


def serves(n, dividing_k=False, cap=None, one_k_chunk=True):
    """Replicates `_tri_att_sdpa_hifi_inner`'s route order. Returns (route, q, k) or None.

    `one_k_chunk` defaults True because that is what the only default-ON HiFi site actually gets:
    `tenstorrent.py:10483` passes `tri_att_one_k_chunk=tri_att_sdpa_hifi`, so openfold3.trunk
    takes the route and the k order together. Leaving it out is what made this model miss 704.
    """
    cap = CAP if cap is None else cap
    padded = T._padded_sdpa_len(n)
    if n > cap:
        for qc, kc in TS.fused_pairs(n, HEADS, HEAD_DIM, CORES, DTYPE):
            if fits(padded, qc, kc):
                return ("large_s", qc, kc)
    shipped_k = T._sdpa_chunks_shipped(n, n)[1]
    k_chunks = T._dividing_k_chunks(n, n) if dividing_k else (shipped_k,)
    if one_k_chunk and padded != k_chunks[0]:
        k_chunks = (padded,) + tuple(k_chunks)
    for kc in k_chunks:
        wide = kc != shipped_k
        for qc in T._tri_att_q_chunks(n, n):
            if wide and padded % qc:
                continue
            if fits(padded, qc, kc):
                return ("ladder", qc, kc)
    return None


# ---- validation: six lengths this row has measured on the device -------------------------
# (n, dividing_k, cap, expected served?, what the device recorded)
CHECKS = [
    (256,  False, None, True,  "khole n=256 shipped arm served, q256 k256"),
    (704,  False, None, True,  "neighbour704 fusedA 384 served, pick q352 k704"),
    (832,  False, None, False, "plddt832 off arm 0 served / 384 declined"),
    (832,  True,  None, True,  "plddt832 on arm 384 served, pick q416 k416"),
    (832,  False, 768,  True,  "singlek832 capA 384 served via large_s, pick q416 k416"),
    (1088, False, None, True,  "verdict table: 384 served, pick q64 k1088"),
]
print("VALIDATION -- the model against this row's own device runs")
ok = True
for n, dk, cap, expect, note in CHECKS:
    got = serves(n, dividing_k=dk, cap=cap)
    good = bool(got) == expect
    ok = ok and good
    print("  %-5s n=%-5d dividing_k=%-5s cap=%-5s predicted=%-24s expected_served=%-5s  %s"
          % ("OK" if good else "MISMATCH", n, dk, cap if cap else CAP, got, expect, note))
if not ok:
    print("\nthe model does not reproduce the device; the sweep below is NOT usable")
    sys.exit(1)
print("  all six reproduce -- the sweep below uses the same function\n")

# ---- the sweep ---------------------------------------------------------------------------
print("SWEEP -- every tile-aligned length from 32 to 1536, shipping defaults, cap = %d" % CAP)
holes, served = [], []
for n in range(TILE, 1536 + TILE, TILE):
    r = serves(n)
    (served if r else holes).append((n, r))
print("  serves fused: %d of %d" % (len(served), len(served) + len(holes)))
print("  NO fused pair (falls to the materialised/stock path): %d" % len(holes))
print("  the holes: %s" % [n for n, _ in holes])

print("\n  of those, how many are BELOW the cap (so lifting it could reach them)?")
below = [n for n, _ in holes if n <= CAP]
above = [n for n, _ in holes if n > CAP]
print("    below or at %d: %d -> %s" % (CAP, len(below), below))
print("    above %d:      %d -> %s" % (CAP, len(above), above))

print("\n  would lifting the cap alone close them? (cap lowered to 32, nothing else changed)")
fixed, still = [], []
for n in below:
    (fixed if serves(n, cap=TILE) else still).append(n)
print("    closed by the cap lift: %d -> %s" % (len(fixed), fixed))
print("    still no pair:          %d -> %s" % (len(still), still))

print("\n  and would TT_BIO_TRIATT_DIVIDING_K alone close them?")
dfix, dstill = [], []
for n in below:
    (dfix if serves(n, dividing_k=True) else dstill).append(n)
print("    closed by dividing-k:   %d -> %s" % (len(dfix), dfix))
print("    still no pair:          %d -> %s" % (len(dstill), dstill))
