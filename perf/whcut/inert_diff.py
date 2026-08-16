#!/usr/bin/env python3
"""Diff two inert_check.py dumps and rule on §4.1's ACCEPT condition."""
import json
import sys

a = json.load(open(sys.argv[1]))  # main
b = json.load(open(sys.argv[2]))  # assembled

print("main      %s" % a["head"])
print("assembled %s\n" % b["head"])

fail = []
for grid in [k for k in a if k.startswith("grid_")]:
    ka, kb = a[grid], b[grid]
    moved = sorted(k for k in set(ka) | set(kb) if ka.get(k, "<absent>") != kb.get(k, "<absent>"))
    bh = grid in ("grid_13x10", "grid_11x10")
    print("%s: %d constants, %d differ" % (grid, len(ka), len(moved)))
    for k in moved:
        print("   %-42s %r -> %r" % (k, ka.get(k, "<absent>"), kb.get(k, "<absent>")))
        if bh and not k.startswith("_SDPA"):
            fail.append("%s changed on a Blackhole grid: %s" % (k, grid))

pa, pb = a["sdpa_picks"], b["sdpa_picks"]
moved_padded = sorted({pa[s]["padded"] for s in pa if pa[s]["shipped"] != pb[s]["shipped"]})
print("\nSDPA picks differ at padded sizes: %s" % moved_padded)
for s in sorted(pa, key=int):
    if pa[s]["shipped"] != pb[s]["shipped"]:
        print("   seq %4s padded %4d   %s -> %s" % (s, pa[s]["padded"], pa[s]["shipped"], pb[s]["shipped"]))

# Derive the expectation from K3's own rule rather than quoting a list: it moves the k
# pick exactly where the capped 256 does not divide the padded length AND a 32-aligned
# divisor at or above the cap/2 floor exists. That floor is why 704 and 832 do NOT move
# (largest divisor 64), which the plan's §4.1 list had wrong -- it quoted the sizes where
# the fused kernel silently declines, a superset of the sizes K3 acts on.
CAP, TILE = 256, 32


def expect_move(padded):
    # The 256 < seq <= 384 band returns (64, 64) before the dividing pick is consulted,
    # and K4, which is the lever that would move it, ships off.
    if padded <= CAP or padded % CAP == 0 or padded <= 384:
        return False
    return any(padded % c == 0 for c in range(CAP - TILE, CAP // 2 - 1, -TILE))


derived = sorted({pa[s]["padded"] for s in pa if expect_move(pa[s]["padded"])})
if moved_padded != derived:
    fail.append("SDPA picks moved at %s, K3's rule predicts %s" % (moved_padded, derived))

# Only 64-aligned lengths are reachable from the fold: `_tri_att_sdpa_at` reads q.shape[2],
# which every caller pads to PAIRFORMER_PAD_MULTIPLE = 64. The 32-aligned-only entries are
# dumped for completeness and cannot be hit in production.
reachable = [p for p in moved_padded if p % 64 == 0]
print("reachable from the fold (64-aligned): %s" % reachable)
print("32-aligned only, unreachable:        %s" % [p for p in moved_padded if p % 64])

print("\nswitches main      %s" % a["switches"])
print("switches assembled %s" % b["switches"])

print("\n" + ("REJECT\n  " + "\n  ".join(fail) if fail else "ACCEPT: inert on both Blackhole grids; "
      "SDPA picks move only where K3's rule says, reachable set %s" % reachable))
sys.exit(1 if fail else 0)
