#!/usr/bin/env python3
"""What generic_op's measured headroom does to C12's 12.5 s and 10.0 s targets.

Nothing here is hand-arithmetic: the generic_op ceiling comes from `headroom.json`, which comes
from in-fold seconds joined to roofs measured on the same part in the same session. The rest of the
book is quoted twice -- once as the campaign states it, once re-based by `c12-profiled-fold`'s
measured replay bias -- so the conclusion does not depend on this row re-pricing anyone else's row.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MHZ = 1350.0
FOLD = 14.881          # the campaign's quiet 512 aa fold median, at 1350 MHz
H = json.load(open(HERE / "headroom.json"))

# every other candidate, as the campaign's own book states it (workstream brief, pass 18)
BOOK_AS_STATED = {
    "cond-hoist": 0.2134,
    "silu": 0.2843,
    "kblock": 0.035,
    "cross-family": 0.081,
}
# the same rows re-based by c12-profiled-fold's measured linear/matmul replay bias, which found
# Axis B sized off linear = 4.6604 s against a real 3.3991 s, i.e. over by 1.371x
BOOK_REBASED = {
    "cond-hoist": 0.105,
    "silu": 0.197,
    "kblock": 0.035 / 1.371,
    "cross-family": 0.081 / 1.371,
}
# fused-eltwise (0.1321 s) is excluded from both: its accuracy FAILED, so it is not bankable.

CEILINGS = {
    "f=1.00 (every site at its own measured roof)": H["total_s"] - H["floor_s"],
    "f=0.90": H["total_s"] - H["floor_s"] / 0.90,
    "f=%.3f (best efficiency any site reaches today)" % H["best_eff"]:
        H["total_s"] - H["floor_s"] / H["best_eff"],
}

for bname, book in (("campaign's own book", BOOK_AS_STATED), ("re-based by c12-profiled-fold",
                                                              BOOK_REBASED)):
    rest = sum(book.values())
    print("== rest of the book, %s: %.4f s (%s)" % (bname, rest, ", ".join(
        "%s %.4f" % (k, v) for k, v in book.items())))
    for cname, c in CEILINGS.items():
        tot = c + rest
        print("   generic_op %.4f s + rest %.4f s = %.4f s / %.1f Mc  ->  fold %.3f s"
              % (c, rest, tot, tot * MHZ, FOLD - tot))
        for target in (12.5, 10.0):
            need = FOLD - target
            print("      %s %.1f s: needs %.4f s, short by %.4f s / %.1f Mc"
                  % ("REACHED" if tot >= need else "NOT reachable", target, need,
                     need - tot, (need - tot) * MHZ))
    print()

# sensitivity: the sites were measured on the b2z tt-metal source build, the roofs on the
# ttnn==0.68.0 wheel. The one arm on both is the 2048^3 cube: wheel 0.1695 ms, b2z 0.1520 ms.
OFF = 0.1695 / 0.1520
print("cross-build offset on the one arm measured on both builds: %.3fx" % OFF)
print("  if the streaming roof is understated by the same factor, floor -> %.4f s and the"
      % (H["floor_s"] / OFF))
print("  f=1.00 ceiling -> %.4f s; 12.5 s still needs %.4f s"
      % (H["total_s"] - H["floor_s"] / OFF, FOLD - 12.5))
