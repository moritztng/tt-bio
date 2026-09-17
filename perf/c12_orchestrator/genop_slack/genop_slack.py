#!/usr/bin/env python3
"""How much of generic_op's 1.2050 s slack is NOT already owned by c12-reblock-delete.

The campaign book carries two entries that touch the same seconds:

  reblock-delete   1.0062 s   priced, unbuilt. It fuses the gate and the reblock into the producing
                              matmul's own writer, and its own brief says it attacks
                              `trimul_in` 0.8392 s + `reblock_gated` 0.7225 s = 1.5617 s in situ.
  generic_op gap   1.2050 s   3.3743 s in situ against a 2.1693 s per-site max(traffic, arithmetic)
                              floor, carried as "a GAP, not a lever".

Both are correct in isolation and adding them double-counts, because 0.4707 s of the gap lives in
exactly the two sites reblock-delete attacks. This splits the gap by site so a row can be opened on
the part that is actually unowned, and prices it at the only efficiency with an existence proof on
this part rather than at the roof.

Inputs, verbatim, no re-derivation: perf/c12_genop_rate/headroom.json at 4ed84475d on
wk/c12-genericop-rate (copied here as headroom_4ed84475d.json), whose per-site in-fold seconds come
from c12-profiled-fold's ops reports and whose floors use the roof matching each site's read:write
mix -- bw_add8192 (2R+1W, 435.73 GB/s) for the four matmul-shaped sites and bw_clone (1R+1W,
393.31 GB/s) for the two reblocks, both measured on qb2 card 3 at a forced and sampled 1350 MHz.

Checked and NOT a defect: pass 22 found `perf/roof_shape/shape_roofs.py:177` mislabels the 1R+1W
roof, and that bug does not reach these numbers -- headroom.py picks the roof per site from an
explicit MIX table. Nothing here needs re-measuring on that account.
"""
import json
from pathlib import Path

MHZ = 1350.0
HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "headroom_4ed84475d.json"))
S = {s["site"]: s for s in D["sites"]}

# The two sites c12-reblock-delete attacks, from its own brief: 11 Z -> 3 Z per call on the
# in-projection's writer, which deletes reblock_gated's pass entirely and most of trimul_in's.
OWNED = ("trimul_in", "reblock_gated")
# reblock_back is NOT in reblock-delete's scope: its producer is ttnn.matmul inside
# MatmulDeviceOperation, so there is no wheel route to its writer. That row excluded it explicitly.

print("generic_op, six sites, in situ against their own measured roofs")
print("site             calls  in-situ s   floor s   slack s   %roof  owner")
for name in sorted(S, key=lambda n: -(S[n]["sec"] - S[n]["floor_sec"])):
    s = S[name]
    slack = s["sec"] - s["floor_sec"]
    print("%-15s %6d %10.4f %9.4f %9.4f %7.2f  %s"
          % (name, s["calls"], s["sec"], s["floor_sec"], slack, s["pct_roof"],
             "c12-reblock-delete" if name in OWNED else "-- none --"))

tot_sec = sum(s["sec"] for s in S.values())
tot_floor = sum(s["floor_sec"] for s in S.values())
owned_slack = sum(S[n]["sec"] - S[n]["floor_sec"] for n in OWNED)
free = [n for n in S if n not in OWNED]
free_sec = sum(S[n]["sec"] for n in free)
free_floor = sum(S[n]["floor_sec"] for n in free)
free_slack = free_sec - free_floor
best_eff = D["best_eff"]

print()
print("total in situ      %.4f s / %.1f Mc" % (tot_sec, tot_sec * MHZ))
print("total floor        %.4f s / %.1f Mc" % (tot_floor, tot_floor * MHZ))
print("total slack        %.4f s   (the book's 1.2050 s)" % (tot_sec - tot_floor))
print("  owned by reblock-delete  %.4f s  %.1f %%   (%s)"
      % (owned_slack, 100 * owned_slack / (tot_sec - tot_floor), ", ".join(OWNED)))
print("  UNOWNED                  %.4f s  %.1f %%   (%s)"
      % (free_slack, 100 * free_slack / (tot_sec - tot_floor), ", ".join(sorted(free))))
print()
print("Pricing the unowned four. %.4f s in situ, %.4f s floor." % (free_sec, free_floor))
for f, why in ((1.00, "every site at its own roof -- nothing on this part reaches it"),
               (0.90, "90 % of roof -- no existence proof on this part"),
               (best_eff, "reblock_gated's measured %.2f %% -- the best any of the six reaches"
                          % (100 * best_eff))):
    tgt = free_floor / f
    print("  at f = %.4f: %.4f s -> %.4f s, prize %.4f s / %.1f Mc   [%s]"
          % (f, free_sec, tgt, free_sec - tgt, (free_sec - tgt) * MHZ, why))

# What that does to the 12.5 s arithmetic. Every term is named with its provenance; nothing is summed
# that is not measured or explicitly flagged as predicted.
NEED_125 = 2.3810
NEED_100 = 4.8810
BOOK = [
    ("silu",            0.2843, "measured op-level, GO, accuracy clean, fold owed"),
    ("cond-hoist",      0.2415, "measured block-level, GO, fold owed"),
    ("reblock-delete",  1.0062, "PREDICTED central, band 0.9342-1.2007, unbuilt"),
    ("matmul class",    0.1352, "measured in-situ cap for ALL matmul levers"),
]
genop_prize = free_sec - free_floor / best_eff
print()
print("Against the targets. 12.5 s needs %.4f s, 10.0 s needs %.4f s." % (NEED_125, NEED_100))
run = 0.0
for n, v, prov in BOOK:
    run += v
    print("  %-16s %+.4f  running %.4f   [%s]" % (n, v, run, prov))
run += genop_prize
print("  %-16s %+.4f  running %.4f   [PREDICTED at the best measured efficiency, no row]"
      % ("genop unowned", genop_prize, run))
print()
print("  12.5 s: %.1f %% covered, %.4f s %s" % (100 * run / NEED_125, abs(run - NEED_125),
                                                "SURPLUS" if run >= NEED_125 else "short"))
print("  10.0 s: %.1f %% covered, %.4f s short" % (100 * run / NEED_100, NEED_100 - run))
print()
print("Read that as a ROUTE, not a result. Two of the five terms are predictions, the stack is")
print("sub-additive by policy and must be approved as a stack, and the fold has banked 0.000 s in")
print("35 passes. What changed is only this: before splitting the gap by site, the book's largest")
print("unowned item was quoted at 1.2050 s and overlapped a queued 1.0062 s lever by 0.4707 s; after,")
print("%.4f s of it is genuinely unowned and %.4f s of that has an existence proof on this part."
      % (free_slack, genop_prize))
