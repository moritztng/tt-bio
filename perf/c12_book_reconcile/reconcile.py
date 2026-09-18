#!/usr/bin/env python3
"""The C12 book after reblock-delete's sign flip, recomputed from each row's own final verdict.

`c12-orchestrator` closed at pass 46 with the named route at 2.1975 s of the 2.3810 s that 12.5 s
needs (92.3 %), the last 0.1835 s contingent on `c12-reblock-delete` landing near the top of a band
it had never measured. Three rows concluded after that book was printed. This script re-derives the
route from their verdicts instead of copying the 2.1975 s forward -- which is the exact failure mode
`book.py`'s own docstring was written to prevent, twice.

WHAT CHANGED SINCE PASS 46, three facts, each from the concluding row's own DONE line:

  1. reblock-delete is REFUTED, not short. Measured +2.7578 s on the fold against a predicted
     -1.0062 s: the fused gate costs 7.4369 ms/call against 1.2226 ms/call for the same matmul
     ungated at the production 512 aa shape, 6.083x, against a pre-registered 1.1206 ms/call kill
     bar (6.6x miss) and ~1300x the session A/A floor. Its 1.0062 s credit comes out of the book on
     that row's own instruction. The flag ships default-off, so the fold today is unaffected -- this
     is the removal of a credit, not a debit against the fold.

  2. fused-eltwise cannot be added to the only fold-measured stack. `c12-compose-fold` s6 gave it
     its first fold number, +0.1016 s, CI [+0.0255, +0.1777] -- and found that
     TT_BIO_DIT_COND_HOIST + TT_BIO_FUSE_COND_MULADD together silently drop the AdaLN sigmoid and
     collapse 512 aa plDDT from 0.8641 to 0.3679. eltwise stacks with silu, NOT with cond-hoist.
     Since cond-hoist is half of the 0.5756 s stack and the silu+eltwise alternative is smaller
     (singles sum 0.4308 s before any sub-additivity), the route keeps silu+hoist and eltwise's
     booked 0.1321 s becomes 0.0000 s ADDABLE. Nobody had booked this; pass 46 still carried
     eltwise as additive OP-GO.

  3. T1 (diffusion head-major) repriced itself from a pure delete into a near-cancelling pair.
     `c12-diffusion-head-major` pass 2: the arm deletes 0.34131 s / 460.8 Mc but MOVES 0.31172 s of
     projection off ttnn.linear onto minimal_matmul, delete/move ratio 1.095. Its own words: "a
     minimal_matmul 1.10x slower than the ttnn.linear it replaces takes the entire prize". Two
     passes, zero hardware numbers -- the row's dispatch line is card=cpu and it holds no lease.

    python3 reconcile.py
"""

FOLD = 14.8810          # c10-bare-baseline quiet-box median, pinned during-sampled 1350 MHz
NEED125 = FOLD - 12.5   # 2.3810 s
NEED100 = FOLD - 10.0   # 4.8810 s

# Every entry is a row's own final number. `s` is what it adds to the route TODAY.
ROUTE = [
    dict(name="compose stack (silu+cond-hoist)", s=0.5756, status="FOLD",
         src="c12-compose-fold, s3 (31 reps) + s5 (10 reps) pooled by inverse variance, se 0.0621, "
             "CI [+0.4539,+0.6974], 9.3 sigma; replicated a third time in s6 (silu +0.3292, hoist "
             "+0.2544 against a +0.0242 +/- 0.0616 A/A floor). Accuracy CLEAR: worst 0.38302 A = "
             "0.477x its own 0.80218 A seed floor. Both flags on origin/main, both default-off "
             "(tenstorrent.py:305, :1415)."),
    dict(name="fused-eltwise (at-pin)", s=0.0000, status="BLOCKED-BY-CONFLICT", was=0.1321,
         src="MEASURED at the fold at +0.1016 s (c12-compose-fold s6, CI [+0.0255,+0.1777]) but "
             "NOT ADDABLE: hoist+eltwise collapses 512 aa plDDT to 0.367685, reproducible. Its "
             "flag TT_BIO_FUSE_COND_MULADD is not on main at all -- only on "
             "wk/c12-fused-eltwise-at-pin. Recoverable only by making the two levers compose, "
             "which is unbuilt work nobody has scoped."),
    dict(name="matmul class", s=0.1352, status="CAP",
         src="c12-matmul-key-attribution: class measured in situ at 0.6808 s against 1.4794 s "
             "booked; <= 0.1352 s / 182.5 Mc caps ALL matmul levers, entered once. No row, no "
             "lever built, no mechanism named. The 64-of-110 core pin (0.0584 s) and cross-family "
             "(0.081 s) are inside this cap, not on top."),
    dict(name="Axis A host", s=0.0960, status="PRED",
         src="c12-host-decomp: 0.0000 s reducible at the latency cell, 0.0960 s generous, "
             "0.1896 s absurd, against the 0.8490 s it was asked for. Carried at generous."),
    dict(name="genop recoverable", s=0.0184, status="STOP", 
         src="c12-genop-triatt-slack closed VERDICT STOP at 0.0184 s (was 0.4544 s, 95.9 % of "
             "which was two instrument defects). A STOP'd row's residual, carried for generosity."),
    dict(name="reblock-delete", s=0.0000, status="REFUTED", was=1.0062,
         src="c12-reblock-delete: +2.7578 s measured against -1.0062 s predicted, 6.6x its own "
             "kill bar, ~1300x the session A/A floor. Removed on that row's own instruction."),
]

# T1 is the one row still in flight. Three readings of it, none measured on hardware.
T1 = [
    ("0.0000 s  minimal_matmul 1.10x slower", 0.0000,
     "the row's own refutation condition: delete 0.34131 s, move 0.31172 s, ratio 1.095"),
    ("0.2340 s  parent's PREDICTED", 0.2340, "0.745 discount from tri-attention's measured recovery"),
    ("0.3413 s  full-delete bound", 0.34131, "every converted op deleted AND the move free"),
]

print("=" * 98)
print(f"THE C12 BOOK, RECONCILED     fold of record {FOLD:.4f} s at a pinned during-sampled 1350 MHz")
print("=" * 98)
route = 0.0
for e in ROUTE:
    route += e["s"]
    was = f"   (was {e['was']:.4f} s at pass 46)" if e.get("was") else ""
    print(f"\n  {e['name']:34s} {e['s']:7.4f} s   {e['status']}{was}")
    print(f"      {e['src']}")
print(f"\n  {'NAMED ROUTE, T1 excluded':34s} {route:7.4f} s")
print(f"  {'pass 46 printed':34s} {2.1975:7.4f} s   removed: reblock 1.0062, eltwise 0.1321, T1 0.2340")

fold_measured = sum(e["s"] for e in ROUTE if e["status"] == "FOLD")
print(f"\n  {'measured AT THE FOLD and GO':34s} {fold_measured:7.4f} s -> fold {FOLD-fold_measured:.3f} s")
print(f"  {'SHIPPED (defaults flipped on main)':34s} {0.0:7.4f} s -> fold {FOLD:.3f} s")

print()
print("=" * 98)
print("AGAINST 12.5 s, SWINGING T1 ACROSS EVERY READING IT HAS")
print("=" * 98)
print(f"  12.5 s needs {NEED125:.4f} s.  10.0 s needs {NEED100:.4f} s.\n")
print(f"  {'T1 reading':40s} {'route':>8s} {'% of 2.3810':>12s} {'short':>9s} {'fold':>8s}")
for lbl, v, _ in T1:
    t = route + v
    print(f"  {lbl:40s} {t:8.4f} {100*t/NEED125:11.1f} % {NEED125-t:9.4f} {FOLD-t:8.3f} s")
print()
best = route + max(v for _, v, _ in T1)
print(f"  Best case over all T1 readings: {best:.4f} s = {100*best/NEED125:.1f} % of 12.5 s, "
      f"{100*best/NEED100:.1f} % of 10.0 s.")
print(f"  The decision is INVARIANT to T1: the route is short of 12.5 s by "
      f"{NEED125-best:.4f} s even if T1 lands at its full bound with the move free,")
print(f"  and T1's whole range spans {max(v for _,v,_ in T1)-min(v for _,v,_ in T1):.4f} s against "
      f"a {NEED125-best:.4f} s residual gap. No T1 outcome changes the verdict.")

print()
print("=" * 98)
print("WHAT THE CEILING STILL SAYS, AND WHY IT IS NOT A ROUTE")
print("=" * 98)
# ceiling.py's class bounds, unchanged -- a bound is a class property, not a lever's size.
CEIL = [("Matmul", 3.9705, 0.1352), ("GenericOp", 3.3830, 3.3830-2.1693), ("BinaryNg", 2.3899, 0.1321),
        ("LayerNorm", 1.3808, 0.2415), ("Transpose", 0.5174, 0.2850), ("SDPA", 0.4335, 0.0770),
        ("NlpCreateHeads", 0.3007, 0.30067), ("twelve smaller", 0.8330, 0.01309+0.11179+0.13298)]
dev_bound = sum(b for _, _, b in CEIL)
ceiling = dev_bound + 0.0960
print(f"  device, every class at its measured bound   {dev_bound:.4f} s")
print(f"  host, generous                              {0.0960:.4f} s")
print(f"  CEILING on the enumerated op set            {ceiling:.4f} s -> fold {FOLD-ceiling:.3f} s")
print(f"    unchanged by the sign flip: a class BOUND is a property of the class, not of a lever.")
print()
genop_bound = 3.3830 - 2.1693
print(f"  But {genop_bound:.4f} s of that ceiling ({100*genop_bound/ceiling:.1f} %) is GenericOp, and reblock-delete was the")
print(f"  ONLY mechanism ever named for it. That mechanism measured 6.083x the cost of the op it")
print(f"  fused into. So the largest single block of the ceiling now has zero named mechanism and")
print(f"  one refuted attempt, which is weaker evidence than 'unscreened'.")
print()
print(f"  C12's measured decay against a roof-ratio ceiling, every instance:")
for n, c, m in (("kblock", 0.3423, 0.035), ("cross-family", 0.9454, 0.081),
                ("genop slack", 0.4544, 0.0184), ("64/110 pin (instr. err)", 0.2150, 0.0584)):
    print(f"    {n:20s} {c:.4f} -> {m:.4f} s   decayed {100*(1-m/c):.1f} %")
print(f"  Mean decay 85.6 %. Applying the LEAST of those decays (72.8 %) to the "
      f"{ceiling-route:.4f} s of")
print(f"  unclaimed ceiling gives {0.272*(ceiling-route):.4f} s, which still leaves 12.5 s short by "
      f"{NEED125-route-0.272*(ceiling-route):.4f} s.")

print()
print("=" * 98)
print("VERDICT")
print("=" * 98)
print(f"""
  12.5 s IS NOT REACHABLE on this op set. The named route is {route:.4f} s of the {NEED125:.4f} s needed
  ({100*route/NEED125:.1f} %), {best:.4f} s ({100*best/NEED125:.1f} %) if T1 lands at its full bound with its
  0.31172 s move free. Of pass 46's 92.3 %, {100*1.0062/NEED125:.1f} points were reblock-delete alone -- a band
  that had never been measured, and which then measured the OPPOSITE SIGN at 2.7x the magnitude it
  was booked at. A further {100*0.1321/NEED125:.1f} points were eltwise, additive on paper and not addable in fact.

  10.0 s IS EXCLUDED STRUCTURALLY, unchanged and now over-determined: generic_op's six sites are
  all traffic-bound against a measured 260.9 FLOP/byte machine balance, so its 3.3830 s cannot go
  under a 2.1693 s traffic floor at any arithmetic rate. It needs {NEED100:.4f} s; the whole
  enumerated ceiling with every class at its optimistic bound is {ceiling:.4f} s.

  WHAT IS REAL AND UNSHIPPED: {fold_measured:.4f} s, the silu + cond-hoist stack, measured at the fold
  across three sessions, accuracy clear against its own seed floor, both flags already on
  origin/main default-off. Flipping them is a one-line change and 14.881 -> 14.305 s, 1.0402x.
  That is the campaign's entire deliverable and it is worth more than anything left in the book.
  It is blocked on ask 8879 (Moritz's default-flip OK) plus cond-hoist's 28-line eager-build merge.

  RECOMMENDATION: close C12. Land the {fold_measured:.4f} s. Do not open another lever row -- no named,
  already-measured candidate worth >= 1 s exists, and the one block big enough to matter
  (GenericOp, {genop_bound:.4f} s of bound) just refuted its only proposed mechanism by 6.1x.
""")
