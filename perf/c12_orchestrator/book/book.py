#!/usr/bin/env python3
"""The single authoritative C12 book. Every entry carries its provenance and its status.

This file exists because the book has now been wrong twice in one campaign, in both directions:

  * `genop unowned slack` sat at 0.4544 s for a pass after `c12-genop-triatt-slack` closed STOP at
    0.0184 s. A concluded row leaves `queue.tsv`, so reading the queue cannot show what concluded.
  * `fused-eltwise` sat at "accuracy FAILED" for ~17 passes. That was the row's OWN first n=1 read,
    which the row itself retracted mid-pass on a paired n=8 panel (median dplDDT +0.018, 4/8
    negative, arm stdev 0.7452 against a between-seed 0.7472) and whose final verdict reads "built,
    verified correct against float64, and ACCURACY IS CLEARED". The orchestrator never carried it.

Both errors have the same shape: a number was copied forward from prose instead of re-read from the
row's own final verdict. So each entry below names the artifact its status comes from.

STATUS vocabulary, and the distinction that matters:
  FOLD      measured at the fold, paired, resolved against its own session A/A floor
  OP-GO     measured at op or block level, accuracy settled, OWES a benchlocked fold
  PRED      predicted only, never executed
  CAP       a measured in-situ ceiling for a whole class, not a built lever

    python3 book.py
"""

FOLD_OF_RECORD = 14.881   # c10-bare-baseline quiet-box median, pinned during-sampled 1350 MHz

BOOK = [
    dict(name="compose stack (silu+cond-hoist)", s=0.5756, status="FOLD",
         src="c12-compose-fold s3 (31 reps) and s5 (10 reps), TWO INDEPENDENT SESSIONS pooled by "
             "inverse variance: +0.5756 s, se 0.0621, CI [+0.4539,+0.6974], a 9.3-sigma effect. "
             "Sessions consistent (z=+0.48); both A/A controls unresolved and straddling zero "
             "(-0.1114, +0.1236); pooled composed / pooled A/A floor 3.57x, so the pre-registered "
             "3x margin clause is MET on pooled data where s3 alone gave 2.87-2.98x. Both sessions "
             "host-spin wedged mid-run, neither truncation conditioned on a timing. See "
             "../compose_recovered/pool.py",
         acc="CLEAR: cdk2x2_298 worst 0.38302 A = 0.477x its own 0.80218 A seed floor, A/A 0.0000 A, "
             "HOLD band not PASS; 512 aa inside the seed floor on both metrics at all 5 seeds",
         owes="nothing measurable -- margin settled by pooling two sessions (3.57x). What it owes "
              "is a DEFAULT FLIP, which is ask 8879 with Moritz, and cond-hoist's 28-line eager-"
              "build merge first (see ../landing/LANDING.md)"),
    dict(name="reblock-delete", s=1.0062, status="PRED", band=(0.9342, 1.2007),
         src="c12-reblock-delete PREREG.md line 79 corrected central (1.0062, not the 1.0990 "
             "headline); 771 insertions incl. a 479-line mm_split/compute.cpp; NEVER EXECUTED",
         acc="not yet scored", owes="everything -- correctness, op-level, and a fold"),
    dict(name="fused-eltwise (at-pin)", s=0.1321, status="OP-GO",
         src="c12-fused-eltwise-at-pin: 0.1321 s / 178.3 Mc DRAM-real, of which 0.0953 s built and "
             "op-verified; 0.1649 s correctly written to zero because L1 residency already deleted "
             "those bytes",
         acc="CLEARED on a paired n=8 seed panel, median dplDDT +0.018, 4/8 negative, arm stdev "
             "0.7452 vs between-seed 0.7472; correctness verified against float64. Its own first "
             "n=1 read said FAILED and the row RETRACTED it -- basin-flip rate is 2 in 8",
         owes="a benchlocked fold; its own fold session was discarded on its A/A floor"),
    dict(name="matmul class", s=0.1352, status="CAP",
         src="c12-matmul-key-attribution, in situ 0.6808 s against 1.4794 s booked; <= 0.1352 s / "
             "182.5 Mc is the cap for ALL matmul levers, entered once",
         acc="n/a", owes="no row; a cap, not a lever"),
    dict(name="Axis A host", s=0.0960, status="PRED",
         src="c12-host-decomp: 0.0000 s reducible at the latency cell, 0.0960 generous, 0.1896 "
             "absurd, against the 0.8490 s it was asked for",
         acc="n/a, host work", owes="its 298 aa SCALING: field"),
    dict(name="genop recoverable", s=0.0184, status="PRED",
         src="c12-genop-triatt-slack VERDICT STOP; was 0.4544 s, which was 95.9 % two instrument "
             "defects (a wrong-mix roof and a dropped repair_B byte field)",
         acc="n/a", owes="nothing, row closed STOP"),
]

DEAD = [
    ("kblock", 0.035, "concluded BELOW its own 0.10 s kill criterion; 0.3423 -> 0.2991 -> 0.2052 -> 0.035"),
    ("cross-family", 0.081, "CLOSED on opportunity cost; realized recovery 1.80 %, not 10.55 %"),
    ("layout dim0/dim1 mover", 0.2850, "LEAD only, no row; bounded well under this by the 22.5-30.9 % "
                                       "transaction-issue floor its closest analogue measured"),
]


need125, need100 = FOLD_OF_RECORD - 12.5, FOLD_OF_RECORD - 10.0
print("=" * 100)
print(f"THE C12 BOOK   fold of record {FOLD_OF_RECORD:.3f} s at a pinned during-sampled 1350 MHz")
print("=" * 100)
tot = 0.0
for e in BOOK:
    tot += e["s"]
    band = f" band {e['band'][0]:.4f}-{e['band'][1]:.4f}" if e.get("band") else ""
    print(f"\n  {e['name']:34s} {e['s']:7.4f} s   {e['status']}{band}")
    print(f"      src   {e['src']}")
    print(f"      acc   {e['acc']}")
    print(f"      owes  {e['owes']}")
print(f"\n  {'TOTAL named route':34s} {tot:7.4f} s")

banked = sum(e["s"] for e in BOOK if e["status"] == "FOLD")
opgo = sum(e["s"] for e in BOOK if e["status"] in ("FOLD", "OP-GO"))
print(f"  {'of which measured AT THE FOLD':34s} {banked:7.4f} s")
print(f"  {'of which fold-or-op GO (accuracy settled)':34s} {opgo:7.4f} s")
print(f"  {'of which SHIPPED (defaults flipped)':34s} {0.0:7.4f} s   <- ask 8879 open with Moritz")

print()
print("=" * 100)
print("AGAINST THE TARGETS")
print("=" * 100)
for tgt, need in ((12.5, need125), (10.0, need100)):
    short = need - tot
    print(f"  {tgt:5.1f} s needs {need:.4f} s -> route {tot:.4f} s = {100*tot/need:5.1f} %, "
          f"{'short by %.4f s' % short if short > 0 else 'CLEARS by %.4f s' % -short}")
lo, hi = next(e["band"] for e in BOOK if e.get("band"))
c = next(e["s"] for e in BOOK if e.get("band"))
print("\n  swinging reblock-delete across its own band, everything else held:")
for lbl, v in (("band low ", lo), ("central  ", c), ("band high", hi)):
    t = tot - c + v
    s = need125 - t
    print(f"     reblock {lbl} {v:.4f} -> route {t:.4f} s, 12.5 s "
          f"{'short by %.4f s' % s if s > 0 else 'CLEARS by %.4f s' % -s}"
          f"   projected fold {FOLD_OF_RECORD - t:.3f} s")
print(f"\n  projected fold from the FOLD-measured term alone: "
      f"{FOLD_OF_RECORD:.3f} - {banked:.4f} = {FOLD_OF_RECORD - banked:.3f} s")
print(f"  from everything with accuracy settled:             "
      f"{FOLD_OF_RECORD:.3f} - {opgo:.4f} = {FOLD_OF_RECORD - opgo:.3f} s")

print()
print("=" * 100)
print("NOT IN THE ROUTE")
print("=" * 100)
for n, s, why in DEAD:
    print(f"  {n:26s} {s:7.4f} s   {why}")
print()
print("  10.0 s is not reachable on the current op set -- six independent derivations agree, and the")
print("  strongest is that generic_op alone keeps a 2.1693 s traffic floor. 12.5 s does not clear on")
print("  the named set at any point in reblock-delete's band.")
