#!/usr/bin/env python3
"""What the ENUMERATED op set can still give, class by class, against the two targets.

Every number here is measured in situ or is a class cap another C12 row measured. Nothing is a new
claim. The point is to answer the question the campaign has not answered in one place: not "did a
lever land" but "is there enough left in the fold's own op set to reach the targets at all".

Sources, all committed on wk/c12-orchestrator:
  device split, 13.2090 s in situ    perf/c12_orchestrator/insitu_budget/insitu_budget.py
                                     (c12-profiled-fold, six units, integer call weights)
  generic_op per-site traffic floors perf/c12_genop_rate/headroom.json via the same script
  the book                           perf/c12_orchestrator/book/book.py

    python3 ceiling.py
"""
FOLD = 14.8810
DEVICE = 13.2090          # measured in situ, 88.76 % of the fold
NONDEV_LO, NONDEV_HI = 1.6489, 1.6720
AXIS_A_REDUCIBLE = 0.0960  # c12-host-decomp "generous"; 0.0000 at the latency cell, 0.1896 absurd

# Per class: measured in-situ seconds, then what is CLAIMED by a named lever and what BOUNDS the
# rest. "bound" is the most that class could ever give on evidence this campaign measured.
CLASSES = [
    dict(name="Matmul", s=3.9705, claimed=0.1352, bound=0.1352,
         why="c12-matmul-key-attribution measured the class in situ at 0.6808 s against 1.4794 s "
             "booked and put whole-class headroom at <= 0.1352 s / 182.5 Mc. Entered ONCE: the "
             "64-of-110 core pin and the cross-family arm are levers inside that cap, not on top."),
    dict(name="GenericOp", s=3.3830, claimed=1.0062 + 0.0184, bound=3.3830 - 2.1693,
         why="ALL SIX SITES ARE TRAFFIC-BOUND -- zero of six are arithmetic-bound against a measured "
             "260.9 FLOP/byte machine balance. Its traffic floor is 2.1693 s, so the most this class "
             "can ever give is 1.2137 s, and reblock-delete (1.0062 s) plus genop-triatt (0.0184 s) "
             "already claim 84.6 % of that."),
    dict(name="BinaryNg", s=2.3899, claimed=0.1321, bound=0.1321,
         why="in-place eltwise measured AT the DRAM roof (415.9 and 423.6 GB/s against 442.9). Not a "
             "rate target at any value; the only lever is deleting the traffic, and "
             "c12-fused-eltwise-at-pin enumerated that exhaustively at 0.1321 s DRAM-real after "
             "correctly zeroing 0.1649 s that L1 residency had already deleted."),
    dict(name="LayerNorm", s=1.3808, claimed=0.2415, bound=0.2415,
         why="cond-hoist takes 0.2415 s of it (inside the 0.6118 s compose stack, not additional). "
             "The remainder has no measured mechanism; residual_input_tensor fusion is the one "
             "candidate and c12-fused-eltwise-at-pin already priced its DRAM-real part."),
    dict(name="Transpose", s=0.5174, claimed=0.0, bound=0.2850,
         why="triaged from source: eligibility-widening is 0.000 s at 512 aa, transpose deferral is "
             "already default-on and reaching every call, the L1 route is already taken for free. "
             "What survives is a hand-written dim0/dim1 mover, bound by the 0.2850 s shape and "
             "well under it in truth -- its closest analogue measured a 22.5-30.9 % "
             "width-independent transaction-issue floor. LEAD, no row."),
    dict(name="SDPA", s=0.4335, claimed=0.0, bound=0.0,
         why="no row, no measured mechanism. Counted at zero rather than guessed."),
    dict(name="NlpCreateHeads", s=0.3007, claimed=0.0, bound=0.0,
         why="the attention head split; counted in the layout screen but no mechanism measured."),
    dict(name="twelve smaller", s=0.8330, claimed=0.0, bound=0.0,
         why="unscreened by shape. Counted at zero rather than guessed."),
]

print("=" * 100)
print(f"WHAT THE ENUMERATED OP SET CAN GIVE     fold {FOLD:.4f} s, device {DEVICE:.4f} s in situ "
      f"({100*DEVICE/FOLD:.2f} %)")
print("=" * 100)
print(f"\nStep 1 -- host cannot reach either target, so both are DEVICE problems.")
print(f"  non-device measured        {NONDEV_LO:.4f} - {NONDEV_HI:.4f} s")
print(f"  delete 100 % of it -> fold {DEVICE:.4f} s: misses 12.5 by {DEVICE-12.5:.4f} s, "
      f"10.0 by {DEVICE-10.0:.4f} s")
print(f"  realistically reducible    {AXIS_A_REDUCIBLE:.4f} s (c12-host-decomp, generous)")
for tgt in (12.5, 10.0):
    need = FOLD - tgt
    dev_need = need - AXIS_A_REDUCIBLE
    print(f"  -> {tgt:.1f} s needs {need:.4f} s total, so DEVICE must give {dev_need:.4f} s "
          f"= {100*dev_need/DEVICE:.1f} % of measured device time")

print(f"\nStep 2 -- class by class, what is claimed and what bounds the rest.\n")
print(f"  {'class':16s} {'in situ':>9s} {'claimed':>9s} {'max ever':>9s}  {'% of class':>10s}")
tot_s = tot_claimed = tot_bound = 0.0
for c in CLASSES:
    tot_s += c["s"]; tot_claimed += c["claimed"]; tot_bound += c["bound"]
    print(f"  {c['name']:16s} {c['s']:9.4f} {c['claimed']:9.4f} {c['bound']:9.4f}  "
          f"{100*c['bound']/c['s']:9.1f} %")
print(f"  {'-'*58}")
print(f"  {'TOTAL':16s} {tot_s:9.4f} {tot_claimed:9.4f} {tot_bound:9.4f}  "
      f"{100*tot_bound/tot_s:9.1f} %")

print(f"\nStep 3 -- the ceiling. Device bound + host reducible, every lever at its optimistic end.\n")
ceil_total = tot_bound + AXIS_A_REDUCIBLE
print(f"  device, every class at its measured bound   {tot_bound:.4f} s")
print(f"  host, generous                              {AXIS_A_REDUCIBLE:.4f} s")
print(f"  CEILING on the enumerated op set            {ceil_total:.4f} s -> fold {FOLD-ceil_total:.3f} s")
for tgt in (12.5, 10.0):
    need = FOLD - tgt
    d = need - ceil_total
    print(f"    vs {tgt:.1f} s (needs {need:.4f} s): "
          f"{'SHORT by %.4f s' % d if d > 0 else 'reachable, %.4f s of slack' % -d}")

print(f"""
Step 4 -- reading it honestly.

  12.5 s is AT THE EDGE of the enumerated set, not comfortably inside or outside it. The ceiling
  above assumes every remaining lever lands at its optimistic end SIMULTANEOUSLY, including a
  0.2850 s transpose mover that has no row and whose own analogue says it should come in well under
  that. Against a campaign record where every lever sized off a roof ratio collapsed on contact
  (kblock 0.3423 -> 0.035, cross-family 0.9454 -> 0.081, genop slack 0.4544 -> 0.0184, a 40-90 %
  decay each time), the expected value is clearly below 12.5 s even though the ceiling brushes it.

  10.0 s is EXCLUDED by the op set, and the reason is structural rather than a shortfall of effort:
  it needs {FOLD-10.0:.4f} s and the whole enumerated set bounds at {ceil_total:.4f} s. The single
  hardest constraint is that generic_op's six sites are ALL traffic-bound, so its 3.3830 s cannot go
  below a 2.1693 s traffic floor however fast the arithmetic gets. That also voids the route passes
  10-11 built the 10.0 s case on -- "raise the arithmetic rate 1.33-2.64x on three ops" -- which
  compared an op moving 402.9 MB per call against a dense cube and was a 4.4x phantom.

  So reaching 10.0 s requires a DIFFERENT OP SET, not a faster one: fewer or cheaper ops for the
  same model arithmetic. That is kernel and algorithm work outside what this campaign scoped, and
  it must not be confused with doing less of the model's own work, which is a cheat and barred.
""")
