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
    # --- the three classes below were counted at ZERO until pass 45. c12-tail-classes-screen
    # itemised all 1.5672 s of them (perf/c12_tail_screen/leads.json on wk/c12-tail-classes-screen,
    # d94f6dbc5) and its signature sums reconcile to these class totals exactly: SDPA 0.43348 vs
    # 0.4335, NlpCreateHeads 0.30067 vs 0.3007, and the six leads' 1.22312 s + 0.34414 s
    # unaccounted = 1.56726 s. Bounds below are that row's own per-lead prices, NOT new claims.
    dict(name="SDPA", s=0.4335, claimed=0.0, bound=0.0770,
         why="L6, BLOCKED not STOP. Traffic-bound with margin (50.1 FLOP per charged DRAM byte "
             "against the measured 260.9 balance; arithmetic is 11.2 % of it) and the q_chunk is "
             "NOT mis-set -- the executed program_config reads 128/256 at the token site, which is "
             "the best rung of _grid_q_chunk's own sweep. The one mechanism CPU evidence can name "
             "is the [1,16,512,512] bias in bfp8: 44.5 % of the token site's traffic at 0.792x the "
             "bytes = 0.077 s if the rate holds. Needs an Angstrom reading against the seed floor."),
    dict(name="NlpCreateHeads", s=0.3007, claimed=0.0, bound=0.30067,
         why="L1, the screen's one GO and a PURE DELETE. The head-major writer that removes these "
             "ops is ALREADY WRITTEN AND SHIPPED DEFAULT-ON for tri-attention "
             "(tt_bio/triatt_qkv.py:39,131 on origin/main, TRIATT_HEAD_MAJOR_QKV/TAIL = True), "
             "bit-exact by torch.equal at six sizes, measured in-fold at 512 aa TriAtt body "
             "19719.8 -> 16716.5 ms = 1.1797x. The DIFFUSION side never got it. Precondition holds "
             "at all four signatures: head_dim is a whole number of tiles in PADDED form (64 = 2 "
             "tiles token transformer, 32 = 1 tile atom blocks), so output tile (i,n) of the qkv "
             "matmul already IS tile (batch,head,row) of q/k/v -- only the destination address "
             "changes and no element moves inside a tile. Bound is the full class (delete); the "
             "row's own PREDICTED is 0.234 s at a 0.745 discount taken from tri-attention's "
             "measured recovery rather than invented."),
    dict(name="twelve smaller", s=0.8330, claimed=0.0, bound=0.01309 + 0.11179 + 0.13298,
         why="screened into twelve items; largest is Slice at 0.1917 s, eight are under 0.06 s. "
             "Bound is three pieces and nothing else: L1's NLPConcatHeads signature 0.01309 s "
             "(same delete as above); L2 diffusion head-48 merge 0.13539 s MINUS the 0.0236 s the "
             "enabling matmul grows by (head_dim 48 means elements DO move inside tiles, so "
             "nlp_concat_heads cannot be used and L1's tile re-point does not transcribe -- the "
             "out projection would absorb it as a 16-way batched matmul, K 768 -> 1024) = 0.11179 s "
             "net; and L4 OuterProductMean's layout round-trip 0.13298 s, which needs a custom "
             "outer-product kernel writing the consumer's axis order. L3 (pair Transition chunk, "
             "0.15038 s) is STOP: byte-neutral in the chunk count, so chunk-height tuning cannot "
             "move one byte, and both ops already run at 92.8-103.3 % of their mix-matched roof. "
             "L5 Embeddings (0.05713 s) is STOP: 5.5 % of roof is the known per-element "
             "gather/scatter floor on Blackhole, proven not-bandwidth by a bfp8 arm at half the "
             "bytes landing 0.1 % apart. 0.34414 s of the tail stays unaccounted and at zero."),
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

# ---------------------------------------------------------------------------------------------
# Step 4 -- the ceiling by MATURITY, because "the most a class could give" mixes a lever whose
# kernel is already shipped with one nobody has written. Pass 45 added 0.6355 s of screened bound
# and the whole question of whether 12.5 s is reachable now turns on which tier you credit.
# ---------------------------------------------------------------------------------------------
TIERS = [
    ("T1 built + measured elsewhere", 0.30067 + 0.01309,
     "L1 head-major qkv/out on the diffusion side. Kernel written, default-on for tri-attention, "
     "bit-exact at six sizes, 1.1797x measured in-fold on the TriAtt body. Transcription, not "
     "new kernel work. At the row's own PREDICTED discount this is 0.234 s, not 0.3138 s."),
    ("T2 accuracy reading needed", 0.0770,
     "L6 SDPA bias in bfp8. One dtype change, prize computed from audited traffic, but it is a "
     "precision change on an attention bias and owes an Angstrom reading against its own "
     "own-fixture seed floor."),
    ("T3 unbuilt design work", 0.11179 + 0.13298,
     "L2's 16-way batched-matmul absorb and L4's custom outer-product kernel. Both have a named "
     "mechanism and a priced prize; neither has a line of code."),
]
ENUMERATED_BEFORE = 2.1035   # what this script printed before the tail was screened (passes 40-44)

print("=" * 100)
print("Step 4 -- BY MATURITY, and what each tier does to the 12.5 s question")
print("=" * 100)
print(f"\n  superseded ceiling, tail counted at zero   {ENUMERATED_BEFORE:.4f} s -> fold "
      f"{FOLD-ENUMERATED_BEFORE:.3f} s  (SHORT of 12.5 by {(FOLD-ENUMERATED_BEFORE)-12.5:.4f} s)")
print(f"  that gap was the campaign's headline answer for five passes, with 1.5672 s unscreened.\n")

run = AXIS_A_REDUCIBLE + sum(c["bound"] for c in CLASSES if c["name"] not in
                             ("SDPA", "NlpCreateHeads", "twelve smaller"))
print(f"  {'tier':32s} {'gives':>8s} {'cumulative':>11s} {'fold':>8s}   vs 12.5 s")
print(f"  {'(device enumerated + host)':32s} {'':>8s} {run:11.4f} {FOLD-run:8.3f}   "
      f"short {(FOLD-run)-12.5:.4f} s")
for name, give, _ in TIERS:
    run += give
    d = (FOLD - run) - 12.5
    print(f"  {name:32s} {give:8.4f} {run:11.4f} {FOLD-run:8.3f}   "
          f"{'short %.4f s' % d if d > 0 else 'CLEARS by %.4f s' % -d}")

print(f"\n  T1 at its own PREDICTED 0.234 s instead of its 0.3138 s bound:")
run_p = AXIS_A_REDUCIBLE + sum(c["bound"] for c in CLASSES if c["name"] not in
                               ("SDPA", "NlpCreateHeads", "twelve smaller")) + 0.234
print(f"    cumulative {run_p:.4f} s -> fold {FOLD-run_p:.3f} s, "
      f"{'short %.4f s' % ((FOLD-run_p)-12.5) if (FOLD-run_p)>12.5 else 'CLEARS by %.4f s' % (12.5-(FOLD-run_p))}")

print(f"""
Step 5 -- reading it honestly.

  WHAT CHANGED. For five passes this campaign's answer was "12.5 s is short {(FOLD-ENUMERATED_BEFORE)-12.5:.4f} s at a
  ceiling that bounds every enumerated device class", carried with the explicit caveat that
  1.5672 s of measured device time had never been screened and was counted at zero. That screen
  landed. It found {0.30067+0.01309+0.0770+0.11179+0.13298:.4f} s of bound in the block, which is {(0.30067+0.01309+0.0770+0.11179+0.13298)/((FOLD-ENUMERATED_BEFORE)-12.5):.1f}x the gap it had to close, so
  the caveat was load-bearing and the old headline does not survive it.

  12.5 s IS NOW REACHABLE, AND T1 ALONE STRADDLES IT. The single lever whose kernel already exists
  takes the ceiling to a {FOLD-(AXIS_A_REDUCIBLE+sum(c['bound'] for c in CLASSES if c['name'] not in ('SDPA','NlpCreateHeads','twelve smaller'))+0.30067+0.01309):.3f} s fold at its full-delete bound and a {FOLD-run_p:.3f} s fold at its own
  predicted discount -- i.e. it lands either side of the target by under 0.05 s in both directions.
  That is not a claim that 12.5 s will happen; it is the first time in the campaign that the
  arithmetic does not forbid it. Crediting T2 or either half of T3 puts it clear with real slack.

  WHY T1 DESERVES MORE CONFIDENCE THAN THE CAMPAIGN'S DECAYED LEVERS. C12's record is that every
  lever sized off a ROOF RATIO collapsed on contact (kblock 0.3423 -> 0.035, cross-family
  0.9454 -> 0.081, genop slack 0.4544 -> 0.0184) while both levers sized off an EXECUTED GRAPH met
  or beat their predictions (silu by 5.3 %, cond-hoist against 0.1445 s). T1 is neither an estimate
  nor a rate bet: it is a DELETION of ops priced at their own measured in-situ time, its kernel is
  already running default-on elsewhere in the same tree at a measured 1.1797x, and its precondition
  was checked at all four executed signatures. Its risk is transcription risk, not prize risk.

  10.0 s IS STILL EXCLUDED and the screen did not move it. It needs {FOLD-10.0:.4f} s; the whole set
  including every tier above bounds at {AXIS_A_REDUCIBLE+sum(c['bound'] for c in CLASSES):.4f} s. The binding constraint is unchanged and
  structural: generic_op's six sites are ALL traffic-bound against a measured 260.9 FLOP/byte
  balance, so its 3.3830 s cannot go below a 2.1693 s traffic floor at any arithmetic rate, and the
  tail's own biggest item (SDPA, 0.4335 s) is traffic-bound with margin too. Reaching 10.0 s needs a
  DIFFERENT op set, not a faster one -- which is kernel and algorithm work outside this campaign's
  scope, and is not the same thing as doing less of the model's own work, which is barred.
""")
