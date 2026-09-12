#!/usr/bin/env python3
"""Re-derive wave 1's closing ceiling table from primary measurements only.

Every input below is a number somebody measured and wrote down, with its source. Nothing is
taken from wave 1's summary block: the point of this script is to check that block, so reading
it would be circular. Run it and the review's arithmetic reproduces:

    python3 perf/b2z2_redteam/ceiling_recheck.py

Sources
-------
PAGE      site/data/perf-512aa.json, models[0].cells.p150a  (committed, this repo)
INTEG     perf/b2z-integrate/results/integ.json, quoted in state/b2z/FINDINGS.md ~line 2144
STALL     b2z-kernel-cycle-census, qb2 card 0, state/b2z/FINDINGS.md (the block identity)
STEPS50   b2z-sampler-steps, state/b2z/FINDINGS.md ~line 1781
CENSUS    b2z-kernel-cycle-census / b2z-diffusion-utilization per-container costs
HOST      b2z-host-residual-kill pass 3, state/b2z-host-residual-kill.md section 1
MEGA      b2z-pairformer-megakernel pass 2, state/b2z/FINDINGS.md ~line 592
"""
import json

# ---------------------------------------------------------------- primary numbers
PAGE_CELL = 20.079          # PAGE   live published Boltz-2 512 aa cell, s/fold
PAGE_TRUNK = 12.22          # PAGE   "the trunk does not move: 12.20-12.24 s in all four arms"
REF_CELL = 23.841           # the campaign's reference cell (b2x-baseline-attrib, never published)
PAGE_PREV_CELL = 23.504     # PAGE   what the page published before fc7fed56

BASE = 19.6525              # INTEG  shipped incumbent, card 3
ALL = 18.594                # INTEG  both code levers, 200 steps
ALLSTEPS = 14.965           # INTEG  both code levers + 50 steps
TRUNK_ALL = 11.3209         # INTEG  trunk phase in the `all` arm
TRUNK_ALLSTEPS = 11.3009    # INTEG  trunk phase in the `allsteps` arm
SAMP_200 = 5.2547           # INTEG  sampler phase at 200 steps
SAMP_50 = 1.6397            # INTEG  sampler phase at 50 steps
TRUNK_NOLADDER = 12.2173    # INTEG  trunk before the MSA ladder

STEPS50_BASE = 19.693       # STEPS50 its own base, median of 5
STEPS50_ARM = 16.107        # STEPS50 50 steps, NO code levers, median of 5

WAIT_IN = 18.3366           # STALL  ms/block, math thread blocked on input tiles
WAIT_OUT = 3.1066           # STALL  ms/block, blocked on output room
COMPUTE = 10.7010           # STALL  ms/block, actually computing
NONRES = 4.1995             # STALL  ms/block, not resident
SPAN = 36.3438              # STALL  ms, the block's measured span

PAIRFORMER_FOLD = 10.2005   # CENSUS device s/fold on current main
DERIVED_FOLD = 22.3142      # CENSUS the derived fold that share was taken against
DIFF_FOLD = 6.5036          # CENSUS diffusion sampler, device s/fold
DIFF_NOKERNEL = 0.32        # CENSUS <=32 % of the traced step is in no kernel

HOST_ZERODEV = 2.586        # HOST   residual with zero device work under it
HOST_INBRACKET = 1.136      # HOST   dispatch exposed inside the trunk/sampler brackets

TRIMUL_FOLD = 5.2626        # MEGA   trimul s/fold (subunit table)
TRIMUL_CALL_MS = 16.961     # MEGA   one trimul call today, WH eager
TRIMUL_CALL_FUSED = 10.5    # MEGA   "realistic" fused call, 1.62x
OLD_PAIRFORMER = 11.596     # MEGA   the Pairformer s/fold the 1.09x was priced against

out = {}
def show(k, v, note=""):
    out[k] = v
    print(f"{k:<44} {v:>10.4f}   {note}")

print("=" * 100)
print("1. THE BLOCK IDENTITY AND THE THREE REGIME MULTIPLIERS")
print("=" * 100)
ident = WAIT_IN + WAIT_OUT + COMPUTE + NONRES
show("block identity sum (ms)", ident, f"vs measured span {SPAN} -> closes to {abs(ident-SPAN)*1e3:.1f} us")
MOVE = WAIT_IN + WAIT_OUT
show("movement term (ms)", MOVE, f"{100*MOVE/SPAN:.1f} % of the block")
half = MOVE / 2 + COMPUTE + NONRES
free = COMPUTE + NONRES
show("block, movement halved (ms)", half, "wave 1 row 2 = 25.622")
show("block, movement free (ms)", free, "wave 1 row 4 = 14.901")
M_HALF, M_FREE = half / SPAN, free / SPAN
show("multiplier, movement halved", M_HALF)
show("multiplier, movement free", M_FREE)

print()
print("=" * 100)
print("2. THE ARMS, AND WHAT WAVE 1'S 'today' ROW ACTUALLY IS")
print("=" * 100)
show("code levers, paired (BASE/ALL)", BASE / ALL, "wave 1 quotes 1.05738")
show("step cut, measured (s)", SAMP_200 - SAMP_50, "150 extra steps cost this much")
show("allsteps arm (s)", ALLSTEPS, "the arm wave 1's row 1 is LABELLED as")
show("steps50 arm, no levers (s)", STEPS50_ARM, "the number wave 1's row 1 actually USES")
show("gap between them (s)", STEPS50_ARM - ALLSTEPS, "= the code levers, from a different run/card")
show("wave 1 non-trunk constant (s)", STEPS50_ARM - TRUNK_ALLSTEPS, "16.107 - 11.300 = 4.807, CROSS-ARM")
show("self-consistent non-trunk @50 (s)", ALLSTEPS - TRUNK_ALLSTEPS, "the allsteps arm's own")
show("  -> inflation (s)", (STEPS50_ARM - TRUNK_ALLSTEPS) - (ALLSTEPS - TRUNK_ALLSTEPS))

print()
print("the ruling re-scored every row by +2.487 s. check it is consistent, and what it leaves:")
RESCORE = ALL - STEPS50_ARM
show("re-scoring constant (s)", RESCORE, "= 18.594 - 16.107, by construction")
show("non-trunk @200 after re-scoring (s)", STEPS50_ARM + RESCORE - TRUNK_ALLSTEPS)
show("non-trunk @200, self-consistent (s)", ALL - TRUNK_ALL, "the `all` arm's own")
show("  -> residual error (s)", abs((STEPS50_ARM + RESCORE - TRUNK_ALLSTEPS) - (ALL - TRUNK_ALL)))
NONTRUNK200 = ALL - TRUNK_ALL

print()
print("=" * 100)
print("3. THE TABLE, RE-DERIVED FROM THE `all` ARM ONLY (200 steps, both code levers)")
print("=" * 100)
def fold(trunk_s):
    return trunk_s + NONTRUNK200

rows = {}
rows["today (measured)"] = ALL
rows["movement halved"] = fold(TRUNK_ALL * M_HALF)
rows["movement free (trunk only)"] = fold(TRUNK_ALL * M_FREE)
for k, v in rows.items():
    print(f"  {k:<32} {v:>8.3f} s   vs ref {REF_CELL/v:.4f}x   vs live page {PAGE_CELL/v:.4f}x")
print("  wave 1 published, 200-step re-score: 18.594 / 15.260 / 11.927 s = 1.2822 / 1.562 / 1.999x")
show("row 2 reproduction error (s)", abs(rows["movement halved"] - 15.260))
show("row 4 reproduction error (s)", abs(rows["movement free (trunk only)"] - 11.927))

print()
print("=" * 100)
print("4. ROW 3: THE MEGAKERNEL'S 1.09x, IN THE UNITS ITS SOURCE STATES")
print("=" * 100)
call_ratio = TRIMUL_CALL_MS / TRIMUL_CALL_FUSED
show("fused trimul call ratio", call_ratio, "MEGA: 16.961 -> 10.5 ms")
saved = TRIMUL_FOLD * (1 - 1 / call_ratio)
show("trimul s/fold removed", saved)
show("=> FOLD ratio vs ref cell", REF_CELL / (REF_CELL - saved), "MEGA states: '~1.09x on the fold'")
show("=> TRUNK ratio (same saving)", OLD_PAIRFORMER / (OLD_PAIRFORMER - saved), "what a trunk multiplier must be")
print("  wave 1's table applied 1.09x as a TRUNK multiplier (7.966 -> 7.309 s).")
print("  Those are two different quantities; the table used the fold number in the trunk slot.")

blk_frac = saved / OLD_PAIRFORMER
show("share of the block it deletes", blk_frac, "all of it movement (launches are 1.1 % here)")
show("  as a share of the movement term", blk_frac * SPAN / MOVE)
# compose correctly: the megakernel deletes a FRACTION of movement, applied to the already-halved term
mv_frac = blk_frac * SPAN / MOVE
blk_both = (MOVE / 2) * (1 - mv_frac) + COMPUTE + NONRES
show("block, bfp8 THEN megakernel (ms)", blk_both)
r3 = fold(TRUNK_ALL * blk_both / SPAN)
show("row 3 re-derived (s)", r3, f"vs ref {REF_CELL/r3:.4f}x, vs live {PAGE_CELL/r3:.4f}x")
print("  wave 1 published 14.603 s = 1.633x.  error %.3f s (%.1f %%)" % (abs(r3-14.603), 100*abs(r3-14.603)/14.603))
# what multiplicative stacking would have given
blk_mult = half / 1.09
print("  multiplicative stacking (what the table did) implies removing %.1f %% of the halved movement,"
      % (100 * (half - blk_mult) / (MOVE / 2)))
print("  against %.1f %% of the full movement in the measured regime -- the same lever cannot do both."
      % (100 * mv_frac))

print()
print("=" * 100)
print("5. THE HARD BOUND, APPLIED SYMMETRICALLY INSTEAD OF TO ONE BLOCK")
print("=" * 100)
REST = ALL - TRUNK_ALL - SAMP_200
show("rest of the fold (s)", REST, "everything not trunk and not sampler, in the `all` arm")
tf = TRUNK_ALL * M_FREE
show("trunk, movement free (s)", tf)
sf_opt = SAMP_200 * M_FREE
sf_con = SAMP_200 * (DIFF_NOKERNEL + (1 - DIFF_NOKERNEL) * M_FREE)
show("sampler free, optimistic (s)", sf_opt, "trunk's own movement fraction")
show("sampler free, conservative (s)", sf_con, "<=32 % of the step is in no kernel; movement-free cannot touch it")
for name, s in (("optimistic", sf_opt), ("conservative", sf_con)):
    f1 = tf + s + REST
    f2 = tf + s
    print(f"  all three blocks free, {name:<13} {f1:>7.3f} s  vs ref {REF_CELL/f1:.3f}x  vs live {PAGE_CELL/f1:.3f}x")
    print(f"  ...and host residual zero too      {f2:>7.3f} s  vs ref {REF_CELL/f2:.3f}x  vs live {PAGE_CELL/f2:.3f}x")

print()
print("=" * 100)
print("6. WHAT 2x ACTUALLY NEEDS, AGAINST EACH DENOMINATOR")
print("=" * 100)
for name, cell in (("campaign ref 23.841 s", REF_CELL), ("live page cell 20.079 s", PAGE_CELL)):
    tgt = cell / 2
    print(f"  {name:<26} 2x = {tgt:>7.3f} s;  from the 18.594 s arm that is another {ALL/tgt:.4f}x")
    need_trunk = tgt - NONTRUNK200
    print(f"      trunk-only route: trunk must reach {need_trunk:>7.3f} s "
          f"(movement-free floor is {tf:.3f} s) -> {'REACHABLE' if need_trunk >= tf else 'IMPOSSIBLE'}")

print()
print("=" * 100)
print("7. THE PAIRFORMER SHARE, RE-DERIVED")
print("=" * 100)
show("wave 1's share", PAIRFORMER_FOLD / DERIVED_FOLD, "10.2005 / 22.3142, a DERIVED fold")
show("share of the measured base", PAIRFORMER_FOLD / BASE, "same numerator, measured card-3 base")
show("trunk share, page's own control", PAGE_TRUNK / PAGE_CELL, "12.22 / 20.079, independent")
show("trunk share, integ arm", TRUNK_NOLADDER / BASE, "12.2173 / 19.6525, independent")
u = 1.0733   # b2z-bfp8-narrow union, block ratio
for s, lab in ((PAIRFORMER_FOLD/DERIVED_FOLD, "wave 1 share"), (PAIRFORMER_FOLD/BASE, "measured share")):
    print(f"  bfp8 union 1.0733x on the block -> fold {1/(1-s+s/u):.4f}x  ({lab})")

print()
print("=" * 100)
print("8. THE 1.2822x, DECOMPOSED")
print("=" * 100)
show("1.2822x", REF_CELL / ALL)
show("  = pre-campaign, already on the page", REF_CELL / PAGE_CELL, "23.841 -> 20.079, levers merged at fc7fed56")
show("  x this campaign, vs the live cell", PAGE_CELL / ALL)
show("      of which paired A/B in one run", BASE / ALL, "the only part measured against its own incumbent")
show("      residual = card/window drift", (PAGE_CELL / ALL) / (BASE / ALL), "page card 0 vs campaign card 3")
show("product check", (REF_CELL / PAGE_CELL) * (PAGE_CELL / ALL))

print()
print("=" * 100)
print("9. HOST RESIDUAL")
print("=" * 100)
show("zero-device residual (s)", HOST_ZERODEV)
show("in-bracket dispatch (s)", HOST_INBRACKET, "closed at a 1.050x ceiling, NOT zero-device")
show("total (s)", HOST_ZERODEV + HOST_INBRACKET)
print("  the b2z2-host-residual-zero brief says '~3.2 s of the fold has ZERO device work under it'.")
print(f"  measured zero-device is {HOST_ZERODEV} s; the brief's premise is {3.2-HOST_ZERODEV:.3f} s too large.")

print()
print("=" * 100)
print("10. CONTEXT.md SECTION 1 vs THE MEASURED ARM")
print("=" * 100)
print(f"  CONTEXT: diffusion 32.5179 ms x 200 = {DIFF_FOLD:.4f} s device")
print(f"  INTEG:   sampler phase, 200 steps, WALL = {SAMP_200:.4f} s")
print(f"  a device-only cost cannot exceed the wall of the phase containing it: overshoot "
      f"{DIFF_FOLD - SAMP_200:.4f} s")
print(f"  CONTEXT: PairformerLayer 41.4152 ms x 280 = 11.60 s -- retired, current main is "
      f"{PAIRFORMER_FOLD} s (1.1368x on the block)")

json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
print("\nwrote", __file__.replace(".py", ".json"))

# ---------------------------------------------------------------------------------------------
# 11. The refinement wave 1's rows 2-4 do not make: a movement-free DEVICE cannot delete the HOST
# dispatch that is exposed INSIDE the same wall-clock phase. b2z-host-residual-kill measured it
# per bracket: trunk 0.355 s, sampler 0.647 s, confidence 0.134 s (state/b2z-host-residual-kill.md).
# ---------------------------------------------------------------------------------------------
DISP_TRUNK, DISP_SAMP = 0.355, 0.647
print()
print("=" * 100)
print("11. THE SAME BOUNDS WITH IN-BRACKET HOST DISPATCH HELD FIXED")
print("=" * 100)
def floor_phase(wall, disp, mult):
    return (wall - disp) * mult + disp
tf2 = floor_phase(TRUNK_ALL, DISP_TRUNK, M_FREE)
sf2_opt = floor_phase(SAMP_200, DISP_SAMP, M_FREE)
sf2_con = floor_phase(SAMP_200, DISP_SAMP, DIFF_NOKERNEL + (1 - DIFF_NOKERNEL) * M_FREE)
show("trunk floor, host held (s)", tf2, f"vs {tf:.3f} s without the correction")
show("sampler floor, optimistic (s)", sf2_opt)
show("sampler floor, conservative (s)", sf2_con)
for lab, f in (("trunk only (wave 1's row 4)", tf2 + NONTRUNK200),
               ("trunk + sampler, optimistic", tf2 + sf2_opt + REST),
               ("trunk + sampler, conservative", tf2 + sf2_con + REST)):
    print(f"  {lab:<32} {f:>7.3f} s   vs ref {REF_CELL/f:.3f}x   vs live {PAGE_CELL/f:.3f}x"
          f"   vs paired incumbent {BASE/f:.3f}x")

print()
print("=" * 100)
print("12. 2x AGAINST THE THREE CANDIDATE DENOMINATORS")
print("=" * 100)
for name, cell, why in (
        ("campaign reference", REF_CELL, "b2x-baseline-attrib re-measure; the page never published it"),
        ("live page cell", PAGE_CELL, "site/data/perf-512aa.json today; the page calls it an upper bound"),
        ("paired incumbent", BASE, "same card, same run, same process as the 18.594 s arm")):
    print(f"  {name:<20} {cell:>7.3f} s -> 2x = {cell/2:>7.3f} s; from 18.594 s another {ALL/(cell/2):.4f}x   ({why})")
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)

# ---------------------------------------------------------------------------------------------
# 13. Why CONTEXT's 6.5036 s sampler is stale, with two independent live numbers that agree.
# PAGE ref: "the two levers are worth 1.1054x on the fold and 1.4008x on the diffusion sampler
# alone, 7.396 s to 5.280 s" (fc7fed56, merged to main).
# ---------------------------------------------------------------------------------------------
PAGE_SAMP_BEFORE, PAGE_SAMP_AFTER = 7.396, 5.280
print()
print("=" * 100)
print("13. THE SAMPLER TERM, LIVE vs CONTEXT")
print("=" * 100)
print(f"  PAGE  sampler, both diffusion levers off -> on: {PAGE_SAMP_BEFORE} -> {PAGE_SAMP_AFTER} s "
      f"({PAGE_SAMP_BEFORE/PAGE_SAMP_AFTER:.4f}x)")
print(f"  INTEG sampler phase at 200 steps, card 3:      {SAMP_200} s   "
      f"(agrees with the page to {100*abs(SAMP_200-PAGE_SAMP_AFTER)/PAGE_SAMP_AFTER:.2f} %)")
print(f"  CONTEXT b2z2 section 1:                        {DIFF_FOLD} s   "
      f"({100*(DIFF_FOLD-SAMP_200)/SAMP_200:.1f} % above the live measurement)")
print(f"  live device sum-of-units: pairformer {PAIRFORMER_FOLD} + sampler {SAMP_200} + MSA ~1.92 = "
      f"{PAIRFORMER_FOLD + SAMP_200 + 1.92:.3f} s, against CONTEXT's 'device 19.362 s'")
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)

# ---------------------------------------------------------------------------------------------
# 14. Closing the two budgets that did not close. b2z2-orchestrator pass 1 handed this to me:
#     (a) the three blocks sum to 20.0245 s against a stated 19.362 s device total (+0.662 s);
#     (b) 19.362 + 3.2 = 22.562 against 23.841 leaves 1.279 s unattributed.
# Both are artifacts of mixing three trees. Rebuild the budget inside ONE measured run.
# ---------------------------------------------------------------------------------------------
PF_BLOCK_MS, PF_CALLS = 36.4994, 280      # b2z-kernel-cycle-census, current main, synced wall
MSA_CALL_MS, MSA_CALLS = 120.2894, 16     # b2x census; orchestrator marks this CONTESTED
print()
print("=" * 100)
print("14. THE FOLD BUDGET, REBUILT INSIDE ONE MEASURED RUN (INTEG base arm, card 3)")
print("=" * 100)
pf = PF_BLOCK_MS * PF_CALLS / 1000
msa = MSA_CALL_MS * MSA_CALLS / 1000
rest_base = BASE - TRUNK_NOLADDER - SAMP_200
print(f"  trunk bracket wall            {TRUNK_NOLADDER:>7.4f} s   {100*TRUNK_NOLADDER/BASE:>5.1f} %   MEASURED")
print(f"    = PairformerLayer            {pf:>7.4f} s   ({PF_BLOCK_MS} ms x {PF_CALLS})")
print(f"    + MSALayer                   {msa:>7.4f} s   ({MSA_CALL_MS} ms x {MSA_CALLS}, CONTESTED)")
print(f"    + exposed host               {TRUNK_NOLADDER - pf - msa:>7.4f} s   residual, closes to "
      f"{100*abs(TRUNK_NOLADDER-pf-msa)/TRUNK_NOLADDER:.1f} %")
print(f"  sampler bracket wall          {SAMP_200:>7.4f} s   {100*SAMP_200/BASE:>5.1f} %   MEASURED "
      f"({1000*SAMP_200/200:.2f} ms x 200)")
print(f"  everything else               {rest_base:>7.4f} s   {100*rest_base/BASE:>5.1f} %   "
      f"vs {HOST_ZERODEV} s zero-device measured on card 2")
print(f"  ---------------------------------------------------")
print(f"  fold                          {BASE:>7.4f} s   MEASURED, closes by construction")
print()
print("  Why the old budget did not close: 41.4152 / 32.5179 / 120.2894 ms are trace-replay costs")
print("  from the b2x tree, and 19.362 s was a fold-level derivation from a third measurement.")
print(f"  Their sum 20.0245 s exceeds a 19.362 s total because the parts are older than the whole.")
print(f"  On one tree and one run the budget closes to {abs(TRUNK_NOLADDER-pf-msa):.3f} s.")
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
