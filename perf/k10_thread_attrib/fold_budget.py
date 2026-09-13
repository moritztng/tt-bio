#!/usr/bin/env python3
"""Where would 7.989 s have to come from? The fold budget, from measured shares only.

Host only, no device, no I/O. Every input is a measured number from a concluded row, quoted here.

The 10 s target needs 1.7989x on the 17.989 s cell, i.e. 7.989 s removed. This asks the only
question that decides a go/stop: is there anywhere for that to come from, given what each region of
the fold has been measured to contain and what has been bounded inside it.
"""

FOLD_S = 17.989           # published 512 aa cell, site/data/perf-512aa.json:176
TARGET_S = 10.0
NEED_S = FOLD_S - TARGET_S

# region -> (seconds, what is bounded inside it, source)
PAIRFORMER_S = 10.22      # campaign brief, pairformer track per fold
DIFFUSION_FRAC = 0.325    # census: the diffusion step is 32.5 % of the fold
DIFFUSION_S = FOLD_S * DIFFUSION_FRAC
REST_S = FOLD_S - PAIRFORMER_S - DIFFUSION_S

# bounded producer-side recoverable inside the pairformer track (dataflow_bound.py)
TRUNK_TIGHT_S, TRUNK_LOOSE_S = 2.392, 4.073
# diffusion: host side is dead, 93.8 % of the loop is device (b2z2-diffusion-loop-attack)
DIFF_HOST_S = DIFFUSION_S * 0.062
DIFF_DEVICE_S = DIFFUSION_S * 0.938

print(f"fold {FOLD_S} s -> {TARGET_S} s needs {FOLD_S/TARGET_S:.4f}x, i.e. {NEED_S:.3f} s removed\n")
print(f"{'region':34s} {'s/fold':>8} {'% fold':>7}  what is known")
print("-" * 96)
print(f"{'pairformer track':34s} {PAIRFORMER_S:8.3f} {100*PAIRFORMER_S/FOLD_S:6.1f}%  "
      f"producer-side bounded at {TRUNK_TIGHT_S}-{TRUNK_LOOSE_S} s")
print(f"{'diffusion step, device':34s} {DIFF_DEVICE_S:8.3f} {100*DIFF_DEVICE_S/FOLD_S:6.1f}%  "
      f"NEVER MEASURED -- the open question")
print(f"{'diffusion step, host':34s} {DIFF_HOST_S:8.3f} {100*DIFF_HOST_S/FOLD_S:6.1f}%  "
      f"dead: --diffusion_trace 0.9948x on BH")
print(f"{'everything else':34s} {REST_S:8.3f} {100*REST_S/FOLD_S:6.1f}%  "
      f"host residual, repeatedly attacked")
print("-" * 96)

print(f"\nThe decisive arithmetic. Grant EVERY bound in full, simultaneously, at zero cost:\n")
for label, trunk in (("tight", TRUNK_TIGHT_S), ("loose", TRUNK_LOOSE_S)):
    left = NEED_S - trunk
    print(f"  trunk producer-side at its {label:5s} bound removes {trunk:5.3f} s "
          f"-> still need {left:5.3f} s")
    print(f"     and the diffusion step's ENTIRE device time is {DIFF_DEVICE_S:.3f} s, so that "
          f"shortfall is {100*left/DIFF_DEVICE_S:.1f} % of it")

print(f"\nSo 10 s requires the whole trunk producer-side bound AND {100*(NEED_S-TRUNK_LOOSE_S)/DIFF_DEVICE_S:.0f}-"
      f"{100*(NEED_S-TRUNK_TIGHT_S)/DIFF_DEVICE_S:.0f} % of the diffusion step's device time to vanish.")
print("Removing the diffusion step's work is not available: fewer steps is a cheat, not a speedup.")
print("So the honest question for the one unmeasured region is not 'is there a prize' but")
print(f"'is over half of {DIFF_DEVICE_S:.1f} s of device time recoverable without doing less work'.\n")

print("For scale, what Phase 1 actually found, composed:")
for name, r in (("bank permutation B1+B2+B3", 1.0254), ("BinaryNg class (3 sites)", 1.0094),
                ("matmul L1 residency (pred BH)", 1.0300), ("M_block 4->8 (pred BH)", 1.0100)):
    print(f"  {name:34s} {r:.4f}x")
stack = 1.0254 * 1.0094 * 1.0300 * 1.0100
print(f"  {'composed, optimistic (sub-additive in truth)':34s} {stack:.4f}x "
      f"-> {FOLD_S/stack:.2f} s")
print(f"\n  measured/predicted Phase 1 total is {100*(stack-1)/(FOLD_S/TARGET_S-1):.1f} % "
      f"of the margin 10 s requires.")

# ---------------------------------------------------------------------------------------------
# CLOSING THE BUDGET, 2026-09-13 17:31Z. The diffusion step's stall structure, measured at last.
#
# k10-p1-diffusion-measure's census (whglx card 16, grid 8x9, --enable-sum-profiling, thread cap
# applied, EXIT=0 at 17:31:18Z) run through thread_attrib.py --by-op, 11083 compute rows:
#
#     wait_front / TRISC0 = 63.1 %      reserve_back / TRISC2 = 13.6 %      in:out = 4.59 : 1
#     falsifier again 0/11083 on both counters against their own thread, 2483/11083 against TRISC1
#
# Against the pairformer block on the same part: 65.7 % / 13.3 % / 4.91:1.
# THE DIFFUSION STEP HAS THE SAME SHAPE AS THE BLOCK. There is no different regime hiding in it.
#
# The whglx profiler is STOCK, so it carries no DM-thread zones: this capture cannot say whether the
# reader is on DRAM or idle. So bound it two ways instead, and note that both are generous.
DIFF_INPUT_STALL = 0.631
BLOCK_CONV_MAX = 0.444   # the most generous measured conversion: at most 44.4 % of the block's input
                         # stall has a memory-bound producer behind it (k10-p1-redteam's recalibration
                         # of k10-instrument's 39 %, using the starved arm at 98.9 % of the DRAM roof)

print("\n" + "=" * 96)
print("CLOSING THE BUDGET: the diffusion step measured, 63.1 % input stall, same shape as the block")
print("=" * 96)
absurd = DIFF_DEVICE_S * DIFF_INPUT_STALL
realistic = absurd * BLOCK_CONV_MAX
print(f"\n  diffusion device time                                    {DIFF_DEVICE_S:6.3f} s")
print(f"  x its measured compute input stall (63.1 %)              {absurd:6.3f} s  "
      f"<- absurd ceiling: assumes ALL of it is producer-caused AND free to remove")
print(f"  x the block's measured conversion ceiling (44.4 %)       {realistic:6.3f} s  "
      f"<- the honest one")
for label, trunk in (("tight", TRUNK_TIGHT_S), ("loose", TRUNK_LOOSE_S)):
    need_from_diff = NEED_S - trunk
    print(f"\n  10 s needs {need_from_diff:5.3f} s from diffusion (trunk at its {label} bound).")
    print(f"     absurd ceiling supplies {absurd:.3f} s -> "
          f"{'ENOUGH' if absurd >= need_from_diff else 'SHORT by %.3f s' % (need_from_diff - absurd)}")
    print(f"     honest ceiling supplies {realistic:.3f} s -> "
          f"SHORT by {need_from_diff - realistic:.3f} s")
whole = TRUNK_TIGHT_S + realistic
whole_loose = TRUNK_LOOSE_S + absurd
print(f"\n  PRODUCER-SIDE CEILING FOR THE WHOLE FOLD, both regions, honest:  "
      f"{whole:.3f} s -> {FOLD_S/(FOLD_S-whole):.4f}x  ({FOLD_S-whole:.2f} s)")
print(f"  Same, granting every generous end simultaneously:                 "
      f"{whole_loose:.3f} s -> {FOLD_S/(FOLD_S-whole_loose):.4f}x  ({FOLD_S-whole_loose:.2f} s)")
print(f"\n  10 s requires {FOLD_S/TARGET_S:.4f}x. Neither reaches it.")
