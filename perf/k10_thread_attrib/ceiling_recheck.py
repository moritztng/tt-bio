#!/usr/bin/env python3
"""The sampler ceiling map's 2x ladder, recomputed with its refuted first rung removed.

Host only, no device, no I/O. Every input is quoted from a concluded row so the arithmetic is
checkable without rerunning anything.

`b2z2-sampler-ceiling-map` concluded that 2x "is arithmetically reachable and it has 1.2 % of
margin", landing at 10.158 s (1.9767x) against the 20.113 s cell. That stack is built on top of a
trunk held at a "movement-free floor nobody knows how to reach" (4.689 s) and a `rest` bracket held
at its measured 2.1805 s, and its FIRST rung is `--diffusion_trace`, "built, default off", credited
with 26.400 -> 23.378 ms/step.

`b2z2-diffusion-loop-attack` then measured that lever on Blackhole -- qb2 card 1, the cell's own
fixture and protocol -- at **0.9948x**: eager 19.937 s against traced 20.041 s, three folds each,
A/A spread 0.5-1.0 %, ranges overlapping, identical CIF. Under trace the loop is device wait
5024.6 ms against host-serial 332.3 ms, i.e. 93.8 % device. The mechanism is architectural: qb2
dispatches a one-tile op in 9.56 us against qb1's 22.92 us, so at ~2466 ttnn calls/step the eager
issue cost hides behind a 22.02 ms device step. The same harness reads 1.016x on qb1.

So on Blackhole that rung is worth zero, and the three levers above it start from 26.400 ms/step
rather than 23.378. This recomputes the ladder on the map's own basis and terms.

It does NOT restate the ladder for today's 17.989 s cell: the map is denominated in the 20.113 s
cell and its trunk/rest brackets belong to that tree. Structure carries, absolutes do not.
"""

CELL_S = 20.113        # the cell the ceiling map is denominated in
TRUNK_FLOOR_S = 4.689  # "a movement-free floor nobody knows how to reach"
REST_S = 2.1805        # measured `rest` bracket
STEPS = 200
TODAY_MS_STEP = 26.400

# rung -> ms/step it lands at, per the map's table
LADDER = [
    ("--diffusion_trace (built, default off)", 23.378),
    ("distributed LayerNorm",                  21.208),
    ("fuse the SDPA head plumbing",            18.454),
    ("fuse half of BinaryNg",                  16.442),
]

def fold(ms_step):
    return TRUNK_FLOOR_S + REST_S + ms_step * STEPS / 1000.0

print(f"cell {CELL_S} s   trunk floor {TRUNK_FLOOR_S} s (unreachable by anyone's account)   "
      f"rest {REST_S} s   2x = {CELL_S/2:.3f} s\n")

print("AS PUBLISHED by b2z2-sampler-ceiling-map:")
prev, deltas = TODAY_MS_STEP, []
print(f"  {'today, untraced':42s} {TODAY_MS_STEP:7.3f} ms/step  "
      f"fold {fold(TODAY_MS_STEP):7.3f} s  {CELL_S/fold(TODAY_MS_STEP):.4f}x")
for name, ms in LADDER:
    deltas.append((name, prev - ms))
    prev = ms
    print(f"  + {name:40s} {ms:7.3f} ms/step  fold {fold(ms):7.3f} s  {CELL_S/fold(ms):.4f}x")

print("\nper-rung delta (ms/step), read out of that table:")
for name, d in deltas:
    print(f"  {name:44s} {d:6.3f}")

trace_delta = deltas[0][1]
rest_deltas = sum(d for _, d in deltas[1:])
corrected = TODAY_MS_STEP - rest_deltas

print(f"\nThe trace rung is credited {trace_delta:.3f} ms/step and MEASURES 0.9948x on Blackhole.")
print("Remove it; the other three levers still land in full, starting from today's 26.400:\n")
print(f"  corrected ms/step  = {TODAY_MS_STEP:.3f} - {rest_deltas:.3f} = {corrected:.3f}")
print(f"  corrected sampler  = {corrected * STEPS / 1000.0:.3f} s")
print(f"  corrected fold     = {fold(corrected):.3f} s")
print(f"  corrected ratio    = {CELL_S / fold(corrected):.4f}x   (published: "
      f"{CELL_S / fold(LADDER[-1][1]):.4f}x)")
gap = fold(corrected) - CELL_S / 2
print(f"\n  2x target {CELL_S/2:.3f} s -> misses by {gap:+.3f} s "
      f"({100*gap/(CELL_S/2):+.1f} %), where the map reported "
      f"{fold(LADDER[-1][1]) - CELL_S/2:+.3f} s.")
print("\nAnd that is still the optimistic case: it grants a trunk floor nobody knows how to reach,")
print("and three levers that are projected rather than built.")

# Reconstruction check, stated rather than buried: this script's fold seconds reproduce the map's
# table EXACTLY at every rung (12.149 / 11.545 / 11.111 / 10.560 / 10.158 s), so the ladder is
# rebuilt faithfully. Its ratios differ from the map's in the third decimal (1.6555 vs its 1.6527,
# 1.9800 vs its 1.9767) because the map's implied denominator is ~20.079 s rather than the 20.113 s
# cell it names. That 0.17 % does not touch the conclusion, which is a difference of 0.706 s.
