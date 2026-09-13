#!/usr/bin/env python3
"""Upper bound on what perfect dataflow work could return, from k10-instrument's measured split.

Host only, no device, no I/O. The inputs are six measured numbers per op class and they are quoted
in the source so the arithmetic can be checked without rerunning anything.

The campaign was sized on a "3.4x envelope if circular-buffer stalls were eliminated". k10-instrument
measured, per op class, BOTH how long the compute cluster is blocked on input AND how long the reader
is actually waiting on DRAM/NoC. Those bound each other: making memory infinitely fast can remove at
most the time the reader spends waiting for memory, and it can relieve at most the compute stall that
exists. So per class the recoverable ceiling is min(compute input stall, reader NoC wait).

ASSUMPTION, stated because it is the weak point and the red team is asked to attack it: reader
NoC-wait converts to compute input-stall no worse than one for one. If a late tile can idle the
consumer for longer than the reader waited -- a granularity effect -- this bound is too tight.
It is an UPPER bound on dataflow work only; compute-side and kernel-internal levers are not bounded
by it, and the diffusion third of the fold is not measured at all.
"""

FOLD_S = 17.989          # published 512 aa cell, site/data/perf-512aa.json
PAIRFORMER_S = 10.22     # pairformer track per fold, campaign brief
BLOCK_MS = 35.3360       # k10-instrument, one PairformerLayer, 3 fenced reps, 266 programs/rep

# op class -> (ms per rep, compute wait-front %, reader NCRISC noc-read %)
OPS = {
    "GenericOp (fused trimul+TriAtt)": (13.5101, 58.0, 22.6),
    "Matmul":                          (8.1519, 29.3, 11.2),
    "BinaryNg":                        (5.9594, 79.0, 52.8),
    "LayerNorm":                       (3.8138, 57.2, 18.2),
    "Transpose":                       (2.1998, 22.2, 24.9),
}

print(f"fold {FOLD_S} s, pairformer track {PAIRFORMER_S} s, block {BLOCK_MS} ms/rep\n")
print(f"{'op class':34s} {'s/fold':>7} {'stall%':>7} {'noc%':>6} {'bound%':>7} {'s recoverable':>14}")
print("-" * 82)
total = 0.0
for name, (ms, stall, noc) in OPS.items():
    s_fold = ms / BLOCK_MS * PAIRFORMER_S
    bound = min(stall, noc)          # memory work cannot remove more than either side allows
    rec = s_fold * bound / 100.0
    total += rec
    print(f"{name:34s} {s_fold:7.3f} {stall:6.1f}% {noc:5.1f}% {bound:6.1f}% {rec:14.3f}")

print("-" * 82)
print(f"{'top five, total':34s} {'':7} {'':7} {'':6} {'':7} {total:14.3f}")
print()
print(f"fold if ALL of it were removed at zero cost: {FOLD_S - total:.3f} s"
      f"  ->  {FOLD_S / (FOLD_S - total):.4f}x")
print(f"the campaign was sized on                  : 3.4x")
print()
print(f"to reach 10 s from {FOLD_S} s needs {FOLD_S / 10.0:.4f}x, i.e. "
      f"{FOLD_S - 10.0:.3f} s must come out.")
print(f"perfect dataflow work on the whole pairformer track supplies at most {total:.3f} s of that, "
      f"{100 * total / (FOLD_S - 10.0):.1f} %.")
