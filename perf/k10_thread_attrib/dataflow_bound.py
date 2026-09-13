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

# op class -> (ms/rep, compute wait-front %, reader DRAM %, reader ISSUING %)
# k10-instrument's closing five-zone capture (SUM_COUNT 4 -> 6), qb2 card 0.
OPS = {
    "GenericOp (fused trimul+TriAtt)": (13.4468, 57.9, 22.7, 25.6),
    "Matmul":                          (8.1630, 29.4, 11.2, 4.3),
    "BinaryNg":                        (5.9451, 79.1, 52.7, 20.2),
    "LayerNorm":                       (3.6400, 59.5, 18.8, 22.4),
    "Transpose":                       (2.2057, 22.1, 24.8, 13.5),
}

print(f"fold {FOLD_S} s, pairformer track {PAIRFORMER_S} s, block {BLOCK_MS} ms/rep")
print("TIGHT bound = min(compute stall, reader DRAM).  Bandwidth/latency work only.")
print("LOOSE bound = min(compute stall, reader DRAM + reader ISSUING).  Also credits driving")
print("              address-generation and NoC-issue time to ZERO, which no real change does.\n")
print(f"{'op class':34s} {'s/fold':>7} {'stall%':>7} {'dram%':>6} {'issue%':>7} "
      f"{'tight s':>8} {'loose s':>8}")
print("-" * 90)
tight = loose = 0.0
for name, (ms, stall, dram, issue) in OPS.items():
    s_fold = ms / BLOCK_MS * PAIRFORMER_S
    t_pct = min(stall, dram)
    l_pct = min(stall, dram + issue)
    tight += s_fold * t_pct / 100.0
    loose += s_fold * l_pct / 100.0
    print(f"{name:34s} {s_fold:7.3f} {stall:6.1f}% {dram:5.1f}% {issue:6.1f}% "
          f"{s_fold*t_pct/100:8.3f} {s_fold*l_pct/100:8.3f}")

print("-" * 90)
print(f"{'top five, total':34s} {'':7} {'':7} {'':6} {'':7} {tight:8.3f} {loose:8.3f}")
print()
for label, tot in (("TIGHT", tight), ("LOOSE", loose)):
    print(f"{label:5s}: remove it all at zero cost -> {FOLD_S - tot:6.3f} s  "
          f"= {FOLD_S / (FOLD_S - tot):.4f}x")
print(f"{'':5s}  the campaign was sized on 3.4x")
print()
need = FOLD_S - 10.0
print(f"to reach 10 s from {FOLD_S} s needs {FOLD_S / 10.0:.4f}x, i.e. {need:.3f} s must come out.")
print(f"  producer-side work supplies at most {tight:.3f}-{loose:.3f} s of that, "
      f"{100 * tight / need:.1f}-{100 * loose / need:.1f} %.")
