#!/usr/bin/env python3
"""Which arm is closer to a HIGHER-PRECISION trunk: the shipped fall-back, or the lever?

Two references, deliberately one from each arm's own kernel family, because either alone would
flatter its own side:

  refFUSED  lever ON + one_k_chunk   the fused route with no running-max rescale
  refMAT    lever OFF + accurate_softmax   the materialised route with the better softmax

If both references rank the arms the same way, that is a direction. If each flatters its own
family, the test is inconclusive and says so.
"""
import json, math, sys


def probes(path):
    rec = json.loads(open(path).readline())
    return {t["i"]: t["probe"] for t in rec["tensors"]}


def rel(a, b):
    n = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    d = math.sqrt(sum(y * y for y in b))
    return n / d if d else float("nan")


off, on, rf, rm = (probes(p) for p in sys.argv[1:5])
NAME = {0: "s", 1: "z"}
print(f"{'':6s} {'vs refFUSED (one_k_chunk)':>28s} {'vs refMAT (accurate_softmax)':>30s}")
verdict = {}
for i in sorted(off):
    a, b = rel(off[i], rf[i]), rel(on[i], rf[i])
    c, d = rel(off[i], rm[i]), rel(on[i], rm[i])
    # A reference at distance 0 from one arm IS that arm, and "closer to itself" is not a
    # reading. refFUSED came back exactly 0 from ON at 832: the one-chunk config does not fit
    # there, so the ladder falls back to the same (416, 416) pick and the reference collapses
    # onto the arm it was meant to grade. Refuse to score a degenerate reference.
    DEG = 0.0
    wf = ("DEGENERATE" if min(a, b) <= DEG else
          "ON closer" if b < a else ("OFF closer" if a < b else "tie"))
    wm = ("DEGENERATE" if min(c, d) <= DEG else
          "ON closer" if d < c else ("OFF closer" if c < d else "tie"))
    verdict[NAME[i]] = (wf, wm)
    print(f"{NAME[i]:6s}  OFF {a:.4e} / ON {b:.4e}  -> {wf:10s}"
          f"   OFF {c:.4e} / ON {d:.4e}  -> {wm}")

print()
for t, (wf, wm) in verdict.items():
    live = [w for w in (wf, wm) if w not in ("DEGENERATE", "tie")]
    if len(live) < 2:
        print(f"{t}: only {len(live)} usable reference ({wf} / {wm}) -> NOT a direction; a "
              f"single-family reference flatters its own arm")
    elif wf == wm:
        print(f"{t}: both usable references agree -> {wf}")
    else:
        print(f"{t}: usable references DISAGREE ({wf} / {wm}) -> inconclusive")
