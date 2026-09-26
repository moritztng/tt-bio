#!/usr/bin/env python3
"""Blast radius of lifting `_Q_SPLIT_MAX_S`: which lengths change the pair they already serve?

Closing a hole is the benefit. The cost is any length that serves today and would serve a
DIFFERENT (q_chunk, k_chunk) after the lift, because k_chunk sets the online-softmax reduction
order and a changed pair is a changed digest. That is the number a cap change has to be argued
on, and it is pure host arithmetic.

Imports the validated model from capreach.py rather than restating it, so the two cannot drift.
"""
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing/perf/land_standing")
import capreach as C                                               # noqa: E402

TILE = C.TILE
CAP = C.CAP
LENGTHS = list(range(TILE, 1536 + TILE, TILE))

print("\n\nBLAST RADIUS of lowering the cap from %d to %d" % (CAP, TILE))
same, changed, opened, closed = [], [], [], []
for n in LENGTHS:
    a = C.serves(n)                 # shipping today
    b = C.serves(n, cap=TILE)       # cap lifted
    if a is None and b is None:
        continue
    if a is None:
        opened.append((n, b))
    elif b is None:
        closed.append((n, a))
    elif (a[1], a[2]) == (b[1], b[2]):
        same.append(n)
    else:
        changed.append((n, a, b))

print("  lengths that serve today and keep the IDENTICAL pair: %d" % len(same))
print("  lengths that serve today and CHANGE pair:             %d" % len(changed))
for n, a, b in changed:
    print("      n=%-5d %s -> %s" % (n, a, b))
print("  lengths that gain a fused pair (holes closed):         %d -> %s"
      % (len(opened), [(n, b[1], b[2]) for n, b in opened]))
print("  lengths that LOSE their fused pair:                    %d -> %s"
      % (len(closed), [n for n, _ in closed]))

print("\n  which of the changed/opened lengths can OpenFold3 actually present?")
print("  (its pair axis pads to a multiple of 64, measured: 684 aa -> a (704, 704) pick)")
touched = [n for n, _, _ in changed] + [n for n, _ in opened]
of3 = [n for n in touched if n % 64 == 0]
not_of3 = [n for n in touched if n % 64]
print("    presentable by OpenFold3 (n %% 64 == 0): %s" % of3)
print("    not presentable by OpenFold3:           %s" % not_of3)

print("\n  and how does that compare with TT_BIO_TRIATT_DIVIDING_K, the lever this row holds?")
dk_changed, dk_opened = [], []
for n in LENGTHS:
    a = C.serves(n)
    b = C.serves(n, dividing_k=True)
    if a is None and b is not None:
        dk_opened.append((n, b))
    elif a and b and (a[1], a[2]) != (b[1], b[2]):
        dk_changed.append((n, a, b))
print("    dividing-k opens:  %s" % [(n, b[1], b[2]) for n, b in dk_opened])
print("    dividing-k changes: %s" % [(n, a[1:], b[1:]) for n, a, b in dk_changed])
