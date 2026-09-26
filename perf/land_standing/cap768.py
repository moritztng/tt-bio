#!/usr/bin/env python3
"""The blast radius of the cap variant actually run on the device: 1024 -> 768.

Lowering the cap all the way to 32 is the extreme and it changes 12 picks. 768 is the value that
served 832 in `perf/land_standing/out/singlek832/`, and it is the fair comparison against the
lever, which opens 832 and changes nothing.
"""
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing/perf/land_standing")
import capreach as C                                               # noqa: E402

LENGTHS = list(range(C.TILE, 1536 + C.TILE, C.TILE))

for newcap in (768, 800):
    changed, opened, lost = [], [], []
    for n in LENGTHS:
        a, b = C.serves(n), C.serves(n, cap=newcap)
        if a is None and b is not None:
            opened.append((n, b[1], b[2]))
        elif a is not None and b is None:
            lost.append(n)
        elif a and b and (a[1], a[2]) != (b[1], b[2]):
            changed.append((n, a[1:], b[1:]))
    print("\ncap %d -> %d" % (C.CAP, newcap))
    print("  holes closed:  %d  %s" % (len(opened), opened))
    print("  picks changed: %d  %s" % (len(changed), changed))
    print("  pairs lost:    %d  %s" % (len(lost), lost))
    of3 = [n for n, _, _ in changed if n % 64 == 0]
    print("  of the changed, presentable by OpenFold3 (n %% 64 == 0): %s" % of3)

print("\nfor comparison, TT_BIO_TRIATT_DIVIDING_K at the shipped cap:")
changed, opened = [], []
for n in LENGTHS:
    a, b = C.serves(n), C.serves(n, dividing_k=True)
    if a is None and b is not None:
        opened.append((n, b[1], b[2]))
    elif a and b and (a[1], a[2]) != (b[1], b[2]):
        changed.append((n, a[1:], b[1:]))
print("  holes closed:  %d  %s" % (len(opened), opened))
print("  picks changed: %d  %s" % (len(changed), changed))
