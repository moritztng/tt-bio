"""Host-side cost of the size guard, matched-pair with an A/A control.

The guard runs once per CLI invocation, before any device work, so its cost is a fixed host-side
addition and not a per-step one. Measured against an A/A control (the same arm run twice) so a
delta can be told from box noise, which is the only way a number this small means anything.
"""
import statistics as st
import time
from tt_bio import size_limits as sl

SIZES = (128, 256, 512)
REPS = 200


def yaml_of(n):
    return f"sequences:\n  - protein:\n      id: A\n      sequence: {'A' * n}\n"


def timeit(fn, reps=REPS):
    xs = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); xs.append((time.perf_counter() - t) * 1e6)
    return st.median(xs)


print(f"{'size':>6} {'guard us':>10} {'A/A us':>10} {'A/A spread':>12}")
for n in SIZES:
    text = yaml_of(n)
    # arm: the whole guard, scan + check, as the CLI calls it
    guard = timeit(lambda: sl.check(  "opendde", sl.scan_residues(text), arch="wormhole_b0"))
    aa1 = timeit(lambda: sl.check("opendde", sl.scan_residues(text), arch="wormhole_b0"))
    aa2 = timeit(lambda: sl.check("opendde", sl.scan_residues(text), arch="wormhole_b0"))
    spread = abs(aa1 - aa2) / min(aa1, aa2) * 100
    print(f"{n:>6} {guard:>10.1f} {aa1:>10.1f} {spread:>11.1f}%")

# The Blackhole path: no rows, so check() returns before it reads a size.
bh = timeit(lambda: sl.check("opendde", 1024, arch="blackhole"))
wh = timeit(lambda: sl.check("opendde", 100, arch="wormhole_b0"))
print(f"\ncheck() alone: blackhole {bh:.2f} us, wormhole {wh:.2f} us")
