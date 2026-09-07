#!/usr/bin/env python3
"""The size exponent measured from trunk recycle time instead of whole-fold wall.

The size-ladder baseline fits its exponent to `runtime_s`, the whole-fold wall. That wall carries
a large size-INDEPENDENT term -- weight load, kernel compile, host featurisation -- which
`release_gate.py:733` says biases both exponents downward, and it carries the diffusion head,
whose cost scales with the atom count rather than the token count. At the bottom of a ladder the
fixed term is most of the number: rf3 reads k=1.171 across 256->512 aa on Wormhole purely because
21 s of its 36 s at 256 aa is not folding.

A recycle of the trunk has neither problem. It is the dominant quadratic-and-worse term on its
own, it repeats ten times per fold so its spread is visible, and it starts after everything fixed
has already been paid. rf3 prints one `trunk N/10` line per recycle, so the per-recycle time is
free to recover from a log that already exists.

It is also robust where the wall is not. rf3-640-rep0 on the Galaxy walled 154.7 s against 124.3 s
for the same rung on another chip, a 24 % gap that looked like a per-chip difference; its recycles
read 25, 9, 10, 9, 9, 9, 9, 9, 9, so the chip was fine and one recycle got interrupted. A median
over the recycles ignores that outright, which is why this reports the median and shows min/max
next to it rather than a mean.

The one limit worth knowing: the log stamps whole seconds, so a rung whose recycle is only a few
seconds long is coarsely quantised and its exponent should not be read tightly. Above ~10 s per
recycle the quantisation is under a percent.

Usage:
    trunk_recycle_exponent.py <dir-of-rf3-*.log>   # or a list of log files
"""
import datetime
import math
import os
import re
import statistics
import sys

LOG = re.compile(r"^(?P<model>[a-z0-9-]+)-(?P<rung>\d+)-(?P<fold>warmup|rep\d+)\.log$")
LINE = re.compile(r"(\d\d:\d\d:\d\d) .*trunk (\d+)/\d+")


def recycles(path: str) -> list[float]:
    """Seconds per trunk recycle, from the fold's own progress lines."""
    stamps = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = LINE.match(line)
            if m:
                stamps.append(datetime.datetime.strptime(m.group(1), "%H:%M:%S"))
    # A fold that crosses midnight would read negative; drop those rather than report a wrong
    # exponent from one, since a rung is cheap to re-measure and a silent sign error is not.
    return [d for d in ((b - a).total_seconds() for a, b in zip(stamps, stamps[1:])) if d >= 0]


def main(paths: list[str]) -> int:
    if len(paths) == 1 and os.path.isdir(paths[0]):
        paths = [os.path.join(paths[0], f) for f in sorted(os.listdir(paths[0]))]

    per_rung: dict[int, list[float]] = {}
    print(f"{'rung':>6} {'fold':<8} {'median':>8} {'min':>6} {'max':>6}   n")
    for path in sorted(paths):
        m = LOG.match(os.path.basename(path))
        if not m:
            continue
        d = recycles(path)
        if len(d) < 5:          # a fold that died early cannot give a median worth having
            continue
        med = statistics.median(d)
        per_rung.setdefault(int(m.group("rung")), []).append(med)
        print(f"{int(m.group('rung')):>6} {m.group('fold'):<8} {med:>8.1f} "
              f"{min(d):>6.0f} {max(d):>6.0f}  {len(d)}")

    rungs = sorted(per_rung)
    if len(rungs) < 2:
        raise SystemExit("need trunk timings from at least two rungs")
    print("\nper-recycle trunk median, and the exponent between consecutive rungs:")
    prev = None
    for rung in rungs:
        med = statistics.median(per_rung[rung])
        if prev:
            k = math.log(med / prev[1]) / math.log(rung / prev[0])
            note = "  (coarse: recycle under 10 s)" if prev[1] < 10 else ""
            print(f"  {prev[0]:>4} -> {rung:<4}  {prev[1]:>5.1f} -> {med:<5.1f} s   "
                  f"k = {k:.3f}{note}")
        prev = (rung, med)
    print("\nPair work is quadratic and triangle work is worse, so k near 2 and drifting up is "
          "the expected shape. A k that JUMPS between one pair of rungs is the finding.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1:]))
