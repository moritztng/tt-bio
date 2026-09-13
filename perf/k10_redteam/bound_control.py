#!/usr/bin/env python3
"""The control k10-orchestrator's thread-attribution falsifier did not run.

Its argument: a DeviceZoneScopedSumN cannot exceed its own thread's duration, and
`wait_front > TRISC0` holds 0/4441 while `wait + reserve > TRISC1` holds 1010/4441,
therefore wait-front belongs to TRISC0 and not TRISC1.

That is only evidence if the TRISC1 bound is a DIFFERENT test. If TRISC0 and TRISC1
durations are within a percent of each other, `wait_front > TRISC1` would also return
~0 and the 0/4441 proves nothing about ownership. So run both, plus the reserve-back
half against all three threads, and report the TRISC0/TRISC1 duration spread.

Result on b2z2-whglx-profiler-build's WH block capture, 4441 compute rows:
    wf > TRISC0     0      rb > TRISC2     0
    wf > TRISC1    51      rb > TRISC1     0
    wf > TRISC2   234      rb > TRISC0     0
    wf+rb > TRISC1  1010   wf+rb > TRISC0  975
    TRISC0/TRISC1 duration: min 0.388, p50 0.996, p99 4.52, max 21.17; TRISC0 longer on 1266 rows

So the wait-front claim survives a test built to break it, and the reserve-back claim
is vacuous from the data alone: rb exceeds no thread's duration, so 0/4441 against
TRISC2 is not a discriminating observation.

Usage: bound_control.py <ops_perf_results.csv[.gz]>
"""
import csv
import gzip
import sys

T = {n: f"DEVICE TRISC{n} KERNEL DURATION [ns]" for n in (0, 1, 2)}
WF = "DEVICE COMPUTE CB WAIT FRONT [ns]"
RB = "DEVICE COMPUTE CB RESERVE BACK [ns]"


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main(path):
    op = gzip.open if path.endswith(".gz") else open
    rows = list(csv.DictReader(op(path, "rt")))
    tests = ["wf>T0", "wf>T1", "wf>T2", "rb>T0", "rb>T1", "rb>T2", "wf+rb>T0", "wf+rb>T1"]
    hits = dict.fromkeys(tests, 0)
    ratio, n = [], 0
    for r in rows:
        t = {k: num(r[v]) for k, v in T.items()}
        cc, wf, rb = num(r["CORE COUNT"]), num(r[WF]), num(r[RB])
        if None in (*t.values(), cc, wf, rb) or not t[1] or cc <= 0:
            continue
        n += 1
        wf, rb = wf / cc, rb / cc  # each sum accumulates over every core in the grid
        for k, v in (("wf", wf), ("rb", rb), ("wf+rb", wf + rb)):
            for j in (0, 1, 2):
                key = f"{k}>T{j}"
                if key in hits and v > t[j]:
                    hits[key] += 1
        ratio.append(t[0] / t[1])
    ratio.sort()
    print(f"{path}\n  rows in file {len(rows)}, compute rows {n}")
    for k in tests:
        print(f"  {k:<10} {hits[k]:5d} / {n}")
    print(f"  TRISC0/TRISC1 duration: min {ratio[0]:.3f} p50 {ratio[n // 2]:.3f} "
          f"p99 {ratio[int(n * .99)]:.3f} max {ratio[-1]:.3f}; "
          f"TRISC0 longer on {sum(x > 1 for x in ratio)} rows")


if __name__ == "__main__":
    main(sys.argv[1])
