#!/usr/bin/env python3
"""Which thread does a CB stall counter belong to? Answer it from the data, not from a label.

Host only, read-only, opens no device. Point it at a `--enable-sum-profiling` ops CSV.

`b2z-kernel-cycle-census` reported the Boltz-2 pairformer block as "the math thread resident 88.4 %
and stalled on a circular buffer for two thirds of that", and recorded an error bar: "the per-core
normalisation is violated on 50 of 232 ops, so that split is directional, not exact". Both the
headline and the error bar come from `perf/b2z2_profiler/cb_split.py`, which computes

    compute_ms = TRISC1_duration - WAIT_FRONT/cores - RESERVE_BACK/cores

and divides all three terms by `DEVICE TRISC1 KERNEL DURATION`.

Those are three different threads. In tt-metal v0.68.0 the two sum accumulators are declared in
    tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_io/llk_io_unpack.h:  CB-COMPUTE-WAIT-FRONT
    tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_io/llk_io_pack.h:    CB-COMPUTE-RESERVE-BACK
i.e. on the UNPACK thread (TRISC0) and the PACK thread (TRISC2). Neither is a TRISC1 counter.

A sum accumulator measures time spent inside its own thread's kernel, so it can never exceed that
thread's duration. That gives a falsifier that needs no source access at all, and this script runs
it: if WAIT_FRONT belongs to TRISC0 it is bounded by TRISC0 on every row, and the rows where
WAIT+RESERVE exceeds TRISC1 are not a normalisation defect but the wrong divisor.

Prints both normalisations so the size of the correction is visible rather than asserted.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys

CORES = "CORE COUNT"
WAIT = "DEVICE COMPUTE CB WAIT FRONT [ns]"
RES = "DEVICE COMPUTE CB RESERVE BACK [ns]"
T0 = "DEVICE TRISC0 KERNEL DURATION [ns]"
T1 = "DEVICE TRISC1 KERNEL DURATION [ns]"
T2 = "DEVICE TRISC2 KERNEL DURATION [ns]"


def num(row: dict, col: str) -> float:
    try:
        return float(row.get(col, "") or 0.0)
    except ValueError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="ops_perf_results CSV from a --enable-sum-profiling run (.csv or .csv.gz)")
    a = ap.parse_args()

    op = gzip.open if str(a.csv).endswith(".gz") else open
    rows = list(csv.DictReader(op(a.csv, "rt")))
    comp = [r for r in rows if num(r, T1) > 0 and num(r, WAIT) + num(r, RES) > 0]
    if not comp:
        print(f"{a.csv}: no compute rows carrying stall counters. Was --enable-sum-profiling on?")
        return 2

    s = dict.fromkeys(("wait", "res", "t0", "t1", "t2"), 0.0)
    bad_t1 = bad_t0 = bad_t2 = 0
    for r in comp:
        n = max(num(r, CORES), 1.0)
        wi, wo = num(r, WAIT) / n, num(r, RES) / n
        t0, t1, t2 = num(r, T0), num(r, T1), num(r, T2)
        s["wait"] += wi; s["res"] += wo
        s["t0"] += t0; s["t1"] += t1; s["t2"] += t2
        bad_t1 += (wi + wo) > t1
        bad_t0 += wi > t0
        bad_t2 += wo > t2

    n = len(comp)
    print(f"{a.csv}: {len(rows)} rows, {n} compute rows carrying stall counters\n")
    print(f"thread duration sums (ms): TRISC0 {s['t0']/1e6:.3f}  TRISC1 {s['t1']/1e6:.3f}  "
          f"TRISC2 {s['t2']/1e6:.3f}   (TRISC0 is {100*s['t0']/s['t1']:.2f} % of TRISC1)")
    print(f"per-core-normalised stalls (ms): wait_front {s['wait']/1e6:.3f}  "
          f"reserve_back {s['res']/1e6:.3f}\n")

    print("AS THE CENSUS NORMALISED IT -- every term divided by TRISC1:")
    print(f"  wait_front   / TRISC1 = {100*s['wait']/s['t1']:5.1f} %")
    print(f"  reserve_back / TRISC1 = {100*s['res']/s['t1']:5.1f} %")
    print(f"  remainder called 'not stalled' = {100*(s['t1']-s['wait']-s['res'])/s['t1']:5.1f} %"
          "   <- this is TRISC1 minus two OTHER threads' stalls\n")

    print("UNIT-CORRECT -- each stall divided by the thread it is actually counted on:")
    print(f"  wait_front   / TRISC0 = {100*s['wait']/s['t0']:5.1f} %   unpack blocked on input tiles")
    print(f"  reserve_back / TRISC2 = {100*s['res']/s['t2']:5.1f} %   pack blocked on output room")
    print(f"  input:output stall ratio = {s['wait']/s['res']:.2f} : 1\n")

    print("THE FALSIFIER -- a sum accumulator cannot exceed its own thread's duration:")
    print(f"  wait+reserve > TRISC1      {bad_t1:5d} / {n}   <- what the census called a "
          "per-core normalisation defect")
    print(f"  wait_front   > TRISC0      {bad_t0:5d} / {n}")
    print(f"  reserve_back > TRISC2      {bad_t2:5d} / {n}")
    if bad_t0 == 0 and bad_t2 == 0 and bad_t1 > 0:
        print("\n  => Both counters are bounded by TRISC0 and TRISC2 respectively and by neither "
              "of\n     them is TRISC1 a bound. The anomaly is the divisor, not the normalisation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
