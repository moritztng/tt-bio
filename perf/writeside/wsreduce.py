#!/usr/bin/env python3
"""Per-arm thread residency and CB stalls from a --enable-sum-profiling ops CSV.

Host only, read-only, opens no device.  Arms are separated by OP CODE + OUTPUT_0_MEMORY +
INPUT_0_X_PAD, so no call-index bookkeeping is needed and the DRAM-out and L1-out arms of the
same op fall out on their own.

Each stall accumulator is divided by THAT OP'S OWN `CORE COUNT`, never a shared constant --
`ttnn-stall-accumulator-must-divide-by-own-core-count`.  And each is divided by the thread it is
actually counted on: WAIT FRONT is declared on the unpack thread (TRISC0) and RESERVE BACK on the
pack thread (TRISC2), neither is a TRISC1 counter -- `perf/k10_thread_attrib/thread_attrib.py`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import statistics as st
import sys
from collections import defaultdict

COLS = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "DEVICE BRISC KERNEL DURATION [ns]",
        "DEVICE NCRISC KERNEL DURATION [ns]", "DEVICE TRISC0 KERNEL DURATION [ns]",
        "DEVICE TRISC1 KERNEL DURATION [ns]", "DEVICE TRISC2 KERNEL DURATION [ns]",
        "DEVICE COMPUTE CB WAIT FRONT [ns]", "DEVICE COMPUTE CB RESERVE BACK [ns]"]


def num(r, c):
    try:
        return float(r.get(c, "") or 0.0)
    except ValueError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None,
                    help="ops_perf_results CSV (.csv or .csv.gz); default: newest under prof_*/")
    ap.add_argument("--min-rows", type=int, default=5)
    a = ap.parse_args()
    path = a.csv or sorted(glob.glob("perf/writeside/prof_*/**/ops_perf_results_*.csv*",
                                     recursive=True))[-1]
    op = gzip.open if path.endswith(".gz") else open
    rows = list(csv.DictReader(op(path, "rt")))
    print("%s: %d rows\n" % (path, len(rows)))

    g = defaultdict(list)
    for r in rows:
        g[(r["OP CODE"], r.get("OUTPUT_0_MEMORY", ""), r.get("INPUT_0_X_PAD[LOGICAL]", ""))].append(r)

    for key, v in g.items():
        if len(v) < a.min_rows:
            continue
        m = {c: st.median([num(r, c) for r in v]) for c in COLS}
        cores = max(m["CORE COUNT"], 1.0)
        dur = m["DEVICE KERNEL DURATION [ns]"]
        t0, t2 = m["DEVICE TRISC0 KERNEL DURATION [ns]"], m["DEVICE TRISC2 KERNEL DURATION [ns]"]
        wf = m["DEVICE COMPUTE CB WAIT FRONT [ns]"] / cores
        rb = m["DEVICE COMPUTE CB RESERVE BACK [ns]"] / cores
        print("ARM %s  n=%d  cores=%d  op=%.2f us" % (key, len(v), cores, dur / 1e3))
        for c, lab in (("DEVICE BRISC KERNEL DURATION [ns]", "BRISC  (in1 sender + OUTPUT WRITER)"),
                       ("DEVICE NCRISC KERNEL DURATION [ns]", "NCRISC (in0)                      "),
                       ("DEVICE TRISC0 KERNEL DURATION [ns]", "TRISC0 (unpack)                   "),
                       ("DEVICE TRISC1 KERNEL DURATION [ns]", "TRISC1 (math)                     "),
                       ("DEVICE TRISC2 KERNEL DURATION [ns]", "TRISC2 (pack)                     ")):
            print("   %s %8.2f us  %5.1f %% of op" % (lab, m[c] / 1e3, 100 * m[c] / dur))
        print("   wait_front   / TRISC0 %8.2f us  %5.1f %%   unpack blocked on input tiles"
              % (wf / 1e3, 100 * wf / t0 if t0 else 0))
        print("   reserve_back / TRISC2 %8.2f us  %5.1f %%   PACK BLOCKED ON OUTPUT ROOM"
              % (rb / 1e3, 100 * rb / t2 if t2 else 0))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
