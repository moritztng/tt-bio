#!/usr/bin/env python3
"""Split the math thread's input-tile wait into arrival, dependency and buffer terms.

Reads tt-metal's raw device profiler log (``profile_log_device.csv``), pairs the
``B2Z2-*`` zone begins and ends written by ``instrument_kernels.py``, and reports, per
Pairformer block, how long the trunk's matmul dataflow cores spent in each.

The zones nest: ``B2Z2-RDBAR`` is inside ``B2Z2-INx-SRC``. Issue time is SRC - RDBAR.

Usage: arrival_split.py <profile_log_device.csv> --reps N [--ops-csv ops_perf_results.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ZONE = re.compile(r"^B2Z2-")


def load(path: Path):
    """Yield dicts from a device profiler log, skipping its two-line preamble."""
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith("PCIe slot"))
    rdr = csv.DictReader(lines[start:], skipinitialspace=True)
    for row in rdr:
        yield {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
               for k, v in row.items() if k}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--freq-mhz", type=float, default=1000.0,
                    help="Tensix clock; WH galaxy is 1.0 GHz, BH p300c 1.35 GHz")
    ap.add_argument("--gcc-min", type=int, default=0,
                    help="only count programs whose run host ID (= ops CSV GLOBAL CALL COUNT) "
                         "is in [gcc-min, gcc-max]; this is how the profiled window is cut out "
                         "of a log that also contains the precursor fold")
    ap.add_argument("--gcc-max", type=int, default=1 << 62)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = list(load(a.log))
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    cols = {c.lower(): c for c in rows[0]}

    def col(*cands):
        for c in cands:
            for lc, orig in cols.items():
                if c in lc:
                    return orig
        return None

    c_zone = col("zone name")
    c_phase = col("zone phase") or cols.get("type")
    c_time = col("time[cycles", "time [cycles", "cycles since reset")
    c_x, c_y = col("core_x"), col("core_y")
    c_risc = col("risc processor type", "risc")
    c_runid = col("run host id") or col("run id")
    missing = [n for n, c in [("zone", c_zone), ("phase", c_phase), ("time", c_time),
                              ("x", c_x), ("y", c_y), ("risc", c_risc)] if c is None]
    if missing:
        print("missing columns " + ",".join(missing) + " in " + ",".join(rows[0]),
              file=sys.stderr)
        return 1

    # pair begin/end per (core, risc, run, zone); the profiler emits them in order
    open_stack = defaultdict(list)
    total = defaultdict(int)
    count = defaultdict(int)
    cores = defaultdict(set)
    percore = defaultdict(lambda: defaultdict(int))
    for r in rows:
        z = r[c_zone]
        if not z or not ZONE.match(z):
            continue
        gid = r.get(c_runid, "")
        if not (gid.isdigit() and a.gcc_min <= int(gid) <= a.gcc_max):
            continue
        key = (r[c_x], r[c_y], r[c_risc], r.get(c_runid, ""), z)
        t = int(r[c_time])
        ph = r[c_phase].lower()
        if ph.startswith("beg") or ph.endswith("start"):
            open_stack[key].append(t)
        elif open_stack[key]:
            dt = t - open_stack[key].pop()
            total[z] += dt
            count[z] += 1
            core = (r[c_x], r[c_y], r[c_risc])
            cores[z].add(core)
            percore[z][core] += dt

    unclosed = sum(len(v) for v in open_stack.values())
    ns = 1000.0 / a.freq_mhz
    res = {}
    for z in sorted(total):
        n_cores = len(cores[z]) or 1
        vals = sorted(percore[z].values())
        res[z] = {"summed_us_per_block": round(total[z] * ns / a.reps / 1000.0, 4),
                  "us_per_core_per_block": round(total[z] * ns / a.reps / n_cores / 1000.0, 4),
                  "max_core_us_per_block": round(vals[-1] * ns / a.reps / 1000.0, 4),
                  "entries_per_block": round(count[z] / a.reps, 1),
                  "cores": len(cores[z])}

    def g(z, f="us_per_core_per_block"):
        return res.get(z, {}).get(f, 0.0)

    derived = {
        "IN0-SRC_us": round(g("B2Z2-IN0-SRC"), 4),
        "IN1-SRC_us": round(g("B2Z2-IN1-SRC"), 4),
        "RDBAR_us": round(g("B2Z2-RDBAR"), 4),
        "CHAINWAIT_us": round(g("B2Z2-IN0-CHAINWAIT") + g("B2Z2-IN1-CHAINWAIT"), 4),
        "DOWNWAIT_us": round(g("B2Z2-IN0-DOWNWAIT") + g("B2Z2-IN1-DOWNWAIT"), 4),
        "FWD_us": round(g("B2Z2-IN0-FWD") + g("B2Z2-IN1-FWD"), 4),
        "CBRES_us": round(g("B2Z2-IN0-CBRES") + g("B2Z2-IN1-CBRES"), 4),
    }
    src = derived["IN0-SRC_us"] + derived["IN1-SRC_us"]
    derived["SRC_total_us"] = round(src, 4)
    derived["SRC_issue_minus_barrier_us"] = round(src - derived["RDBAR_us"], 4)
    out = {"reps": a.reps, "gcc_window": [a.gcc_min, a.gcc_max], "freq_mhz": a.freq_mhz, "rows": len(rows),
           "unclosed_zone_begins": unclosed, "zones": res, "derived": derived}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
