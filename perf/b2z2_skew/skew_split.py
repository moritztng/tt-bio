#!/usr/bin/env python3
"""Is the matmul operand wait a chain ramp, or an axis that arrives skewed?

`b2z2-tile-arrival-latency` measured that 67.6 % of the trunk matmul reader's own time is one core
waiting for another, and `b2z2-mcast-operand-build` measured a broadcast of the same bytes 6.6 %
SLOWER and read the loss as arrival skew: a multicast cannot send until the LAST receiver credits
it. Nobody measured the spread. Two accounts of the same number want opposite levers:

  H-RAMP   every receiver asks at about the same time and then waits for the block to walk the
           chain, so the wait is a staircase in hop index and the lever is fewer hops.
  H-SKEW   receivers ask at very different times and the chain absorbs it, so the lever has to
           make the cores arrive together and a shorter chain buys nothing.

The `B2Z2-INx-CHAINWAIT` zone brackets exactly the three statements that ask and wait, so its
BEGIN is when a core asked and its END is when its operand arrived. This pairs those per core, per
chain group, per block iteration, and reports the two spreads, the ramp fit against the hop index
`core_map.py` took from `mm_generic` itself, and how much of the wait lives in a tail off that ramp.

Usage: skew_split.py <profile_log_device.csv> --coremap coremap.json [--gcc-min N --gcc-max N]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    """Rows of a raw device profiler log, streamed.

    The log is 1.4 GB for three replays of one Pairformer block, so it is read a line at a time and
    everything without a B2Z2 zone on it is dropped before the CSV reader ever sees it.
    """
    with path.open() as fh:
        header = None
        for line in fh:
            if header is None:
                if line.lstrip().startswith("PCIe slot"):
                    header = [c.strip() for c in next(csv.reader([line]))]
                continue
            if "B2Z2" not in line:
                continue
            vals = next(csv.reader([line]))
            yield {k: (v.strip() if isinstance(v, str) else v)
                   for k, v in zip(header, vals) if k}


def spearman(xs, ys):
    """Rank correlation, ties averaged. n < 3 has no meaning here and returns 0.0."""
    if len(xs) < 3:
        return 0.0

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def linfit(xs, ys):
    """Least-squares slope, intercept and R^2 of ys on xs."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icept = my - slope * mx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (slope * x + icept)) ** 2 for x, y in zip(xs, ys))
    return slope, icept, (1.0 - ssr / sst) if sst else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--coremap", type=Path, required=True)
    ap.add_argument("--freq-mhz", type=float, default=1000.0,
                    help="Tensix clock; WH galaxy 1.0 GHz, BH p300c 1.35 GHz")
    ap.add_argument("--gcc-min", type=int, default=0)
    ap.add_argument("--gcc-max", type=int, default=1 << 62)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    cmap = json.loads(a.coremap.read_text())
    stream = load(a.log)
    try:
        first = next(stream)
    except StopIteration:
        print("no B2Z2 zone rows in log", file=sys.stderr)
        return 1
    rows = itertools.chain([first], stream)
    cols = {c.lower(): c for c in first}

    def col(*cands):
        for c in cands:
            for lc, orig in cols.items():
                if c in lc:
                    return orig
        return None

    c_zone, c_phase = col("zone name"), col("zone phase") or cols.get("type")
    c_time = col("time[cycles", "time [cycles", "cycles since reset")
    c_x, c_y, c_risc = col("core_x"), col("core_y"), col("risc processor type", "risc")
    c_run = col("run host id") or col("run id")
    if not all([c_zone, c_phase, c_time, c_x, c_y, c_risc, c_run]):
        print("missing columns in " + ",".join(first), file=sys.stderr)
        return 1

    # (axis, run, core) -> [(ask, arrive), ...] in issue order, one per block iteration
    waits = defaultdict(list)
    open_at = {}
    fwd_end = defaultdict(list)
    fwd_open = {}
    for r in rows:
        z = r[c_zone]
        if not z or not z.startswith("B2Z2-IN"):
            continue
        run = r[c_run]
        if not (run.isdigit() and a.gcc_min <= int(run) <= a.gcc_max):
            continue
        axis = z.split("-")[1].lower()          # in0 / in1
        core = (int(r[c_x]), int(r[c_y]))
        t, ph = int(r[c_time]), r[c_phase].lower()
        begin = ph.startswith("beg") or ph.endswith("start")
        if z.endswith("CHAINWAIT"):
            key = (axis, run, core)
            if begin:
                open_at[key] = t
            elif key in open_at:
                waits[key].append((open_at.pop(key), t))
        elif z.endswith("FWD"):
            key = (axis, run, core)
            if begin:
                fwd_open[key] = t
            elif key in fwd_open:
                fwd_open.pop(key)
                fwd_end[key].append(t)

    if not waits:
        print("no CHAINWAIT zones in window", file=sys.stderr)
        return 1

    ns = 1000.0 / a.freq_mhz
    # group cores of one chain together: (axis, run, chain group) -> {core: [(ask, arr), ...]}
    groups = defaultdict(dict)
    unmapped = 0
    for (axis, run, core), v in waits.items():
        ent = cmap[axis].get(f"{core[0]},{core[1]}")
        if ent is None:
            unmapped += 1
            continue
        groups[(axis, run, ent["group"])][core] = (ent["hop"], v)

    samples = []          # one per (axis, run, group, iteration)
    hop_wait = defaultdict(list)
    causality_checked = causality_bad = 0
    for (axis, run, grp), members in groups.items():
        if len(members) < 3:
            continue
        n_iter = min(len(v) for _, v in members.values())
        for it in range(n_iter):
            asks, arrs, hops, per = [], [], [], []
            for core, (hop, v) in members.items():
                ask, arr = v[it]
                asks.append(ask)
                arrs.append(arr)
                hops.append(hop)
                per.append((hop, arr - ask, core, ask, arr))
                hop_wait[(axis, hop)].append((arr - ask) * ns / 1000.0)
            t0 = min(arrs)
            rel = [(h, (arr - t0) * ns / 1000.0) for h, _, _, _, arr in per]
            sl, ic, r2 = linfit([h for h, _ in rel], [w for _, w in rel])
            samples.append({
                "axis": axis, "group": grp, "iter": it, "cores": len(per),
                "ask_spread_us": (max(asks) - min(asks)) * ns / 1000.0,
                "arr_spread_us": (max(arrs) - min(arrs)) * ns / 1000.0,
                "sum_wait_us": sum(w for _, w, _, _, _ in per) * ns / 1000.0,
                "max_wait_us": max(w for _, w, _, _, _ in per) * ns / 1000.0,
                "ramp_slope_us_per_hop": sl, "ramp_r2": r2,
                "rho": spearman([h for h, _ in rel], [w for _, w in rel])})
            # causality control: cross-core timestamps are only comparable if a core's operand
            # never arrives before the predecessor that forwards it finished forwarding.
            by_hop = {h: (arr, core) for h, _, core, _, arr in per}
            for h, (arr, core) in by_hop.items():
                prev = [(c, hp) for c, (hp, _) in members.items() if hp == h - 1]
                if not prev:
                    continue
                pc = prev[0][0]
                ends = fwd_end.get((axis, run, pc)) or []
                if len(ends) > it:
                    causality_checked += 1
                    if arr < ends[it] - 200:      # 200 cycles of slack for the zone's own cost
                        causality_bad += 1

    if not samples:
        print("no chain groups with >= 3 cores", file=sys.stderr)
        return 1

    def med(k):
        return round(st.median([s[k] for s in samples]), 4)

    ask_over_arr = [s["ask_spread_us"] / s["arr_spread_us"]
                    for s in samples if s["arr_spread_us"] > 0]
    spread_over_wait = [s["arr_spread_us"] / s["sum_wait_us"]
                        for s in samples if s["sum_wait_us"] > 0]
    # the ramp, fitted once over every (hop, wait) sample rather than per iteration
    hop_mean = {k: st.mean(v) for k, v in sorted(hop_wait.items())}
    per_axis = {}
    for axis in ("in0", "in1"):
        hs = [h for (ax, h) in hop_mean if ax == axis]
        if len(hs) < 3:
            continue
        xs = sorted(hs)
        ys = [hop_mean[(axis, h)] for h in xs]
        sl, ic, r2 = linfit(xs, ys)
        allx = [h for (ax, h), v in hop_wait.items() if ax == axis for _ in v]
        ally = [w for (ax, h), v in hop_wait.items() if ax == axis for w in v]
        sl_a, ic_a, r2_a = linfit(allx, ally)
        resid = [(w - (sl_a * h + ic_a)) for h, w in zip(allx, ally)]
        sd = st.pstdev(resid) if len(resid) > 2 else 0.0
        tail = sum(w for h, w, rr in zip(allx, ally, resid) if sd and rr > 2 * sd)
        per_axis[axis] = {
            "hops": xs, "mean_wait_us_by_hop": [round(y, 4) for y in ys],
            "n_by_hop": [len(hop_wait[(axis, h)]) for h in xs],
            "rho_hop_vs_mean_wait": round(spearman(xs, ys), 4),
            "ramp_slope_us_per_hop": round(sl, 5), "ramp_intercept_us": round(ic, 5),
            "r2_on_hop_means": round(r2, 4), "r2_on_all_samples": round(r2_a, 4),
            "tail_share_of_wait": round(tail / sum(ally), 4) if sum(ally) else 0.0,
            "total_wait_us": round(sum(ally), 3)}

    res = {
        "samples": len(samples), "unmapped_cores": unmapped,
        "causality": {"checked": causality_checked, "violations": causality_bad,
                      "note": "a core's operand arriving before its predecessor finished "
                              "forwarding would mean cross-core timestamps are not comparable"},
        "median_ask_spread_us": med("ask_spread_us"),
        "median_arr_spread_us": med("arr_spread_us"),
        "median_sum_wait_us": med("sum_wait_us"),
        "ask_spread_over_arr_spread": {
            "median": round(st.median(ask_over_arr), 4) if ask_over_arr else None,
            "p90": round(sorted(ask_over_arr)[int(0.9 * (len(ask_over_arr) - 1))], 4)
                   if ask_over_arr else None},
        "arr_spread_over_sum_wait": {
            "median": round(st.median(spread_over_wait), 4) if spread_over_wait else None},
        "per_axis": per_axis,
    }
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"doc": __doc__, "result": res, "samples": samples[:4000]},
                                    indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
