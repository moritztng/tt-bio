#!/usr/bin/env python3
"""Turn tracy ops reports from prof_block512.py into per-call and ms/fold device numbers.

    python3 perf/rowblock_prof/analyze_ops.py \
        --run on1:/tmp/p_on1/reports/*/ops_perf_results_*.csv:/tmp/p_on1.json \
        --run rb1:... --run on2:... --run rb2:... \
        --out perf/rowblock_prof/profiler_ab_qb1c3.json

Attribution is by the marker bracket the harness inserted (a `sqrt` on [1, 1, 32, 64]): marker
pairs come in a fixed order -- (warm + iters) * 2 pair-transpose calls, then `--roof-reps` DRAM
clones, then `--roof-reps` L1 clones -- so everything inside a bracket belongs to that call and
nothing it dispatches can be left out of the accounting.

CENSUS x1048 is the whole-fold conversion for this site: 2 call sites x 524 c_z=256
PairformerLayer executions (charter §4.9; trunk-only is x1040). Divide by blocks x recycles, never
by blocks alone (`tt-bio-trunk-perf-ratio-denominator-unit-slip`).
"""

import argparse
import collections
import csv
import glob
import json
import statistics as st

CENSUS = 1048
MARK = ("1[1]", "1[1]", "32[32]", "64[64]")


def shape_of(r, i=0):
    return tuple(r[f"INPUT_{i}_{d}_PAD[LOGICAL]"] for d in "WZYX")


def num(r, key):
    try:
        return float(r[key])
    except (TypeError, ValueError):
        return 0.0


def brackets(rows):
    marks = [i for i, r in enumerate(rows) if shape_of(r) == MARK]
    if len(marks) % 2:
        marks = marks[:-1]
    return [(marks[k], marks[k + 1]) for k in range(0, len(marks), 2)]


def summarize(rows, lo, hi):
    inner = rows[lo + 1:hi]
    by = collections.Counter()
    kern = collections.Counter()
    cores = collections.defaultdict(set)
    risc = collections.defaultdict(collections.Counter)
    for r in inner:
        key = f"{r['OP CODE']} in0={list(shape_of(r))} out={'L1' if 'L1' in r['ATTRIBUTES'] else 'DRAM'}"
        by[key] += 1
        kern[key] += num(r, "DEVICE KERNEL DURATION [ns]")
        cores[key].add(r["CORE COUNT"])
        for rc in ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"):
            risc[key][rc] += num(r, f"DEVICE {rc} KERNEL DURATION [ns]")
    return {
        "programs": len(inner),
        "kernel_ns": sum(num(r, "DEVICE KERNEL DURATION [ns]") for r in inner),
        "fw_ns": sum(num(r, "DEVICE FW DURATION [ns]") for r in inner),
        "o2o_ns": sum(num(r, "OP TO OP LATENCY [ns]") for r in inner[1:]),
        "ops": {k: {"n": by[k], "kernel_ns": kern[k], "cores": sorted(cores[k]),
                    "risc_ns": dict(risc[k])} for k in by},
    }


def one_run(label, csv_glob, meta_path):
    path = sorted(glob.glob(csv_glob))[-1]
    rows = [r for r in csv.DictReader(open(path)) if r.get("OP CODE")]
    meta = json.load(open(meta_path))
    br = brackets(rows)
    n_t = meta["n_warm_transpose_calls"] + meta["n_measured_transpose_calls"]
    rr = meta["roof_reps"]
    order = meta.get("roof_order", ["copy roof (DRAM), DRAM source", "copy roof (L1), DRAM source"])
    want = n_t + len(order) * rr
    assert len(br) >= want, f"{label}: {len(br)} marker brackets, expected {want}"
    tr = [summarize(rows, *b) for b in br[:n_t]][meta["n_warm_transpose_calls"]:]
    pair_gb = 2 * meta["pair_bytes"] / 1e9                      # one read + one write
    roofs = {}
    for i, name in enumerate(order):
        grp = [summarize(rows, *b) for b in br[n_t + i * rr:n_t + (i + 1) * rr]]
        ms = st.median(x["kernel_ns"] for x in grp) / 1e6
        roofs[name] = {"ms": ms, "gbs": pair_gb / (ms / 1e3)}
    med = st.median(t["kernel_ns"] for t in tr)
    agg = collections.Counter()
    aggn = collections.Counter()
    corestr = {}
    aggrisc = collections.defaultdict(collections.Counter)
    for t in tr:
        for k, v in t["ops"].items():
            agg[k] += v["kernel_ns"] / len(tr)
            aggn[k] += v["n"] / len(tr)
            corestr[k] = v["cores"]
            for rc, ns_ in v["risc_ns"].items():
                aggrisc[k][rc] += ns_ / len(tr)
    # per-block device kernel sum: everything dispatched before the roof section
    pre_roof = rows[:br[n_t][0]] if len(br) > n_t else rows
    blocks = meta["warm"] + meta["iters"]
    return {
        "label": label, "arm": meta["arm"], "csv": path,
        "census_unique": meta["census_unique"],
        "blocked_calls": meta["blocked_calls"], "total_calls": meta["total_calls"],
        "block_wall_ms_median": meta["block_wall_ms"]["median"],
        "calls_measured": len(tr),
        "programs_per_call": st.median(t["programs"] for t in tr),
        "kernel_ms_per_call": {
            "median": med / 1e6, "min": min(t["kernel_ns"] for t in tr) / 1e6,
            "max": max(t["kernel_ns"] for t in tr) / 1e6,
            "spread_pct_of_median": 100.0 * (max(t["kernel_ns"] for t in tr)
                                             - min(t["kernel_ns"] for t in tr)) / med,
        },
        "o2o_ms_per_call_median": st.median(t["o2o_ns"] for t in tr) / 1e6,
        "ms_per_fold_x1048": med / 1e6 * CENSUS,
        "per_op_mean_us_per_call": {k: {"n": round(aggn[k], 2), "us": round(agg[k] / 1e3, 1),
                                       "cores": corestr[k],
                                       "risc_us": {rc: round(v / 1e3, 1)
                                                   for rc, v in aggrisc[k].items()}} for k in agg},
        "block_device_kernel_ms": sum(num(r, "DEVICE KERNEL DURATION [ns]")
                                      for r in pre_roof) / 1e6 / blocks,
        "roofs_same_clock": roofs,
        "useful_gbs_per_call": pair_gb / (med / 1e9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="LABEL:CSVGLOB:METAJSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = []
    for spec in args.run:
        label, csv_glob, meta = spec.split(":", 2)
        runs.append(one_run(label, csv_glob, meta))

    print(f"{'run':10} {'arm':7} {'progs':>6} {'ms/call':>9} {'spread%':>8} {'ms/fold':>9} "
          f"{'o2o ms':>7} {'blk dev ms':>10} {'blk wall ms':>11}")
    for r in runs:
        print(f"{r['label']:10} {r['arm']:7} {r['programs_per_call']:6.0f} "
              f"{r['kernel_ms_per_call']['median']:9.4f} "
              f"{r['kernel_ms_per_call']['spread_pct_of_median']:8.2f} "
              f"{r['ms_per_fold_x1048']:9.1f} {r['o2o_ms_per_call_median']:7.4f} "
              f"{r['block_device_kernel_ms']:10.2f} {str(r['block_wall_ms_median']):>11}")

    by_arm = collections.defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r["ms_per_fold_x1048"])
    floors = {}
    for arm, v in by_arm.items():
        if len(v) > 1:
            floors[arm] = {"n": len(v), "spread_ms_fold": max(v) - min(v),
                           "median_ms_fold": st.median(v)}
            print(f"A/A floor, arm {arm}: {max(v) - min(v):.1f} ms/fold over {len(v)} runs "
                  f"(median {st.median(v):.1f})")
    if "on" in by_arm and "rb_fit" in by_arm:
        d = st.median(by_arm["on"]) - st.median(by_arm["rb_fit"])
        print(f"A/B: rb_fit is {d:+.1f} ms/fold of summed device kernel time vs on "
              f"(positive = rb_fit faster); floors above")
    for r in runs:
        print(f"\n--- {r['label']} ({r['arm']}) per-op mean per call, cores of 130")
        for k, v in sorted(r["per_op_mean_us_per_call"].items(), key=lambda x: -x[1]["us"]):
            rr = v["risc_us"]
            print(f"    {v['us']:8.1f} us  n={v['n']:5}  cores={v['cores']}  "
                  f"BRISC {rr.get('BRISC', 0):.0f} NCRISC {rr.get('NCRISC', 0):.0f} "
                  f"TRISC1 {rr.get('TRISC1', 0):.0f} TRISC2 {rr.get('TRISC2', 0):.0f}  {k}")
        print(f"    useful rate {r['useful_gbs_per_call']:.1f} GB/s (268.44 MB r+w per call). "
              "roofs, same clock, same shape:")
        for name, v in r["roofs_same_clock"].items():
            print(f"      {v['gbs']:7.1f} GB/s ({v['ms']:.4f} ms)  {name}   "
                  f"-> this arm is {100.0 * r['useful_gbs_per_call'] / v['gbs']:.1f} % of it")
        print(f"    census: {r['census_unique']}  blocked {r['blocked_calls']}/{r['total_calls']}")

    if args.out:
        json.dump({"runs": runs, "aa_floors": floors, "census_x": CENSUS},
                  open(args.out, "w"), indent=1)
        print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
