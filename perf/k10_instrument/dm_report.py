#!/usr/bin/env python3
"""Split each profiled op's data-movement threads into blocked-on-NoC vs blocked-on-CB.

The sum accumulators are summed over every core in the op's grid, while the KERNEL DURATION
columns are the op's span. So every sum is divided by CORE COUNT before it is compared to a
duration -- the census recorded that this normalisation is the one that goes wrong (it was
violated on 50 of its 232 ops), so rows whose per-core fraction exceeds 1.05 are flagged rather
than averaged in.

Reading:
  reader in NOC-READ-BARRIER while the math cluster is in CB-WAIT-FRONT -> producer-late (a)
  reader in DM-CB-RESERVE-BACK (its output CB full) or finished early    -> structural (b)
"""
from __future__ import annotations

import argparse, csv, glob, json
from pathlib import Path

DUR = "DEVICE KERNEL DURATION [ns]"
COLS = {
    "brisc": "DEVICE BRISC KERNEL DURATION [ns]",
    "ncrisc": "DEVICE NCRISC KERNEL DURATION [ns]",
    "trisc0": "DEVICE TRISC0 KERNEL DURATION [ns]",
    "trisc1": "DEVICE TRISC1 KERNEL DURATION [ns]",
    "trisc2": "DEVICE TRISC2 KERNEL DURATION [ns]",
    "cpu_wait_front": "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "cpu_reserve_back": "DEVICE COMPUTE CB RESERVE BACK [ns]",
    "br_reserve_back": "DEVICE DM CB RESERVE BACK BRISC [ns]",
    "nc_reserve_back": "DEVICE DM CB RESERVE BACK NCRISC [ns]",
    "br_wait_front": "DEVICE DM CB WAIT FRONT BRISC [ns]",
    "nc_wait_front": "DEVICE DM CB WAIT FRONT NCRISC [ns]",
    "br_noc_read": "DEVICE DM NOC READ BARRIER BRISC [ns]",
    "nc_noc_read": "DEVICE DM NOC READ BARRIER NCRISC [ns]",
    "br_noc_write": "DEVICE DM NOC WRITE WAIT BRISC [ns]",
    "nc_noc_write": "DEVICE DM NOC WRITE WAIT NCRISC [ns]",
    "br_sem": "DEVICE DM SEM WAIT BRISC [ns]",
    "nc_sem": "DEVICE DM SEM WAIT NCRISC [ns]",
}
SUMS = [k for k in COLS if k.startswith(("cpu_", "br_", "nc_"))]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load(pattern):
    f = sorted(glob.glob(pattern))[-1]
    return f, list(csv.DictReader(open(f)))


FENCE_DIM, FENCE_N = 32, 3


def is_fence(r):
    """kernel_census.py brackets its profiled region with 3 x ttnn.exp on a 32x32 tile."""
    if not r.get("OP CODE", "").startswith("Unary"):
        return False
    return num(r.get("CORE COUNT")) == 1


def fenced_region(rows):
    """Rows strictly between the last two fence runs -- the profiled repetitions only."""
    runs, i = [], 0
    while i < len(rows):
        if is_fence(rows[i]):
            j = i
            while j < len(rows) and is_fence(rows[j]):
                j += 1
            if j - i >= FENCE_N:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < 2:
        raise SystemExit(f"expected >=2 fence runs, found {len(runs)}")
    return rows[runs[-2][1]:runs[-1][0]]


def per_op(r):
    """One row -> per-core nanoseconds for every duration and every accumulator."""
    cores = num(r.get("CORE COUNT")) or 1
    o = {"op": r["OP CODE"], "cores": int(cores), "span_ns": num(r[DUR])}
    for k, c in COLS.items():
        v = num(r.get(c, 0))
        o[k] = v / cores if k in SUMS else v
    return o


def frac(o):
    """Fractions of the op span. Reader is whichever DM thread does the NoC reads."""
    s = o["span_ns"] or 1
    keys = [k for k in COLS]
    d = {k: o[k] / s for k in keys}
    d["dm_noc_read"] = d["br_noc_read"] + d["nc_noc_read"]
    d["dm_reserve_back"] = d["br_reserve_back"] + d["nc_reserve_back"]
    d["dm_wait_front"] = d["br_wait_front"] + d["nc_wait_front"]
    for r in ("br", "nc"):
        blocked = sum(d[f"{r}_{k}"] for k in
                      ("noc_read", "noc_write", "sem", "reserve_back", "wait_front"))
        d[f"{r}_blocked"] = blocked
        d[f"{r}_unaccounted"] = d["brisc" if r == "br" else "ncrisc"] - blocked
    return d


def aggregate(rows, op_filter=None, min_ns=0):
    """Duration-weighted mean of the per-core fractions over the matching rows."""
    sel = [per_op(r) for r in rows
           if (op_filter is None or op_filter in r["OP CODE"]) and num(r[DUR]) >= min_ns]
    if not sel:
        return None
    tot = sum(o["span_ns"] for o in sel)
    out = {"n_ops": len(sel), "total_span_ms": tot / 1e6,
           "mean_span_us": tot / len(sel) / 1e3,
           "cores": sorted({o["cores"] for o in sel})}
    acc = {}
    over = 0
    for o in sel:
        f = frac(o)
        if max(f["cpu_wait_front"], f["dm_noc_read"]) > 1.05:
            over += 1
        for k, v in f.items():
            acc[k] = acc.get(k, 0.0) + v * o["span_ns"]
    out["frac"] = {k: round(v / tot, 4) for k, v in acc.items()}
    out["rows_over_unity"] = over
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="glob for ops_perf_results_*.csv")
    ap.add_argument("--op", action="append", default=[], help="OP CODE substring; repeatable")
    ap.add_argument("--min-ns", type=float, default=0)
    ap.add_argument("--fenced", action="store_true",
                    help="keep only the rows between the harness's two fence runs")
    ap.add_argument("--reps", type=int, default=1, help="repetitions inside the fenced region")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    f, rows = load(a.csv)
    n_all = len(rows)
    if a.fenced:
        rows = fenced_region(rows)
        if len(rows) % a.reps:
            raise SystemExit(f"{len(rows)} fenced ops is not divisible by {a.reps} reps")
    res = {"csv": f, "n_rows_file": n_all, "n_rows": len(rows), "fenced": a.fenced,
           "reps": a.reps, "ops_per_rep": len(rows) // a.reps, "op_codes": {}}
    codes = sorted({r["OP CODE"] for r in rows})
    res["all_op_codes"] = codes
    for op in (a.op or codes):
        g = aggregate(rows, op, a.min_ns)
        if g:
            res["op_codes"][op] = g
    res["ALL"] = aggregate(rows, None, a.min_ns)

    for op, g in list(res["op_codes"].items()) + [("ALL", res["ALL"])]:
        if not g:
            continue
        fr = g["frac"]
        print(f"\n{op}  n={g['n_ops']}  span={g['total_span_ms']:.4f} ms "
              f"({g['total_span_ms'] / a.reps:.4f} ms/rep)  "
              f"mean={g['mean_span_us']:.1f} us  cores={g['cores']}")
        print(f"  resident   BRISC {fr['brisc']:6.1%}  NCRISC {fr['ncrisc']:6.1%}  "
              f"TRISC0 {fr['trisc0']:6.1%}  TRISC1 {fr['trisc1']:6.1%}  TRISC2 {fr['trisc2']:6.1%}")
        print(f"  compute    wait-front {fr['cpu_wait_front']:6.1%}   reserve-back {fr['cpu_reserve_back']:6.1%}")
        for r, name in (("br", "BRISC "), ("nc", "NCRISC")):
            print(f"  {name}     noc-read {fr[f'{r}_noc_read']:6.1%}  noc-write {fr[f'{r}_noc_write']:6.1%}  "
                  f"sem {fr[f'{r}_sem']:6.1%}  cb-full {fr[f'{r}_reserve_back']:6.1%}  "
                  f"cb-wait {fr[f'{r}_wait_front']:6.1%}  | blocked {fr[f'{r}_blocked']:6.1%}  "
                  f"issuing {fr[f'{r}_unaccounted']:6.1%}")
        if g["rows_over_unity"]:
            print(f"  !! {g['rows_over_unity']} of {g['n_ops']} rows exceed 1.05 per-core; "
                  f"partial grid, treat as directional")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
