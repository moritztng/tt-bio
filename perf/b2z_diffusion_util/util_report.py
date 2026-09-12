#!/usr/bin/env python3
"""Core-utilization census of one Boltz-2 512 aa diffusion step.

Reads either a tracy `ops_perf_results_*.csv` or the per-program records that
`ttnn.profiler.get_all_programs_perf_data()` returns, and answers two questions for the
diffusion step the same way `b2z-grid-utilization` answered them for the pairformer block:

  * duration-weighted mean core utilization, sum(t_i * c_i / C) / sum(t_i)
  * CORE-SECONDS-LOST, sum over the fold of t_i * (C - c_i) / C -- the core-fraction-seconds
    the grid sits idle while a program runs on a subgrid

`C` is the device's available worker core count, read from the report, not assumed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

FENCE_DIM = 32
FENCE_N = 3


def f(row, key, default=0.0):
    """Numeric cell. The shape columns are `padded[logical]`, e.g. `512[500]`; take padded.

    Which of the two forms a report uses depends on whether it was generated with
    `--no-op-info-cache`, so both have to parse or the shapes silently read as zero.
    """
    v = row.get(key, "")
    if v in ("", None):
        return default
    v = str(v).split("[", 1)[0].strip()
    try:
        return float(v)
    except ValueError:
        return default


SHAPE_KEYS = ("{p}_{d}_PAD[LOGICAL]", "{p}_{d}")


def shape(row, pfx):
    """Padded WZYX of an operand.

    The column is `INPUT_0_W_PAD[LOGICAL]`, not `INPUT_0_W`. Reading the short name returns
    the default for every row, which is how a shape table comes out all zeros and looks like
    a program-cache artifact instead of a wrong key.
    """
    out = []
    for d in ("W", "Z", "Y", "X"):
        v = 0.0
        for k in SHAPE_KEYS:
            key = k.format(p=pfx, d=d)
            if key in row:
                v = f(row, key, 0.0)
                break
        out.append(int(v))
    return tuple(out)


def is_fence(row):
    if not row.get("OP CODE", "").startswith("Unary"):
        return False
    return shape(row, "INPUT_0")[2:] in ((FENCE_DIM, FENCE_DIM), (0, 0))


def fence_runs(rows):
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
    return runs


def find_region(rows, reps, period=None):
    """The `reps` repetitions of the profiled call.

    Two anchorings, because the device's marker buffer can wrap on a long precursor and drop
    the opening fence (seen on qb2: 9936 rows survived where 10 calls alone need 10 660):

      * both fences present -> everything strictly between the last two fence runs;
      * only the closing fence -> walk back `reps * period` rows from it.

    Either way the split into repetitions is *checked*, not assumed: the op-code sequence of
    every repetition must be identical. A wrapped buffer that cut a call in half fails here.
    """
    runs = fence_runs(rows)
    if not runs:
        raise SystemExit("no fence run found; the profiled region cannot be located")
    end = runs[-1][0]
    if len(runs) >= 2:
        region = rows[runs[-2][1]:end]
        if len(region) % reps:
            raise SystemExit(f"{len(region)} ops between the fences is not divisible by {reps}")
    else:
        if not period:
            raise SystemExit("only one fence run survived; pass --period to anchor on it")
        region = rows[end - reps * period:end]
        if len(region) != reps * period:
            raise SystemExit(f"only {end} rows before the fence, need {reps * period}")
    n = len(region) // reps
    seqs = {tuple(r.get("OP CODE", "?") for r in region[i * n:(i + 1) * n]) for i in range(reps)}
    if len(seqs) != 1:
        raise SystemExit(f"the {reps} repetitions are not the same program sequence "
                         f"({len(seqs)} distinct); the region is mis-anchored")
    return region


def read_csv(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--calls-per-fold", type=int, default=200)
    ap.add_argument("--label", default="DiffusionStep")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--period", type=int, default=0,
                    help="programs per call, needed when only the closing fence survived")
    ap.add_argument("--half", type=float, default=0.5,
                    help="an op is 'under-grid' below this fraction of the grid")
    a = ap.parse_args()

    region = find_region(read_csv(a.csv), a.reps, a.period)
    n = len(region) // a.reps
    chunks = [region[i * n:(i + 1) * n] for i in range(a.reps)]

    avail = {int(f(r, "AVAILABLE WORKER CORE COUNT")) for r in region}
    avail = {c for c in avail if c}
    if len(avail) != 1:
        raise SystemExit(f"non-uniform available core count: {sorted(avail)}")
    C = avail.pop()

    ops = []
    for i in range(n):
        rs = [ch[i] for ch in chunks]
        r0 = rs[0]
        cores = {int(f(r, "CORE COUNT")) for r in rs}
        ops.append({
            "i": i,
            "op": r0.get("OP CODE", "?"),
            "cores": int(f(r0, "CORE COUNT")),
            "cores_varies": sorted(cores) if len(cores) > 1 else None,
            "kernel_ns": st.median([f(r, "DEVICE KERNEL DURATION [ns]") for r in rs]),
            "in0": shape(r0, "INPUT_0"), "in1": shape(r0, "INPUT_1"),
            "out0": shape(r0, "OUTPUT_0"),
            "strategy": r0.get("PARALLELIZATION STRATEGY", ""),
            "fidelity": r0.get("MATH FIDELITY", ""),
        })
    return emit(ops, C, a)


def emit(ops, C, a):
    tot = sum(o["kernel_ns"] for o in ops)
    for o in ops:
        o["util"] = o["cores"] / C
        o["idle_core_ns"] = o["kernel_ns"] * (C - o["cores"]) / C
    wmean = sum(o["kernel_ns"] * o["util"] for o in ops) / tot
    under = [o for o in ops if o["util"] < a.half]
    s = a.calls_per_fold / 1e9
    by_code = defaultdict(lambda: {"n": 0, "kernel_ns": 0.0, "idle_core_ns": 0.0})
    for o in ops:
        b = by_code[o["op"]]
        b["n"] += 1
        b["kernel_ns"] += o["kernel_ns"]
        b["idle_core_ns"] += o["idle_core_ns"]
    for k, b in by_code.items():
        b["util"] = 1.0 - b["idle_core_ns"] / b["kernel_ns"] if b["kernel_ns"] else 1.0
        b["kernel_ms"] = b["kernel_ns"] / 1e6

    bands = {"100%": 0.0, "90-100%": 0.0, "75-90%": 0.0, "50-75%": 0.0, "<50%": 0.0}
    for o in ops:
        u = o["util"]
        k = ("100%" if u >= 0.999 else "90-100%" if u >= 0.9 else
             "75-90%" if u >= 0.75 else "50-75%" if u >= 0.5 else "<50%")
        bands[k] += o["kernel_ns"]
    out = {
        "label": a.label,
        "source": str(a.csv),
        "available_worker_cores": C,
        "programs_per_call": len(ops),
        "calls_per_fold": a.calls_per_fold,
        "kernel_ms_per_call": tot / 1e6,
        "kernel_s_per_fold": tot * s,
        "duration_weighted_mean_core_utilization": wmean,
        "idle_core_fraction_seconds_per_fold_all_ops":
            sum(o["idle_core_ns"] for o in ops) * s,
        "under_grid_threshold": a.half,
        "under_grid_ops": len(under),
        "under_grid_kernel_ms_per_call": sum(o["kernel_ns"] for o in under) / 1e6,
        "under_grid_share_of_call_time": (sum(o["kernel_ns"] for o in under) / tot),
        "core_seconds_lost_per_fold_under_grid":
            sum(o["idle_core_ns"] for o in under) * s,
        "time_share_by_util_band": {k: v / tot for k, v in bands.items()},
        "by_op_code": dict(sorted(by_code.items(), key=lambda kv: -kv[1]["idle_core_ns"])),
        "top_idle": sorted(ops, key=lambda o: -o["idle_core_ns"])[:25],
        "ops": ops,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    p = {k: v for k, v in out.items() if k not in ("by_op_code", "top_idle", "ops")}
    print(json.dumps(p, indent=1))
    print("\nby op code (idle core-seconds/fold, util):")
    for k, b in list(out["by_op_code"].items())[:12]:
        print(f"  {k:34s} n={b['n']:5d} kernel={b['kernel_ms']:8.4f} ms  "
              f"util={100*b['util']:6.2f}%  idle={b['idle_core_ns']*s:.4f} s/fold")
    print("\ntop idle programs:")
    for o in out["top_idle"][:12]:
        print(f"  #{o['i']:4d} {o['op']:30s} {o['cores']:3d}/{C} "
              f"{o['kernel_ns']/1000:9.2f} us  idle={o['idle_core_ns']*s:.4f} s/fold "
              f"in0={o['in0']} in1={o['in1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
