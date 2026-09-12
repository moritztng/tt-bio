#!/usr/bin/env python3
"""Turn a tracy ops report into the kernel cycle census of one block.

Reads `ops_perf_results_*.csv` produced by `python -m tracy -r`, finds the fence ops that
`kernel_census.py` dispatched around the profiled region, folds the repetitions together and
writes the per-kernel table plus the three-way split of the block's device time:

  (a) inside a kernel, math       -- delivered FLOP/s against the measured 85.96 TFLOP/s roof
  (b) inside a kernel, waiting    -- kernel time not explained by (a)
  (c) in no kernel at all         -- GAP-FRACTION, the sum of the inter-op gaps over the span
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

FENCE_DIM = 32
FENCE_N = 3

NUM = ("DEVICE FW DURATION [ns]", "DEVICE KERNEL DURATION [ns]",
       "DEVICE KERNEL DURATION PER CORE MIN [ns]", "DEVICE KERNEL DURATION PER CORE MAX [ns]",
       "DEVICE KERNEL DURATION PER CORE AVG [ns]", "DEVICE KERNEL FIRST TO LAST START [ns]",
       "OP TO OP LATENCY [ns]", "DEVICE BRISC KERNEL DURATION [ns]",
       "DEVICE NCRISC KERNEL DURATION [ns]", "DEVICE TRISC0 KERNEL DURATION [ns]",
       "DEVICE TRISC1 KERNEL DURATION [ns]", "DEVICE TRISC2 KERNEL DURATION [ns]",
       "DEVICE FW START CYCLE", "DEVICE FW END CYCLE", "CORE COUNT")


def f(row, key, default=0.0):
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def shape(row, pfx):
    return tuple(int(f(row, f"{pfx}_{d}", 0)) for d in ("W", "Z", "Y", "X"))


def is_fence(row):
    return (shape(row, "INPUT_0")[2:] == (FENCE_DIM, FENCE_DIM)
            and "EXP" in row.get("OP CODE", "").upper())


def find_region(rows):
    """Rows strictly between the first fence run and the second fence run."""
    runs = []
    i = 0
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
    return rows[runs[-2][1]:runs[-1][0]], runs


def op_flops(row):
    """FLOPs of the op, when its shape says so unambiguously. 0 means 'not counted'."""
    code = row.get("OP CODE", "")
    a, b = shape(row, "INPUT_0"), shape(row, "INPUT_1")
    if "Matmul" in code or "matmul" in code:
        if a[3] and b[3]:
            batch = max(a[0] * a[1], b[0] * b[1])
            return 2.0 * batch * a[2] * a[3] * b[3]
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True, help="kernel_census.py --out json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tflops-roof", type=float, default=85.96)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    meta = json.loads(a.meta.read_text())
    reps = int(meta["env"]["reps"])
    region, runs = find_region(rows)
    if len(region) % reps:
        raise SystemExit(f"{len(region)} ops in the region is not divisible by {reps} reps")
    n_ops = len(region) // reps
    chunks = [region[i * n_ops:(i + 1) * n_ops] for i in range(reps)]

    # ns per device cycle, taken from the report's own two columns
    ratios = [f(r, "DEVICE FW DURATION [ns]") /
              (f(r, "DEVICE FW END CYCLE") - f(r, "DEVICE FW START CYCLE"))
              for r in region
              if f(r, "DEVICE FW END CYCLE") > f(r, "DEVICE FW START CYCLE")]
    ns_per_cycle = st.median(ratios) if ratios else 1.0

    per_rep = []
    for ch in chunks:
        span_ns = (f(ch[-1], "DEVICE FW END CYCLE") -
                   f(ch[0], "DEVICE FW START CYCLE")) * ns_per_cycle
        fw = sum(f(r, "DEVICE FW DURATION [ns]") for r in ch)
        kern = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in ch)
        gap = sum(f(r, "OP TO OP LATENCY [ns]") for r in ch[1:])
        per_rep.append({"span_ms": span_ns / 1e6, "fw_ms": fw / 1e6, "kernel_ms": kern / 1e6,
                        "gap_ms": gap / 1e6})

    med = {k: st.median([r[k] for r in per_rep]) for k in per_rep[0]}
    # the block's device time is the span; what is not inside a kernel is (c)
    gap_from_span_ms = med["span_ms"] - med["kernel_ms"]

    ops = []
    for i in range(n_ops):
        rs = [ch[i] for ch in chunks]
        r0 = rs[0]
        kd = [f(r, "DEVICE KERNEL DURATION [ns]") for r in rs]
        flops = op_flops(r0)
        k_med = st.median(kd)
        ops.append({
            "i": i,
            "op": r0.get("OP CODE", "?"),
            "cores": int(f(r0, "CORE COUNT")),
            "in0": shape(r0, "INPUT_0"), "in1": shape(r0, "INPUT_1"),
            "out0": shape(r0, "OUTPUT_0"),
            "fidelity": r0.get("MATH FIDELITY", ""),
            "kernel_ns": k_med,
            "kernel_ns_spread": [min(kd), max(kd)],
            "fw_ns": st.median([f(r, "DEVICE FW DURATION [ns]") for r in rs]),
            "gap_ns": st.median([f(r, "OP TO OP LATENCY [ns]") for r in rs]),
            "core_min_ns": st.median([f(r, "DEVICE KERNEL DURATION PER CORE MIN [ns]") for r in rs]),
            "core_max_ns": st.median([f(r, "DEVICE KERNEL DURATION PER CORE MAX [ns]") for r in rs]),
            "core_avg_ns": st.median([f(r, "DEVICE KERNEL DURATION PER CORE AVG [ns]") for r in rs]),
            "skew_ns": st.median([f(r, "DEVICE KERNEL FIRST TO LAST START [ns]") for r in rs]),
            "brisc_ns": st.median([f(r, "DEVICE BRISC KERNEL DURATION [ns]") for r in rs]),
            "ncrisc_ns": st.median([f(r, "DEVICE NCRISC KERNEL DURATION [ns]") for r in rs]),
            "trisc0_ns": st.median([f(r, "DEVICE TRISC0 KERNEL DURATION [ns]") for r in rs]),
            "trisc1_ns": st.median([f(r, "DEVICE TRISC1 KERNEL DURATION [ns]") for r in rs]),
            "trisc2_ns": st.median([f(r, "DEVICE TRISC2 KERNEL DURATION [ns]") for r in rs]),
            "flops": flops,
            "math_ns": (1e9 * flops / (a.tflops_roof * 1e12)) if flops else 0.0,
        })

    math_ms = sum(o["math_ns"] for o in ops) / 1e6
    kernel_ms = sum(o["kernel_ns"] for o in ops) / 1e6
    span_ms = med["span_ms"]
    out = {
        "label": a.label,
        "csv": str(a.csv),
        "ops_per_block": n_ops,
        "reps": reps,
        "ns_per_cycle": ns_per_cycle,
        "per_rep_ms": per_rep,
        "median_ms": med,
        "synced_wall_ms_per_call": meta.get("synced_wall_ms_per_call"),
        "split_ms": {
            "span": span_ms,
            "a_math_at_roof": math_ms,
            "b_in_kernel_waiting": kernel_ms - math_ms,
            "c_outside_kernels": span_ms - kernel_ms,
        },
        "split_pct": {
            "a_math_at_roof": 100.0 * math_ms / span_ms,
            "b_in_kernel_waiting": 100.0 * (kernel_ms - math_ms) / span_ms,
            "c_outside_kernels": 100.0 * (span_ms - kernel_ms) / span_ms,
        },
        "gap_from_op_to_op_latency_ms": med["gap_ms"],
        "gap_from_span_minus_kernel_ms": gap_from_span_ms,
        "ops": ops,
    }
    a.out.write_text(json.dumps(out, indent=1))

    print(f"{a.label}: {n_ops} ops/block, span {span_ms:.4f} ms, "
          f"kernels {kernel_ms:.4f} ms, gaps {span_ms - kernel_ms:.4f} ms")
    print(f"GAP-FRACTION: {out['split_pct']['c_outside_kernels']:.1f} %")
    print(f"  (a) math at roof      {math_ms:9.4f} ms  {out['split_pct']['a_math_at_roof']:5.1f} %")
    print(f"  (b) in-kernel waiting {kernel_ms - math_ms:9.4f} ms  "
          f"{out['split_pct']['b_in_kernel_waiting']:5.1f} %")
    print(f"  (c) outside kernels   {span_ms - kernel_ms:9.4f} ms  "
          f"{out['split_pct']['c_outside_kernels']:5.1f} %")
    top = sorted(ops, key=lambda o: -o["kernel_ns"])[:15]
    print(f"  top ops by kernel ns:")
    for o in top:
        print(f"    {o['op'][:34]:34s} cores {o['cores']:3d}  "
              f"{o['kernel_ns'] / 1e3:9.1f} us  gap {o['gap_ns'] / 1e3:7.1f} us  "
              f"in0 {o['in0']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
