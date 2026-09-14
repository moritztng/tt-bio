#!/usr/bin/env python3
"""Measure the Boltz-2 diffusion step: gap fraction, cost by op code, stall split.

Host only, read-only, opens no device. Point it at a `--enable-sum-profiling` ops CSV taken by
`kernel_census.py --phase step` plus that run's meta json, and at the bare (profiler-off) walls
for the same grabbed call.

Two things this does that the earlier readers did not.

**Gap fraction is never read off the armed span.** `--enable-sum-profiling` inflates the gaps
between programs by ~3x on a 1066-program call and leaves the kernels alone, so the armed span
is not a wall. The device kernel durations from the armed capture are, and the wall comes from a
separate profiler-off run. gap = 1 - kernel_sum / bare_wall, once against the eager wall and
once against the traced wall, which is the only way to separate host dispatch from the device's
own program-to-program latency.

**Each stall counter is divided by its own thread.** `DEVICE COMPUTE CB WAIT FRONT` is declared
on the unpack thread (TRISC0) and `DEVICE COMPUTE CB RESERVE BACK` on the pack thread (TRISC2);
dividing either by TRISC1 is the divisor defect `k10-thread-attrib` root-caused. Both are summed
over the cores that ran the op, so each is divided by that op's own CORE COUNT first.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import json
from collections import defaultdict
from pathlib import Path

FENCE_DIM, FENCE_N = 32, 3
WAIT = "DEVICE COMPUTE CB WAIT FRONT [ns]"
RES = "DEVICE COMPUTE CB RESERVE BACK [ns]"
T0, T1, T2 = (f"DEVICE TRISC{i} KERNEL DURATION [ns]" for i in (0, 1, 2))
KERNEL = "DEVICE KERNEL DURATION [ns]"
BR, NC = "DEVICE BRISC KERNEL DURATION [ns]", "DEVICE NCRISC KERNEL DURATION [ns]"
GAP = "OP TO OP LATENCY [ns]"
CORES = "CORE COUNT"


def f(row, key, default=0.0):
    try:
        return float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def dim(row, key):
    """Shape columns read `32[32]` -- padded[logical]. Take the padded extent."""
    m = re.match(r"\s*(\d+)", str(row.get(key, "")))
    return int(m.group(1)) if m else 0


def is_fence(row):
    """The census fence is FENCE_N consecutive ttnn.exp on a FENCE_DIM x FENCE_DIM tensor."""
    if not row.get("OP CODE", "").startswith("Unary"):
        return False
    return (dim(row, "INPUT_0_Y_PAD[LOGICAL]"), dim(row, "INPUT_0_X_PAD[LOGICAL]")) == \
        (FENCE_DIM, FENCE_DIM)


def find_region(rows):
    """The profiled region is what sits between the last two runs of >=FENCE_N fence ops."""
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


def open_csv(p: Path):
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True, help="armed run's json (for --reps)")
    ap.add_argument("--bare-eager-ms", type=float, required=True,
                    help="profiler-off synced wall per call, eager dispatch")
    ap.add_argument("--bare-traced-ms", type=float, default=None,
                    help="profiler-off wall per call, ttnn trace capture+replay")
    ap.add_argument("--bare-replay-ms", type=float, default=None,
                    help="profiler-off wall of the same call in a tight replay loop, where the "
                         "host is fully ahead. Isolates the device's own program-to-program "
                         "latency from the per-step host cost the fold pays on top of it.")
    ap.add_argument("--label", required=True)
    ap.add_argument("--arch", required=True, choices=("WH", "BH"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open_csv(a.csv)))
    reps = int(json.loads(a.meta.read_text())["env"]["reps"])
    region = find_region(rows)
    if len(region) % reps:
        raise SystemExit(f"{len(region)} ops in the region is not divisible by {reps} reps")
    n_ops = len(region) // reps

    out = {"label": a.label, "arch": a.arch, "csv": str(a.csv), "reps": reps,
           "programs_per_call": n_ops, "region_rows": len(region)}

    # ---- 1. gap fraction -------------------------------------------------------------------
    kern = sum(f(r, KERNEL) for r in region) / reps / 1e6
    armed_gap = sum(f(r, GAP) for r in region) / reps / 1e6
    g = {"device_kernel_ms_per_call": round(kern, 4),
         "armed_op_to_op_ms_per_call": round(armed_gap, 4),
         "bare_eager_ms_per_call": a.bare_eager_ms,
         "eager_gap_ms_per_call": round(a.bare_eager_ms - kern, 4),
         "eager_gap_frac": round(1 - kern / a.bare_eager_ms, 5),
         "eager_gap_us_per_program": round(1e3 * (a.bare_eager_ms - kern) / n_ops, 3)}
    if a.bare_replay_ms:
        g.update({"bare_replay_ms_per_call": a.bare_replay_ms,
                  "device_only_gap_ms": round(a.bare_replay_ms - kern, 4),
                  "device_only_gap_frac": round(1 - kern / a.bare_replay_ms, 5),
                  "device_only_gap_us_per_program": round(
                      1e3 * (a.bare_replay_ms - kern) / n_ops, 3),
                  "host_per_step_ms": round(a.bare_eager_ms - a.bare_replay_ms, 4)})
    if a.bare_traced_ms:
        g.update({"bare_traced_ms_per_call": a.bare_traced_ms,
                  "traced_gap_ms_per_call": round(a.bare_traced_ms - kern, 4),
                  "traced_gap_frac": round(1 - kern / a.bare_traced_ms, 5),
                  "traced_gap_us_per_program": round(1e3 * (a.bare_traced_ms - kern) / n_ops, 3),
                  "trace_speedup": round(a.bare_eager_ms / a.bare_traced_ms, 4)})
    out["gap"] = g

    # ---- 2. cost by op code, and 3. the stall split ----------------------------------------
    per = defaultdict(lambda: defaultdict(float))
    tot = dict.fromkeys(("wait", "res", "t0", "t1", "t2", "kernel", "br", "nc"), 0.0)
    bad0 = bad2 = 0
    for r in region:
        n = max(f(r, CORES), 1.0)
        wi, wo = f(r, WAIT) / n, f(r, RES) / n
        t0, t1, t2, k = f(r, T0), f(r, T1), f(r, T2), f(r, KERNEL)
        c = per[r.get("OP CODE", "?").replace("DeviceOperation", "").replace("Operation", "")]
        c["n"] += 1; c["kernel"] += k; c["t0"] += t0; c["t1"] += t1; c["t2"] += t2
        c["wait"] += wi; c["res"] += wo; c["cores"] += f(r, CORES); c["gap"] += f(r, GAP)
        tot["wait"] += wi; tot["res"] += wo; tot["kernel"] += k
        tot["t0"] += t0; tot["t1"] += t1; tot["t2"] += t2
        tot["br"] += f(r, BR); tot["nc"] += f(r, NC)
        if t0 > 0:
            bad0 += wi > t0
        if t2 > 0:
            bad2 += wo > t2

    n_stall = sum(1 for r in region if f(r, T1) > 0 and f(r, WAIT) + f(r, RES) > 0)
    out["stall_split"] = {
        "compute_rows_with_counters": n_stall,
        "brisc_ms_per_call": round(tot["br"] / reps / 1e6, 4),
        "ncrisc_ms_per_call": round(tot["nc"] / reps / 1e6, 4),
        "brisc_residency": round(tot["br"] / tot["kernel"], 4),
        "ncrisc_residency": round(tot["nc"] / tot["kernel"], 4),
        "trisc0_residency": round(tot["t0"] / tot["kernel"], 4),
        "trisc1_residency": round(tot["t1"] / tot["kernel"], 4),
        "trisc2_residency": round(tot["t2"] / tot["kernel"], 4),
        "trisc0_ms_per_call": round(tot["t0"] / reps / 1e6, 4),
        "trisc1_ms_per_call": round(tot["t1"] / reps / 1e6, 4),
        "trisc2_ms_per_call": round(tot["t2"] / reps / 1e6, 4),
        "wait_front_ms_per_call": round(tot["wait"] / reps / 1e6, 4),
        "reserve_back_ms_per_call": round(tot["res"] / reps / 1e6, 4),
        "wait_front_over_trisc0": round(tot["wait"] / tot["t0"], 4) if tot["t0"] else None,
        "reserve_back_over_trisc2": round(tot["res"] / tot["t2"], 4) if tot["t2"] else None,
        "input_output_ratio": round(tot["wait"] / tot["res"], 3) if tot["res"] else None,
        "wait_over_kernel": round(tot["wait"] / tot["kernel"], 4),
        "residual_wait_gt_trisc0": bad0,
        "residual_reserve_gt_trisc2": bad2,
    }

    table = []
    for code, c in sorted(per.items(), key=lambda x: -x[1]["kernel"]):
        table.append({
            "op": code, "n": int(c["n"] / reps),
            "kernel_ms": round(c["kernel"] / reps / 1e6, 4),
            "pct_kernel": round(100 * c["kernel"] / tot["kernel"], 2),
            "mean_us": round(c["kernel"] / c["n"] / 1e3, 2),
            "mean_cores": round(c["cores"] / c["n"], 1),
            "t0_ms": round(c["t0"] / reps / 1e6, 4),
            "wait_ms": round(c["wait"] / reps / 1e6, 4),
            "in_over_t0": round(c["wait"] / c["t0"], 4) if c["t0"] else None,
            "out_over_t2": round(c["res"] / c["t2"], 4) if c["t2"] else None,
            "in_out": round(c["wait"] / c["res"], 2) if c["res"] else None,
            "armed_gap_ms": round(c["gap"] / reps / 1e6, 4),
        })
    out["by_op"] = table
    out["movement_only"] = {
        "programs": sum(t["n"] for t in table if t["t0_ms"] == 0),
        "kernel_ms": round(sum(t["kernel_ms"] for t in table if t["t0_ms"] == 0), 4),
        "pct_kernel": round(sum(t["pct_kernel"] for t in table if t["t0_ms"] == 0), 2),
        "op_codes": [t["op"] for t in table if t["t0_ms"] == 0],
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))

    print(f"{a.label} ({a.arch}) -- {n_ops} programs/call, {reps} reps\n")
    print(f"GAP: device kernel {kern:.4f} ms  |  bare eager wall {a.bare_eager_ms:.4f} ms  "
          f"-> gap {100*g['eager_gap_frac']:.2f} % ({g['eager_gap_us_per_program']:.2f} us/program)")
    if a.bare_replay_ms:
        print(f"     bare replay wall {a.bare_replay_ms:.4f} ms -> device-only gap "
              f"{100*g['device_only_gap_frac']:.2f} % "
              f"({g['device_only_gap_us_per_program']:.2f} us/program); the fold pays "
              f"{g['host_per_step_ms']:.4f} ms/step of host on top")
    if a.bare_traced_ms:
        print(f"     bare traced wall {a.bare_traced_ms:.4f} ms -> gap "
              f"{100*g['traced_gap_frac']:.2f} % ({g['traced_gap_us_per_program']:.2f} us/program)"
              f"   trace speedup {g['trace_speedup']:.4f}x")
    s = out["stall_split"]
    print(f"\nSTALL (own-thread divisor): wait_front/TRISC0 {100*s['wait_front_over_trisc0']:.1f} %"
          f"   reserve_back/TRISC2 {100*s['reserve_back_over_trisc2']:.1f} %"
          f"   in:out {s['input_output_ratio']}:1")
    print(f"  falsifier: wait>TRISC0 {bad0}, reserve>TRISC2 {bad2} of {n_stall} rows")
    print(f"RESIDENCY over the {kern:.3f} ms of device kernel time: BRISC "
          f"{100*s['brisc_residency']:.1f} %  NCRISC {100*s['ncrisc_residency']:.1f} %  "
          f"TRISC0 {100*s['trisc0_residency']:.1f} %  TRISC1 {100*s['trisc1_residency']:.1f} %  "
          f"TRISC2 {100*s['trisc2_residency']:.1f} %")
    print(f"\n  {'OP CODE':22s} {'n':>5} {'ms':>8} {'%':>6} {'us':>7} {'cores':>6} "
          f"{'in/T0':>7} {'out/T2':>7} {'in:out':>7}")
    for t in table:
        io = f"{t['in_out']:.2f}" if t["in_out"] else "  -"
        i0 = f"{100*t['in_over_t0']:.1f}" if t["in_over_t0"] else "  -"
        o2 = f"{100*t['out_over_t2']:.1f}" if t["out_over_t2"] else "  -"
        print(f"  {t['op'][:22]:22s} {t['n']:5d} {t['kernel_ms']:8.3f} {t['pct_kernel']:5.1f}% "
              f"{t['mean_us']:7.1f} {t['mean_cores']:6.1f} {i0:>7} {o2:>7} {io:>7}")
    m = out["movement_only"]
    print(f"\nno-arithmetic programs: {m['programs']} of {n_ops}, {m['kernel_ms']:.3f} ms, "
          f"{m['pct_kernel']:.1f} % of kernel time -- {', '.join(m['op_codes'])}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
