"""Split a whole-process ttnn op profile into its one-time and per-step parts.

A profiling run captures everything the process issues: the token initializer
(``_init_atoms`` and friends, once per design) followed by the diffusion steps.
Reporting a percentage over that whole stream silently charges one-time setup
to the per-step budget -- the token initializer alone uploads three [L,L,1]
atom-pair masks, which is 13% of a whole-process capture and 0.4% of a real
200-step run.

The diffusion loop issues an identical op sequence every step, so the per-step
region is recoverable without a second device run: find the longest suffix of
the op stream that occurs twice back to back.  That period is one step; the
prefix before the repeats is the one-time work.

Usage:
    python segment_step_profile.py <ops_perf_results.csv> [--label D=1] [--top 16]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DUR = "DEVICE KERNEL DURATION [ns]"
FPU = "PM FPU UTIL (%)"
CORES = "CORE COUNT"


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def signature(row) -> str:
    def shape(prefix):
        return "x".join(
            (row.get(f"{prefix}_{axis}_PAD[LOGICAL]") or "").strip()
            for axis in ("W", "Z", "Y", "X")
        )

    return f"{row.get('OP CODE')}|{shape('INPUT_0')}|{shape('INPUT_1')}"


def find_period(sigs: list[str], min_period: int = 50) -> int | None:
    """Longest P with sigs[-P:] == sigs[-2P:-P]; that is one loop iteration."""
    for period in range(len(sigs) // 2, min_period - 1, -1):
        if sigs[-period:] == sigs[-2 * period : -period]:
            return period
    return None


def report(rows, title: str, top: int) -> None:
    total = 0.0
    fpu_weighted = 0.0
    fpu_ns = 0.0
    agg: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for row in rows:
        dur = num(row.get(DUR))
        if not dur:
            continue
        total += dur
        entry = agg[row.get("OP CODE", "?")]
        entry[0] += 1
        entry[1] += dur
        entry[4] += (num(row.get(CORES)) or 0.0) * dur
        fpu = num(row.get(FPU))
        if fpu is not None:
            entry[2] += dur * fpu
            entry[3] += dur
            fpu_weighted += dur * fpu
            fpu_ns += dur

    print(f"\n## {title}")
    print(f"ops={len(rows)} device_kernel_ms={total / 1e6:.3f}", end="")
    if fpu_ns:
        print(f"  time-weighted FPU util={fpu_weighted / fpu_ns:.2f}%")
    else:
        print()
    print(f"{'OP CODE':36s} {'n':>5s} {'ms':>8s} {'%':>6s} {'FPU%':>7s} {'cores':>6s}")
    for code, e in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
        fpu = e[2] / e[3] if e[3] else float("nan")
        print(
            f"{code[:36]:36s} {e[0]:5d} {e[1] / 1e6:8.3f} "
            f"{e[1] / total * 100:5.1f}% {fpu:7.2f} {e[4] / e[1]:6.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--top", type=int, default=16)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_path.open()))
    sigs = [signature(r) for r in rows]
    period = find_period(sigs)
    label = f" [{args.label}]" if args.label else ""
    print(f"# {args.csv_path.name}{label}: {len(rows)} ops")
    if period is None:
        print("no repeated step period found -- reporting whole stream only")
        report(rows, "whole process", args.top)
        return

    print(f"detected per-step period = {period} ops "
          f"({len(rows) // period} repeats fit in the stream)")
    report(rows[:-2 * period], "one-time (token initializer + setup)", args.top)
    report(rows[-period:], "ONE DIFFUSION STEP", args.top)


if __name__ == "__main__":
    main()
