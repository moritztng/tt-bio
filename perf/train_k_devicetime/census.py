#!/usr/bin/env python3
"""Summed device kernel time and a per-op-shape census, from a tt-metal ops report.

The report covers everything the process dispatched, warmup included, so the first job here is to
find the boundary of ONE step rather than to trust the whole file. The probe runs `--warmup 1
--steps 1`, so the dispatched sequence is two structurally identical steps: the split is the point
where the op-code sequence of the tail matches the head. That is checked rather than assumed, and
if it does not hold the script says so instead of reporting a number.

Device time here is `DEVICE KERNEL DURATION [ns]` summed over the ops of the timed step, which is
the same unit `train-e-gpu-baseline` summed out of `torch.profiler` for the A100 (2072 ms) -- not a
wall clock, and not `DEVICE FW DURATION`, which adds the per-core dispatch prologue.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

DUR = "DEVICE KERNEL DURATION [ns]"
FW = "DEVICE FW DURATION [ns]"


def shape_key(row: dict, n_in: int = 2) -> str:
    parts = []
    for i in range(n_in):
        dims = []
        for ax in ("W", "Z", "Y", "X"):
            v = row.get(f"INPUT_{i}_{ax}_PAD[LOGICAL]", "")
            if v in ("", None):
                dims = []
                break
            dims.append(str(v).strip())
        if dims:
            parts.append("x".join(dims))
    return " @ ".join(parts) if parts else "-"


def load(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return [r for r in csv.DictReader(fh)]


def split_last_step(rows: list[dict]) -> tuple[list[dict], str]:
    """The last complete repeat of the step's op sequence.

    Two things make this less trivial than an exact-halves split. The cold warmup step misses the
    program cache and dispatches a slightly different number of programs (63 fewer here), so
    halves are not equal and an adjacent-equal-blocks test fails on a file that is periodic at the
    tail. And the step contains 8 structurally identical IPA blocks, so a probe window taken from
    the MIDDLE of a step re-matches an inner block and reports a period of ~1000 instead of the
    step's ~11 700 -- measured, that is exactly what a mid-file cross-check did.

    So the probe is always anchored at the END of the file, where the sequence is unique, and it is
    cross-checked by widening the window rather than by moving it. Three widths that agree on one
    period, each using the EARLIEST match so an inner repeat cannot shorten it, is the check.
    """
    codes = [r.get("OP CODE", "") for r in rows]
    n = len(codes)

    def period(width: int) -> int | None:
        probe = codes[n - width:]
        for i in range(n - width):
            if codes[i:i + width] == probe:
                return n - width - i
        return None

    widths = [w for w in (200, 500, 1000) if w < n // 2]
    found = {w: period(w) for w in widths}
    vals = {v for v in found.values() if v is not None}
    if not vals:
        return [], f"no repeat of the tail's op sequence anywhere in {n} rows"
    if len(vals) > 1:
        return [], f"probe widths disagree on the period: {found} -- refusing to report a step"
    p = vals.pop()
    if p > n:
        return [], f"period {p} exceeds the {n} rows captured -- only part of a step was profiled"
    note = f"period {p} ops, agreed by probe widths {widths} anchored at the file end"
    prev = n - 2 * p
    if prev >= 0:
        same = sum(1 for x, y in zip(codes[n - p:], codes[prev:n - p]) if x == y)
        note += f"; {same}/{p} op codes match the preceding block"
    else:
        note += f"; the preceding block is short by {2 * p - n} ops (the cold step)"
    return rows[n - p:], note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--whole-file", action="store_true",
                    help="report the whole report rather than one step")
    args = ap.parse_args()

    rows = load(args.report)
    rows = [r for r in rows if (r.get(DUR) or "").strip() not in ("", "0")]
    if args.whole_file:
        step, how = rows, f"whole file, {len(rows)} ops"
    else:
        step, how = split_last_step(rows)
    if not step:
        print(how)
        return 1

    total_ns = sum(float(r[DUR]) for r in step)
    fw_ns = sum(float(r[FW]) for r in step if (r.get(FW) or "").strip())
    print(f"SPLIT: {how}")
    print(f"OPS: {len(step)} dispatched programs in the timed step")
    print(f"DEVICE KERNEL TIME: {total_ns / 1e6:.3f} ms summed over the step")
    print(f"DEVICE FW TIME:     {fw_ns / 1e6:.3f} ms (kernel + per-core dispatch prologue)")

    by_op = defaultdict(lambda: [0.0, 0])
    by_shape = defaultdict(lambda: [0.0, 0])
    for r in step:
        d = float(r[DUR])
        code = r.get("OP CODE", "?")
        by_op[code][0] += d
        by_op[code][1] += 1
        key = f"{code}  {shape_key(r)}"
        by_shape[key][0] += d
        by_shape[key][1] += 1

    def table(d, title, top):
        print(f"\n{title}")
        print(f"{'entry':<64} {'ms':>9} {'%':>6} {'n':>6} {'us/call':>9}")
        acc = 0.0
        for k, (ns, n) in sorted(d.items(), key=lambda kv: -kv[1][0])[:top]:
            acc += ns
            print(f"{k[:64]:<64} {ns/1e6:>9.3f} {ns/total_ns*100:>6.2f} {n:>6} {ns/n/1e3:>9.1f}")
        print(f"{'-- top ' + str(top) + ' cumulative':<64} {acc/1e6:>9.3f} "
              f"{acc/total_ns*100:>6.2f}")
        return acc

    table(by_op, "BY OP CODE", args.top)
    top_shape = table(by_shape, "BY OP CODE AND INPUT SHAPE", args.top)
    ranked = sorted(by_shape.items(), key=lambda kv: -kv[1][0])
    for frac in (0.5, 0.6, 0.8, 0.9):
        acc, k = 0.0, 0
        for _, (ns, _n) in ranked:
            acc += ns
            k += 1
            if acc >= frac * total_ns:
                break
        print(f"TAIL: {k} op-shapes of {len(ranked)} carry {frac*100:.0f} % of the device time")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "report": str(args.report), "split": how, "ops": len(step),
            "device_kernel_ms": total_ns / 1e6, "device_fw_ms": fw_ns / 1e6,
            "by_op": {k: {"ms": v[0] / 1e6, "n": v[1]} for k, v in by_op.items()},
            "by_shape": {k: {"ms": v[0] / 1e6, "n": v[1]}
                         for k, v in sorted(by_shape.items(), key=lambda kv: -kv[1][0])},
        }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
