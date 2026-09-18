#!/usr/bin/env python3
"""Fold the per-box JSON into one table: step, split, and the comparison against PLAN.md 10c."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# PLAN.md 10c, taken as given. Ours measured by train-b3-train, AICLK 1350 MHz sampled during.
OURS_1CHIP = 30.142
OURS_PAIR = 19.502
DERIVED_ABB3 = 4.167
SCHEDULE_STEPS = 193_512


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.is_file() else None


def days(s: float) -> float:
    return s * SCHEDULE_STEPS / 86400


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    boxes = sorted({p.name.split("_", 1)[1].rsplit("_", 1)[0]
                    for p in root.glob("real_*_s*.json")})
    rows = []
    for tag in boxes:
        row = {"tag": tag}
        for stage in (1, 2):
            r = load(root, f"real_{tag}_s{stage}.json")
            if r:
                res = r["result"]
                row[f"s{stage}"] = res["step_s"]
                row[f"s{stage}_tokens"] = res["padded_tokens_per_step"]
                row[f"s{stage}_peak_GB"] = res["peak_cuda_alloc_GB"]
                row["gpu"] = r["nvidia_smi_idle"]["gpu"][0].split(",")[0]
                row["cpu"] = r["host"]["cpu_model"]
                row["threads"] = r["host"]["cpu_threads"]
            p = load(root, f"prof_{tag}_s{stage}.json")
            if p:
                row[f"s{stage}_cuda_ms"] = p["result"]["cuda_self_ms_per_step"]
        d = load(root, f"data_{tag}_s1.json")
        if d:
            row["data_s"] = d["result"]["data_s_per_step"]
        for ctrl in ("ddp", "dvclive"):
            c = load(root, f"ctrl_{tag}_s1_{ctrl}.json")
            if c:
                row[f"ctrl_{ctrl}"] = c["result"]["step_s"]
        rows.append(row)

    print("=" * 100)
    print("STEP — one optimizer step at global batch 64 (8 micro-batches of 8 + update), full step")
    print("=" * 100)
    print(f"{'box':10} {'GPU':24} {'stage':6} {'median s':>9} {'p05':>7} {'p95':>7} "
          f"{'stdev':>7} {'n':>4} {'tok':>5} {'days/193512':>12}")
    for r in rows:
        for stage in (1, 2):
            s = r.get(f"s{stage}")
            if not s:
                continue
            t = r.get(f"s{stage}_tokens", {})
            print(f"{r['tag']:10} {r.get('gpu','?')[:24]:24} {stage:^6} {s['median']:9.4f} "
                  f"{s['p05']:7.4f} {s['p95']:7.4f} {s['stdev']:7.4f} {s['n']:4d} "
                  f"{t.get('median', 0):5.0f} {days(s['median']):12.2f}")

    print()
    print("=" * 100)
    print("SPLIT — device-busy (CUDA kernel self time) vs host, per step")
    print("=" * 100)
    print(f"{'box':10} {'stage':6} {'step s':>8} {'device s':>9} {'dataload s':>11} "
          f"{'other host s':>13} {'host %':>7}")
    for r in rows:
        for stage in (1, 2):
            s = r.get(f"s{stage}")
            cuda = r.get(f"s{stage}_cuda_ms")
            if not s or cuda is None:
                continue
            dev = cuda / 1e3
            dat = r.get("data_s", {}).get("median", float("nan"))
            other = s["median"] - dev - dat
            host_pct = 100 * (dat + other) / s["median"]
            print(f"{r['tag']:10} {stage:^6} {s['median']:8.4f} {dev:9.4f} {dat:11.4f} "
                  f"{other:13.4f} {host_pct:7.1f}")

    print()
    print("=" * 100)
    print("CONTROLS — the two places the runner departs from upstream's train.py (stage 1)")
    print("=" * 100)
    for r in rows:
        base = r.get("s1")
        for ctrl in ("ddp", "dvclive"):
            c = r.get(f"ctrl_{ctrl}")
            if c and base:
                print(f"{r['tag']:10} +{ctrl:8} {c['median']:8.4f} s vs {base['median']:8.4f} s "
                      f"baseline  ({100 * (c['median'] / base['median'] - 1):+.1f} %, n={c['n']})")

    print()
    print("=" * 100)
    print("A100 AGREEMENT — the >5 % rule from gpu-reference-device-vs-host-split")
    print("=" * 100)
    a = [r for r in rows if r["tag"].startswith("a100") and r.get("s1")]
    if len(a) >= 2:
        for field, label in (("s1", "full step (wall)"),):
            xs = [r[field]["median"] for r in a]
            print(f"{label:22} {' vs '.join(f'{x:.4f}' for x in xs)}  "
                  f"spread {100 * (max(xs) / min(xs) - 1):+.2f} %")
        devs = [r["s1_cuda_ms"] / 1e3 for r in a if r.get("s1_cuda_ms")]
        if len(devs) >= 2:
            print(f"{'device time':22} {' vs '.join(f'{x:.4f}' for x in devs)}  "
                  f"spread {100 * (max(devs) / min(devs) - 1):+.2f} %  "
                  f"{'PASS (<=5 %)' if max(devs) / min(devs) - 1 <= 0.05 else 'FAIL (>5 %) -- publish no bar'}")
        dats = [r["data_s"]["median"] for r in a if r.get("data_s")]
        if len(dats) >= 2:
            print(f"{'host data pipeline':22} {' vs '.join(f'{x:.4f}' for x in dats)}  "
                  f"spread {100 * (max(dats) / min(dats) - 1):+.2f} %")

    print()
    print("=" * 100)
    print("COMPARISON against PLAN.md 10c (our numbers taken as given)")
    print("=" * 100)
    a1 = [r for r in rows if r["tag"].startswith("a100") and r.get("s1")]
    if a1:
        m = statistics.median([r["s1"]["median"] for r in a1])
        print(f"{'ABodyBuilder3 1 A100-80G, MEASURED':45} {m:8.4f} s  {days(m):7.2f} d")
        print(f"{'ABodyBuilder3 1 GPU, PLAN.md derived from >3x':45} {DERIVED_ABB3:8.4f} s  "
              f"{days(DERIVED_ABB3):7.2f} d   derivation is {m / DERIVED_ABB3:.2f}x optimistic")
        print(f"{'ours, 1 Blackhole chip, measured':45} {OURS_1CHIP:8.4f} s  {days(OURS_1CHIP):7.2f} d"
              f"   gap {OURS_1CHIP / m:.2f}x  (was {OURS_1CHIP / DERIVED_ABB3:.2f}x on the derivation)")
        print(f"{'ours, 1 board pair (2 chips)':45} {OURS_PAIR:8.4f} s  {days(OURS_PAIR):7.2f} d"
              f"   gap {OURS_PAIR / m:.2f}x  (was {OURS_PAIR / DERIVED_ABB3:.2f}x)")
    for r in rows:
        if r["tag"].startswith("a100") or not r.get("s1"):
            continue
        m2 = r["s1"]["median"]
        print(f"{r.get('gpu', r['tag'])[:45]:45} {m2:8.4f} s  {days(m2):7.2f} d"
              f"   gap from 1 chip {OURS_1CHIP / m2:.2f}x, from a pair {OURS_PAIR / m2:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
