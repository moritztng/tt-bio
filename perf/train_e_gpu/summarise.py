#!/usr/bin/env python3
"""Fold the per-box JSON into one table: step, device/host split, and the comparison to PLAN.md 10c.

One row = one rented box. One step = one optimizer step at ABodyBuilder3's own global batch of 64
(8 micro-batches of 8 plus the update), their code, their params.yaml, their data.
"""
from __future__ import annotations

import csv
import datetime as dt
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


def util_during(root: Path, tag: str, rec: dict) -> tuple[float, int] | None:
    """Mean GPU utilisation over the run's own window, from the clock samples taken during it.

    nvidia-smi utilisation is the fraction of sampled time with at least one kernel resident, so
    util x step is a device-busy estimate. It is a coarse duty-cycle sampler and is reported as a
    cross-check on the profiler, not as the instrument that settles the 5 % device-time question.
    """
    rows = []
    for cf in sorted(root.glob(f"clock*_{tag}.csv")):
        for r in csv.DictReader(open(cf)):
            try:
                rows.append((dt.datetime.strptime(r["utc"], "%Y-%m-%dT%H:%M:%SZ"),
                             int(r["util_pct"]), int(r["sm_mhz"]), int(r["n_apps"])))
            except (ValueError, KeyError):
                pass
    t0 = dt.datetime.strptime(rec["started_utc"], "%Y-%m-%dT%H:%M:%SZ")
    t1 = dt.datetime.strptime(rec["finished_utc"], "%Y-%m-%dT%H:%M:%SZ")
    w = [r for r in rows if t0 <= r[0] <= t1]
    if not w:
        return None
    return statistics.fmean(r[1] for r in w) / 100.0, max(r[3] for r in w)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    tags = sorted({p.name.split("_", 1)[1].rsplit("_", 1)[0] for p in root.glob("real_*_s*.json")})

    rows = []
    for tag in tags:
        row = {"tag": tag}
        for stage in (1, 2):
            r = load(root, f"real_{tag}_s{stage}.json")
            if not r:
                continue
            res = r["result"]
            row[f"s{stage}"] = res["step_s"]
            row[f"s{stage}_tok"] = res["padded_tokens_per_step"]
            row[f"s{stage}_GB"] = res["peak_cuda_alloc_GB"]
            row["gpu"] = r["nvidia_smi_idle"]["gpu"][0].split(",")[0]
            row["sm_max"] = r["nvidia_smi_idle"]["gpu"][0].split(",")[3].strip()
            row["cpu"] = r["host"]["cpu_model"].replace("(R)", "").replace("CPU @ 2.40GHz", "")
            row["threads"] = r["host"]["cpu_threads"]
            u = util_during(root, tag, r)
            if u and stage == 1:
                row["util"], row["max_apps"] = u
        for key, fname in (("data", f"data_{tag}_s1.json"), ("data2", f"data2_{tag}_s1.json")):
            d = load(root, fname)
            if d:
                row[key] = d["result"]["data_s_per_step"]
        w = load(root, f"w8_{tag}_s1.json")
        if w:
            row["w8"] = w["result"]["step_s"]
            row["w8_loader"] = w["loader"]
            u = util_during(root, tag, w)
            if u:
                row["w8_util"] = u[0]
        for pf in (f"prof2_{tag}_s1.json", f"prof_{tag}_s1.json"):
            p = load(root, pf)
            if p:
                row["cuda_ms"] = p["result"]["cuda_self_ms_per_step"]
                row["cuda_ms_src"] = pf
                row["launches"] = p["result"].get("kernel_launches_per_step")
                break
        for ctrl in ("ddp", "dvclive"):
            c = load(root, f"ctrl_{tag}_s1_{ctrl}.json")
            if c:
                row[ctrl] = c["result"]["step_s"]
        rows.append(row)

    W = 104
    print("=" * W)
    print("STEP  one optimizer step at global batch 64 (8 x 8 + update), ABodyBuilder3's own code")
    print("=" * W)
    print(f"{'box':8} {'GPU':22} {'stage':5} {'median s':>9} {'p05':>7} {'p95':>7} {'stdev':>7} "
          f"{'n':>4} {'tok':>4} {'days':>7}")
    for r in rows:
        for stage in (1, 2):
            s = r.get(f"s{stage}")
            if not s:
                continue
            print(f"{r['tag']:8} {r.get('gpu','?'):22} {stage:^5} {s['median']:9.3f} "
                  f"{s['p05']:7.3f} {s['p95']:7.3f} {s['stdev']:7.3f} {s['n']:4d} "
                  f"{int(r[f's{stage}_tok']['median']):4d} {days(s['median']):7.2f}")

    print()
    print("=" * W)
    print("SPLIT  device vs host, stage 1. num_workers: 0 as shipped, so the host data pipeline is")
    print("       serialised with compute and is valid to subtract.")
    print("=" * W)
    print(f"{'box':8} {'cpu':30} {'thr':>4} {'step s':>7} {'data s':>7} {'data%':>6} "
          f"{'util':>6} {'dev s':>6} {'dev%':>5} {'cuda ms':>8}")
    for r in rows:
        s = r.get("s1")
        if not s:
            continue
        step = s["median"]
        d = r.get("data", {}).get("median")
        u = r.get("util")
        print(f"{r['tag']:8} {r.get('cpu','?')[:30]:30} {r.get('threads',0):4d} {step:7.3f} "
              f"{d if d is None else round(d,3)!s:>7} "
              f"{'' if d is None else f'{100*d/step:5.1f}%':>6} "
              f"{'' if u is None else f'{100*u:5.1f}%':>6} "
              f"{'' if u is None else f'{u*step:6.3f}':>6} "
              f"{'' if u is None else f'{100*u:4.0f}%':>5} "
              f"{r.get('cuda_ms','-') if isinstance(r.get('cuda_ms'),str) else round(r.get('cuda_ms',0),1) or '-':>8}")

    if any("data2" in r for r in rows):
        print("\nhost-pipeline repeat (same box, same code, later):")
        for r in rows:
            if "data2" in r:
                print(f"  {r['tag']:8} first {r['data']['median']:.3f} s  repeat "
                      f"{r['data2']['median']:.3f} s (n={r['data2']['n']})")

    if any("w8" in r for r in rows):
        print()
        print("=" * W)
        print("DATALOADER-OVERLAP ARM  identical code/data/batch, num_workers 8 + pin_memory.")
        print("       params.yaml ships num_workers: 0; this prices the overlap, labelled separately.")
        print("=" * W)
        print(f"{'box':8} {'shipped s':>10} {'workers=8 s':>12} {'speedup':>8} {'util':>7} {'days':>7}")
        for r in rows:
            if "w8" not in r:
                continue
            base = r["s1"]["median"]
            w = r["w8"]["median"]
            wu = r.get("w8_util")
            print(f"{r['tag']:8} {base:10.3f} {w:12.3f} {base/w:7.2f}x "
                  f"{'-' if wu is None else f'{100*wu:.1f}%':>7} {days(w):7.2f}")

    ctrls = [(r["tag"], k, r[k]["median"], r["s1"]["median"]) for r in rows
             for k in ("ddp", "dvclive") if k in r]
    if ctrls:
        print("\nrunner-vs-upstream controls (the two places this runner departs from train.py):")
        for tag, k, v, base in ctrls:
            print(f"  {tag:8} {k:8} {v:7.3f} s vs {base:.3f} s baseline  ({100*(v/base-1):+.1f}%)")

    print()
    print("=" * W)
    print("COMPARISON  against PLAN.md 10c. Ours taken as given (train-b3-train, AICLK 1350 MHz).")
    print("=" * W)
    ref = [r for r in rows if "A100" in r.get("gpu", "") and "s1" in r]
    print(f"{'':44} {'s/step':>9} {'days/193512':>12} {'ours 1chip':>11} {'ours pair':>10}")

    def line(label, s):
        print(f"{label:44} {s:9.3f} {days(s):12.2f} {OURS_1CHIP/s:10.2f}x {OURS_PAIR/s:9.2f}x")

    print(f"{'abb2, 1 GPU, published':44} {'not stated':>9} {'~28 d/model':>12}")
    line("abb3, 1 GPU, DERIVED from >3x (10c)", DERIVED_ABB3)
    for r in ref:
        line(f"abb3, {r['tag']} {r.get('gpu','')}, MEASURED", r["s1"]["median"])
    for r in rows:
        if "A100" not in r.get("gpu", "") and "s1" in r:
            line(f"abb3, {r['tag']} {r.get('gpu','')}, MEASURED", r["s1"]["median"])
    for r in rows:
        if "w8" in r:
            line(f"abb3, {r['tag']}, workers=8 arm", r["w8"]["median"])
    print(f"{'ours, 1 Blackhole chip, measured':44} {OURS_1CHIP:9.3f} {days(OURS_1CHIP):12.2f}")
    print(f"{'ours, 1 board pair (2 chips)':44} {OURS_PAIR:9.3f} {days(OURS_PAIR):12.2f}")

    if len(ref) >= 2:
        a, b = ref[0]["s1"]["median"], ref[1]["s1"]["median"]
        hi, lo = max(a, b), min(a, b)
        print(f"\nTWO-RENTAL AGREEMENT on the full step: {ref[0]['tag']} {a:.3f} s vs "
              f"{ref[1]['tag']} {b:.3f} s -> {100*(hi/lo-1):.1f}% apart "
              f"({'within' if hi/lo-1 <= 0.05 else 'OUTSIDE'} the 5% bar)")
        da = [r.get("cuda_ms") for r in ref]
        if all(da):
            hi, lo = max(da), min(da)
            print(f"TWO-RENTAL AGREEMENT on device time:   {da[0]:.0f} ms vs {da[1]:.0f} ms "
                  f"-> {100*(hi/lo-1):.1f}% apart "
                  f"({'within' if hi/lo-1 <= 0.05 else 'OUTSIDE'} the 5% bar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
