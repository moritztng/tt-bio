"""Assemble the dfd-scaling curve from the per-run JSONs scaling_run.py wrote.

    python3 scripts/diffdock_port/scaling_table.py ~/scratch/dfd/scaling --out curve.json

Efficiency at K is (complexes/s at K) / (K x median complexes/s at K=1), over the reps of each K.
The K=1 median is the baseline the campaign quotes against, so a single lucky rep cannot set it.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def med(xs):
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--out")
    args = ap.parse_args()
    runs = {}
    for f in sorted(Path(args.rundir).glob("k*_r*.json")):
        d = json.loads(f.read_text())
        runs.setdefault(d["k"], []).append(dict(d, name=f.stem))
    if 1 not in runs:
        raise SystemExit("no K=1 run: there is no baseline to take an efficiency against")
    base = med([r["complexes_per_s"] for r in runs[1]])
    rows = []
    for k in sorted(runs):
        cps = [r["complexes_per_s"] for r in runs[k]]
        clocks = [v["median"] for r in runs[k] for v in r["aiclk_mhz"].values() if v]
        busy = [b for r in runs[k] for b in r["card_busy_fraction"] if b]
        host = [r["host_cpu_busy_fraction"]["median"] for r in runs[k]
                if r["host_cpu_busy_fraction"]["median"]]
        cpu_step = [r["host_cpu_s_per_step"] for r in runs[k] if r["host_cpu_s_per_step"]]
        rows.append({
            "k": k, "reps": len(cps), "chips": runs[k][0]["chips"],
            "complexes_per_s": med(cps), "min": min(cps), "max": max(cps),
            "spread_pct": 100 * (max(cps) - min(cps)) / med(cps),
            "efficiency": med(cps) / (k * base),
            "s_per_complex_per_chip": med([r["s_per_complex_per_chip"] for r in runs[k]]),
            "aiclk_mhz": {"median": med(clocks), "min": min(clocks), "max": max(clocks),
                          "chips": {c: [r["aiclk_mhz"][c] for r in runs[k]]
                                    for c in runs[k][0]["chips"]}},
            "card_busy_fraction": {"median": med(busy), "min": min(busy), "max": max(busy)},
            "host_cpu_busy_fraction": med(host),
            "host_cpu_s_per_step": med(cpu_step),
            "steps_counted": [r["steps_counted"] for r in runs[k]],
            "window_s": [round(r["window_s"], 1) for r in runs[k]],
        })
    nproc = runs[1][0]["nproc"]
    cpu_per_worker = med([r["host_cpu_s_per_step"] for r in runs[1]])
    out = {"baseline_complexes_per_s": base, "nproc": nproc, "rows": rows,
           "host_cores_per_chip": cpu_per_worker / med([r["s_per_complex_per_chip"] / 19
                                                        for r in runs[1]]),
           }
    print(f"{'K':>2} {'cpx/s':>10} {'eff':>7} {'spread%':>8} {'AICLK':>7} {'card%':>7} {'host CPU%':>10}")
    for r in rows:
        print(f"{r['k']:>2} {r['complexes_per_s']:>10.5f} {r['efficiency']:>7.3f} "
              f"{r['spread_pct']:>8.1f} {r['aiclk_mhz']['median']:>7.0f} "
              f"{100 * r['card_busy_fraction']['median']:>7.1f} "
              f"{100 * (r['host_cpu_busy_fraction'] or 0):>10.1f}")
    print(f"host cores busy per chip: {out['host_cores_per_chip']:.2f} of {nproc}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
