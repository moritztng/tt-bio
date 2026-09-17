#!/usr/bin/env python3
"""Re-derive F (the clock-immune fold term) from c10-fixed-cost's RAW per-fold rows.

C12's whole 12.5 s route is "bring F from 3.983 s to 1.0 s", so F is the one number in the campaign
that nobody should take on a summary's word. This reads the committed per-fold rows out of
`origin/wk/c10-fixed-cost` (28 warm folds per size, 4 pinned clock arms), fits fold_s = F + W/MHz by
least squares on 1/MHz, and reports the residual structure, a leave-one-out envelope on F, and the
host CPU accounting per arm.

Run from a tt-bio checkout that has the branch fetched:
    python3 perf/c12_orchestrator/verify_F.py [--out runs/verify_F.json]
"""
import argparse
import json
import statistics as st
import subprocess

REF = "origin/wk/c10-fixed-cost"
PATHS = {512: f"{REF}:perf/c10_fixed_cost/runs/sweep1/512/result.json",
         298: f"{REF}:perf/c10_fixed_cost/runs/sweep1/298/result.json"}
# Row of record, state/c10-fixed-cost.md:39 -- what this script is checking against.
OF_RECORD = {512: (3.9830, 0.1181, 14665.0), 298: (1.9500, 0.0438, 10403.4)}


def fit(points):
    """Least squares of t = F + W*(1/MHz). Returns (F seconds, W Mcycles)."""
    n = len(points)
    sx = sum(1.0 / m for m, _ in points)
    sy = sum(t for _, t in points)
    sxx = sum((1.0 / m) ** 2 for m, _ in points)
    sxy = sum(t / m for m, t in points)
    w = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - w * sx) / n, w


def load(size):
    blob = subprocess.run(["git", "show", PATHS[size]], check=True, capture_output=True, text=True)
    rows = [r for r in json.loads(blob.stdout)["rows"] if r["label"] != "cold"]
    by = {}
    for r in rows:
        by.setdefault(r["clock_MHz"], []).append(r)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    report = {"ref": REF, "sizes": {}}
    for size in sorted(PATHS, reverse=True):
        by = load(size)
        pts = [(m, st.median([r["elapsed_s"] for r in by[m]])) for m in sorted(by)]
        f, w = fit(pts)
        f_rec, tol, w_rec = OF_RECORD[size]
        loo = [fit([p for j, p in enumerate(pts) if j != i])[0] for i in range(len(pts))]
        arms = []
        for m, t in pts:
            rows = by[m]
            arms.append({
                "MHz": m, "n": len(rows), "median_s": round(t, 4),
                "fit_s": round(f + w / m, 4), "resid_s": round(t - (f + w / m), 4),
                "resid_pct": round(100 * (t - (f + w / m)) / t, 3),
                "host_cpu_s": round(st.median([r["host_cpu_s"] for r in rows]), 4),
                "host_utime_s": round(st.median([r["host_utime_s"] for r in rows]), 4),
                "host_cpu_pct_of_wall": round(
                    100 * st.median([r["host_cpu_s"] for r in rows]) / t, 1),
                "settled": all(r["settle"]["settled"] for r in rows),
            })
        # The host burns MORE CPU than wall and its utime tracks the clock-scaled term almost 1:1,
        # which is what a spin-wait looks like: a CPU profiler cannot tell host work from device wait.
        u_lo, u_hi = arms[-1]["host_utime_s"], arms[0]["host_utime_s"]
        report["sizes"][size] = {
            "F_s": round(f, 4), "W_Mcycles": round(w, 1),
            "F_of_record": f_rec, "F_tolerance": tol, "W_of_record": w_rec,
            "F_within_record_bound": abs(f - f_rec) <= tol,
            "F_leave_one_out": [round(x, 4) for x in loo],
            "F_loo_envelope_s": [round(min(loo), 4), round(max(loo), 4)],
            "utime_drop_over_clock_range_s": round(u_hi - u_lo, 4),
            "wall_drop_over_clock_range_s": round(arms[0]["median_s"] - arms[-1]["median_s"], 4),
            "arms": arms,
        }
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
