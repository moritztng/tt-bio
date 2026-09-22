#!/usr/bin/env python3
"""What each arm costs on the REAL arm, with the AICLK sampled DURING the timed work.

`of3t-softmax`'s per-op factors (precise 1.46x, accurate 4.71x) are a micro-bench on one
softmax shape. A scope cost is not that: the softmax is one verb among many, so the factor
the arm actually pays is diluted by everything else the 48 structures run. This reads the
per-structure seconds the instrument already prints and pairs each arm with the clock its
own window was measured at.
"""
import argparse
import json
import re
import sys

STRUCT = re.compile(r"structure (\d+): ([0-9.]+)s")
WIN = re.compile(r"ARM_(START|END) (\S+) (\d+)")
CLK = re.compile(r"AICLK during the window: n=(\d+) mean=(\d+) MHz min=(\d+) max=(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="label=path.log")
    ap.add_argument("--baseline", default="shipped")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = {}
    for spec in a.logs:
        lab, _, p = spec.partition("=")
        txt = open(p, errors="replace").read()
        secs = [float(m.group(2)) for m in STRUCT.finditer(txt)]
        win = {m.group(1): int(m.group(3)) for m in WIN.finditer(txt)}
        c = CLK.search(txt)
        secs_sorted = sorted(secs)
        rows[lab] = {
            "log": p,
            "structures_timed": len(secs),
            "seconds_per_structure_mean": (sum(secs) / len(secs)) if secs else None,
            "seconds_per_structure_median": (secs_sorted[len(secs) // 2] if secs else None),
            "seconds_per_structure_min": (secs_sorted[0] if secs else None),
            "seconds_per_structure_max": (secs_sorted[-1] if secs else None),
            "wall_seconds_whole_arm": (win.get("END", 0) - win.get("START", 0)) or None,
            "window_start_epoch": win.get("START"),
            "window_end_epoch": win.get("END"),
            "aiclk_during_the_window": (
                {"n": int(c.group(1)), "mean_mhz": int(c.group(2)),
                 "min_mhz": int(c.group(3)), "max_mhz": int(c.group(4))} if c else None),
        }

    base = rows.get(a.baseline, {}).get("seconds_per_structure_mean")
    for lab, r in rows.items():
        m = r["seconds_per_structure_mean"]
        r["x_" + a.baseline] = (m / base) if (m and base) else None

    print(f"{'arm':<18}{'s/struct':>10}{'median':>9}{'x ' + a.baseline:>12}"
          f"{'wall s':>9}{'AICLK mean':>12}{'min':>6}{'max':>6}{'n':>5}")
    for lab, r in rows.items():
        c = r["aiclk_during_the_window"] or {}
        print(f"{lab:<18}{(r['seconds_per_structure_mean'] or 0):>10.2f}"
              f"{(r['seconds_per_structure_median'] or 0):>9.2f}"
              f"{(r['x_' + a.baseline] or 0):>12.3f}{(r['wall_seconds_whole_arm'] or 0):>9}"
              f"{c.get('mean_mhz', 0):>12}{c.get('min_mhz', 0):>6}{c.get('max_mhz', 0):>6}"
              f"{c.get('n', 0):>5}")

    rep = {"what": __doc__.strip().splitlines()[0], "baseline": a.baseline,
           "micro_bench_for_contrast": {
               "source": "perf/of3t_softmax/softmax_cost_qb2c0.json",
               "precise_config_per_op": 1.46, "accurate_softmax_per_op": 4.71,
               "note": "one softmax shape, not a scope cost"},
           "arms": rows}
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
