#!/usr/bin/env python3
"""What a GiB of host headroom is worth, in seconds of the step's own host arithmetic. No card.

This row's finding is that host memory is a perf variable in its own right: inside ONE run the
same four roots of the same loss arithmetic cost 1.925 s at 11.82 GiB MemAvailable and 4.999 s
at 9.06 GiB (`out/shape_per-root_s4.json`). Two points settle that it happens; they do not
price it, and a two-point rate linearised over a cliff is exactly the mistake this row spent a
pass undoing on somebody else's slope.

So: sweep MemAvailable with a balloon and time the real `af3_loss` at the real crop, plus an
AdamW-shaped fp32 update over a FRACTION of the optimizer's census (`--opt-frac`, scaled up in
the report and labelled as scaled, because running the full 381.3 M-element update needs
4.26 GiB of its own and would move the very variable being swept).

Read the curve, not a slope: the point of the artifact is where the knee is.
"""
from __future__ import annotations

import argparse, gc, json, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf/of3t_p10host"))

from tt_bio.train.objectives import af3_loss          # noqa: E402
from tt_bio.train.losses import of3_loss_weights      # noqa: E402
from lossprofile import build                          # noqa: E402

N_ELEM = 381_302_188


def avail_gib():
    return round([int(l.split()[1]) for l in open("/proc/meminfo")
                  if l.startswith("MemAvailable")][0] / 1048576, 2)


def time_loss(n, weights, reps):
    rng = np.random.default_rng(0)
    labels, outputs, _ = build(n, rng)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        af3_loss(labels, outputs, weights)
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def time_opt(nel, reps):
    """One AdamW update over `nel` fp32 elements: the arithmetic, no crossing."""
    m = np.zeros(nel, np.float32); v = np.zeros(nel, np.float32)
    th = np.zeros(nel, np.float32); g = np.ones(nel, np.float32)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        m *= 0.9;  m += 0.1 * g
        v *= 0.999; v += 0.001 * (g * g)
        th -= 3e-4 * ((m / 0.1) / (np.sqrt(v / 0.001) + 1e-8) + 0.01 * th)
        ts.append(time.perf_counter() - t0)
    del m, v, th, g
    gc.collect()
    return min(ts), float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--opt-frac", type=float, default=0.125)
    ap.add_argument("--hold", default="0,4,6,8,10,12,14",
                    help="GiB to hold resident at each point of the sweep")
    ap.add_argument("--out", type=Path, default=REPO / "perf/of3t_p10host/out/memprice.json")
    a = ap.parse_args()

    weights = of3_loss_weights("initial_training")
    nel = int(N_ELEM * a.opt_frac)
    rows = []
    for gib in [float(x) for x in a.hold.split(",")]:
        hold = None
        if gib > 0:
            hold = np.empty(int(gib * (1024 ** 3) / 8), np.float64)
            hold[:] = 1.0                      # touch every page; np.empty is not resident
        av = avail_gib()
        lo_min, lo_med = time_loss(a.tokens, weights, a.reps)
        op_min, op_med = time_opt(nel, a.reps)
        rows.append({"hold_gib": gib, "mem_available_gib": av,
                     "af3_loss_s_min": round(lo_min, 4), "af3_loss_s_med": round(lo_med, 4),
                     "opt_update_s_min": round(op_min, 4),
                     "opt_update_scaled_to_census_s": round(op_min / a.opt_frac, 4)})
        print(f"hold {gib:5.1f} GiB  MemAvailable {av:6.2f}  af3_loss {lo_min:.3f}s  "
              f"opt update (scaled to 381.3 M) {op_min / a.opt_frac:.3f}s", flush=True)
        del hold
        gc.collect()

    base = rows[0]
    for r in rows:
        r["af3_loss_x_vs_quiet"] = round(r["af3_loss_s_min"] / base["af3_loss_s_min"], 3)
        r["opt_update_x_vs_quiet"] = round(r["opt_update_s_min"] / base["opt_update_s_min"], 3)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"tokens": a.tokens, "opt_frac": a.opt_frac, "opt_elements": nel,
         "census_elements": N_ELEM, "loadavg": open("/proc/loadavg").read().split()[:3],
         "rows": rows,
         "note": "opt_update_scaled_to_census_s is the measured fraction divided by opt_frac, "
                 "so it is a SCALED figure, not a measurement of the full census."},
        indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
