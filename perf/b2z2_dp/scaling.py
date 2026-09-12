#!/usr/bin/env python3
"""Fold the per-width artifacts `dp_width.py` writes into one scaling table.

Efficiency is against linear extrapolation of the WIDTH-1 arm measured in the same session:

    efficiency(W) = folds_per_hour(W) / (W * folds_per_hour(1))

so a box that scales perfectly reads 1.0 at every width. The width-1 arm is measured twice, at
the start and the end of the ladder, and the spread between those two is the A/A floor: any
efficiency deficit smaller than that floor is not a measurement of anything.

    scaling.py --dir perf/b2z2_dp --out ~/.coworker/state/b2z2_dp/scaling.json \
        --bottleneck "..."
"""
import argparse
import json
import statistics as st
from pathlib import Path


def load(p: Path):
    d = json.loads(p.read_text())
    if not d.get("all_children_ok"):
        return None
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bottleneck", default="")
    ap.add_argument("--arch", default="wormhole")
    args = ap.parse_args()

    widths, repeats = {}, []
    for p in sorted(args.dir.glob("w*.json")):
        d = load(p)
        if d is None:
            print(f"skip {p.name}: children not ok")
            continue
        w = str(d["width"])
        r = d["result"]
        rec = {
            "cards": d["cards"],
            "folds_per_hour": r["folds_per_hour"],
            "n_timed_folds": r["n_timed_folds"],
            "window_s": r["window_s"],
            "median_fold_s": r["median_fold_s"],
            "min_fold_s": r["min_fold_s"],
            "max_fold_s": r["max_fold_s"],
            "median_cores_per_fold": r["median_cores_per_fold"],
            "total_host_cores": r["total_cores"],
            "median_step_ms": r["median_step_ms"],
            "cif_digests": r["cif_digests"],
            "digest_unanimous": r["digest_unanimous"],
            "artifact": p.name,
            "started_utc": d["started_utc"],
            "loadavg_at_start": d["loadavg_at_start"],
            "cotenant_chips_at_start": [n for n in d["occupancy_at_start"].get("nodes", [])
                                        if n.rsplit("/", 1)[-1] not in d["cards"]],
        }
        if w in widths:
            repeats.append((w, rec))
            if rec["folds_per_hour"] > widths[w]["folds_per_hour"]:
                widths[w], rec = rec, widths[w]
                repeats[-1] = (w, rec)
        else:
            widths[w] = rec

    assert "1" in widths, "no width-1 arm; there is nothing to call linear"
    base = widths["1"]["folds_per_hour"]
    for w, rec in widths.items():
        rec["efficiency_vs_linear"] = round(rec["folds_per_hour"] / (int(w) * base), 5)
        rec["speedup_vs_width1"] = round(rec["folds_per_hour"] / base, 4)

    # A/A floor: the same estimator, measured twice at width 1. Quoting a per-fold spread beside a
    # window throughput would be the wrong floor for this statistic.
    aa = [r["folds_per_hour"] for w, r in repeats if w == "1"] + [base]
    floor = None
    if len(aa) > 1:
        floor = {"n": len(aa), "folds_per_hour": sorted(aa),
                 "spread_frac": round((max(aa) - min(aa)) / st.mean(aa), 5)}

    # Digests must agree across every width: a DP arm is the same computation or it is not
    # comparable at all.
    all_digests = sorted({d for r in widths.values() for d in r["cif_digests"]})

    out = {
        "doc": __doc__,
        "arch": args.arch,
        "steps": 200,
        "recycles": 3,
        "size_aa": 512,
        "model": "boltz2",
        "host": "j10glx02 (whglx)",
        "widths": widths,
        "aa_floor_width1": floor,
        "repeat_arms": [{"width": w, **r} for w, r in repeats],
        "cif_digests_all_widths": all_digests,
        "bit_exact_across_widths": len(all_digests) == 1,
        "bottleneck": args.bottleneck,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    for w in sorted(widths, key=int):
        r = widths[w]
        print(f"w={w:>2} {r['folds_per_hour']:>9.2f} folds/h  "
              f"eff={r['efficiency_vs_linear']:.4f}  median_fold={r['median_fold_s']:.2f}s  "
              f"cores/fold={r['median_cores_per_fold']:.3f}")
    print("bit-exact across widths:", out["bit_exact_across_widths"], all_digests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
