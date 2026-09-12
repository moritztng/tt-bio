#!/usr/bin/env python3
"""The A/A floor, bootstrapped at the n the ratio is quoted at.

The headline is a MEDIAN OF n PAIRED RATIOS, so its null is the spread of that same estimator when
both arms are the base. A split-half floor answers a different question -- it is the floor of a
median of n/2 -- and is roughly 1.5x too wide, which nearly cost `b2z2-trunk-fold-ab-bh` a real
1.026x lever (2026-09-13). So does the folded median of adjacent pairs this row first quoted: it
is a median of n-1 one-sided values, not a two-sided band on a median of n.

What this does instead: resample n base-vs-base ratios with replacement and take their median,
B times, and report the 2.5 / 97.5 percentiles of that distribution. A lever whose paired ratio
sits above the 97.5th percentile of its own session's null is readable; one inside the band is not,
whatever it is worth on a smaller instrument.

Two null populations, because the pairing is what cancels drift:

  all-pairs   every ordered i != j of the session's base folds. Conservative: it admits pairs
              minutes apart, which the real estimator never forms.
  adjacent    only base folds from consecutive reps, and both orientations. Closer to what the
              paired estimator actually does, fewer distinct values to draw from.

Host only. Reads the committed session JSONs, invents nothing.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from random import Random

H = Path(__file__).resolve().parent
B = 20000


def bootstrap(pool, n, seed=0):
    rng = Random(seed)
    meds = sorted(st.median(rng.choices(pool, k=n)) for _ in range(B))
    lo = meds[int(0.025 * B)]
    hi = meds[int(0.975 * B) - 1]
    return {"lo": round(lo, 5), "hi": round(hi, 5), "n": n, "pool": len(pool), "B": B}


def session(path: Path) -> dict:
    d = json.loads(path.read_text())
    timed = [r for r in d["runs"] if not r["warmup"]]
    base = [r["fold_s"] for r in sorted((r for r in timed if r["arm"] == "base"),
                                        key=lambda r: r["rep"])]
    arms = [a for a in d["env"]["arms"] if a != "base"]
    n = len(base)
    all_pairs = [base[i] / base[j] for i in range(n) for j in range(n) if i != j]
    adj = [f(base[k], base[k + 1]) for k in range(n - 1)
           for f in (lambda a, b: a / b, lambda a, b: b / a)]
    out = {"file": path.name, "n_base": n, "quoted_n": d["env"]["reps"],
           "loadavg": [d["env"]["loadavg"][0], d["env"].get("loadavg_end", [None])[0]],
           "base_fold_s": [round(x, 3) for x in base],
           "floor_all_pairs": bootstrap(all_pairs, d["env"]["reps"]),
           "floor_adjacent": bootstrap(adj, d["env"]["reps"]),
           "old_folded_median_floor": d.get("aa_floor_paired"),
           "arms": {}}
    # The same bootstrap on a STAGE wall, because a trunk lever's instrument is the trunk stage
    # and a stage is quieter than the fold it sits in. Still an in-fold measurement: no block is
    # grabbed and replayed.
    out["stages"] = {}
    for stage in sorted({k for r in timed for k in r["stages_s"]}):
        bs = [r["stages_s"][stage] for r in sorted(
            (r for r in timed if r["arm"] == "base" and stage in r["stages_s"]),
            key=lambda r: r["rep"])]
        if len(bs) < 3:
            continue
        m = len(bs)
        pool = [bs[i] / bs[j] for i in range(m) for j in range(m) if i != j]
        f = bootstrap(pool, d["env"]["reps"])
        row = {"floor": f, "arms": {}}
        for a in arms:
            per = []
            for i in range(d["env"]["reps"]):
                b = [r["stages_s"][stage] for r in timed
                     if r.get("rep") == i and r["arm"] == "base" and stage in r["stages_s"]]
                x = [r["stages_s"][stage] for r in timed
                     if r.get("rep") == i and r["arm"] == a and stage in r["stages_s"]]
                if b and x:
                    per.append(b[0] / x[0])
            if per:
                r_ = st.median(per)
                row["arms"][a] = {"paired_ratio": round(r_, 5), "readable": r_ > f["hi"],
                                  "all_reps": [round(v, 5) for v in per]}
        out["stages"][stage] = row

    hi = out["floor_all_pairs"]["hi"]
    for a in arms:
        r = d["paired_ratio"].get(a)
        if r is None:
            continue
        out["arms"][a] = {
            "paired_ratio": r, "floor_hi": hi,
            "excess_over_floor": round((r - 1) / (hi - 1), 2) if hi > 1 else None,
            "readable": r > hi,
            "all_reps": d["paired_ratio_all"][a]}
    return out


def main() -> int:
    files = [Path(p) for p in sys.argv[1:]] or sorted(H.glob("*_everything_wh_c*.json"))
    rep = {"doc": __doc__.splitlines()[0], "sessions": [session(p) for p in files]}
    (H / "floor.json").write_text(json.dumps(rep, indent=1))
    for s in rep["sessions"]:
        f, adj = s["floor_all_pairs"], s["floor_adjacent"]
        print(f"\n=== {s['file']}  n={s['quoted_n']}  loadavg {s['loadavg']}")
        print(f"  floor all-pairs  [{f['lo']}, {f['hi']}]   adjacent [{adj['lo']}, {adj['hi']}]"
              f"   (this row's first, folded-median: {s['old_folded_median_floor']})")
        for a, v in s["arms"].items():
            print(f"  {a:9s} {v['paired_ratio']:.5f}  {v['excess_over_floor']:>6}x the floor  "
                  f"{'READABLE' if v['readable'] else 'inside the floor'}")
        for stage, row in s.get("stages", {}).items():
            f2 = row["floor"]
            print(f"  -- stage {stage:22s} floor [{f2['lo']}, {f2['hi']}]")
            for a, v in row["arms"].items():
                print(f"       {a:9s} {v['paired_ratio']:.5f}  "
                      f"{'READABLE' if v['readable'] else 'inside the floor'}")
    print("\nwrote", H / "floor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
