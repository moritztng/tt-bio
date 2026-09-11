#!/usr/bin/env python3
"""Reduce the interleaved A/B to per-arm ratios three ways, and say where they disagree.

The box was co-tenanted for the whole run (`issue14-l1-overflow-release-gate` on card 0, loadavg
8-12), and the load trended DOWN across the session, which is the one kind of noise interleaving
does not remove: a monotone drift biases whichever arm runs later in the rep. So three estimators,
and a result is only claimed where they agree:

1. MEDIAN      medians per arm, and the session A/A floor from the two base folds per rep split by
               their position. This is the form the campaign's protocol mandates.
2. PAIRED      each arm against the base fold nearest to it in the same rep. Cancels drift on the
               scale of one rep at the cost of n.
3. TREND       least squares on log(time) with arm dummies and a linear term in fold index. Uses
               all 25 folds and removes a monotone session drift; the residual RMSE is the honest
               noise level and the A/A floor under this model.

Stages are reported separately because they are an internal control: both levers live inside the
diffusion sampler, so `prepare_and_trunk` and `confidence` must NOT move. If they do, the run read
load, not the lever.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

ARMS = ["base", "A", "B", "AB"]
METRICS = [("fold_s", lambda r: r["fold_s"]),
           ("sampler_s", lambda r: r["stages_s"].get("sampler")),
           ("ms_per_step", lambda r: r["sampler_ms_per_step"]),
           ("trunk_s", lambda r: r["prepare_and_trunk_s"]),
           ("confidence_s", lambda r: r["stages_s"].get("confidence")),
           ("conditioning_s", lambda r: r["stages_s"].get("diffusion_conditioning"))]


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def trend_fit(runs, get):
    """log(t) = mu + alpha[arm] + beta * k. Returns per-arm ratio base/arm and the residual RMSE."""
    rows = [(k, r["arm"], get(r)) for k, r in enumerate(runs) if get(r)]
    if len(rows) < len(ARMS) + 2:
        return None
    arms = [a for a in ARMS if any(r[1] == a for r in rows)]
    other = [a for a in arms if a != "base"]
    X = np.zeros((len(rows), 2 + len(other)))
    y = np.array([math.log(r[2]) for r in rows])
    for i, (k, a, _t) in enumerate(rows):
        X[i, 0] = 1.0
        X[i, 1] = k
        if a in other:
            X[i, 2 + other.index(a)] = 1.0
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rmse = float(np.sqrt((resid ** 2).mean()))
    # parameter standard errors, so a 1 % claim can be checked against its own uncertainty
    dof = max(len(rows) - X.shape[1], 1)
    cov = float(resid @ resid) / dof * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = {"per_fold_drift_pct": round(100 * (math.exp(coef[1]) - 1), 4),
           "resid_rmse_pct": round(100 * rmse, 3), "n": len(rows)}
    for j, a in enumerate(other):
        out[a] = {"speedup": round(math.exp(-coef[2 + j]), 5),
                  "pct_saved": round(100 * (1 - math.exp(coef[2 + j])), 3),
                  "se_pct": round(100 * se[2 + j], 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = json.loads(a.json.read_text())
    runs = [r for r in d["phase1"] if not r.get("warmup")]
    print(f"{len(runs)} warm folds, {len({r['rep'] for r in runs})} reps, "
          f"loadavg at start {d['env']['loadavg']}\n")

    out = {"n_warm_folds": len(runs), "estimators": {}}
    for name, get in METRICS:
        e = {"median": {}, "paired": {}, "trend": trend_fit(runs, get)}
        for arm in ARMS:
            v = [get(r) for r in runs if r["arm"] == arm]
            e["median"][arm] = {"n": len([x for x in v if x is not None]),
                                "median": round(med(v), 4) if med(v) else None,
                                "values": [round(x, 4) for x in v if x is not None]}
        bm = e["median"]["base"]["median"]
        for arm in ("A", "B", "AB"):
            m = e["median"][arm]["median"]
            if bm and m:
                e["median"][arm]["speedup"] = round(bm / m, 5)

        # the A/A floor: base folds split by their position inside the rep
        b = [r for r in runs if r["arm"] == "base"]
        f1 = [get(r) for k, r in enumerate(b) if k % 2 == 0]
        f2 = [get(r) for k, r in enumerate(b) if k % 2 == 1]
        if med(f1) and med(f2):
            e["AA_floor"] = {
                "first_in_rep": round(med(f1), 4), "second_in_rep": round(med(f2), 4),
                "ratio": round(med(f1) / med(f2), 5),
                "pct": round(100 * (med(f1) / med(f2) - 1), 3),
                "spread_pct": round(100 * (max(x for x in f1 + f2 if x) -
                                           min(x for x in f1 + f2 if x)) / med(f1 + f2), 2)}

        # paired: each arm against the nearest base fold in its own rep
        for arm, ref_idx in (("A", 0), ("B", 2), ("AB", 2)):
            rr = []
            for rep in sorted({r["rep"] for r in runs}):
                blk = [r for r in runs if r["rep"] == rep]
                if len(blk) < 5:
                    continue
                ref, tgt = get(blk[ref_idx]), get(next(r for r in blk if r["arm"] == arm))
                if ref and tgt:
                    rr.append(ref / tgt)
            if rr:
                e["paired"][arm] = {"n": len(rr), "median_speedup": round(st.median(rr), 5),
                                    "ratios": [round(x, 5) for x in rr]}
        # and the paired A/A: base@0 against base@2 in the same rep
        rr = []
        for rep in sorted({r["rep"] for r in runs}):
            blk = [r for r in runs if r["rep"] == rep]
            if len(blk) < 5:
                continue
            x, y = get(blk[0]), get(blk[2])
            if x and y:
                rr.append(x / y)
        if rr:
            e["paired"]["AA"] = {"n": len(rr), "median_ratio": round(st.median(rr), 5),
                                 "ratios": [round(x, 5) for x in rr]}
        out["estimators"][name] = e

        print(f"== {name}")
        print(f"   base   n={e['median']['base']['n']:2d} med {e['median']['base']['median']}")
        for arm in ("A", "B", "AB"):
            m = e["median"][arm]
            t = (e["trend"] or {}).get(arm, {})
            p = e["paired"].get(arm, {})
            print(f"   {arm:4s}   n={m['n']:2d} med {m['median']}  "
                  f"median {m.get('speedup')}x  paired {p.get('median_speedup')}x  "
                  f"trend {t.get('speedup')}x +-{t.get('se_pct')}%")
        if "AA_floor" in e:
            print(f"   A/A floor  {e['AA_floor']['ratio']}x "
                  f"({e['AA_floor']['pct']:+.2f} %), spread {e['AA_floor']['spread_pct']} %, "
                  f"paired A/A {e['paired'].get('AA', {}).get('median_ratio')}x")
        if e["trend"]:
            print(f"   drift {e['trend']['per_fold_drift_pct']:+.3f} %/fold, "
                  f"resid RMSE {e['trend']['resid_rmse_pct']} %")
        print()

    # parity / determinism read straight off the CIFs
    out["cif"] = {arm: sorted({r["cif_sha256"] for r in runs if r["arm"] == arm}) for arm in ARMS}
    out["plddt"] = {arm: sorted({round(float(r["plddt"]), 6) for r in runs if r["arm"] == arm})
                    for arm in ARMS}
    out["shape"] = {arm: next((r["diffusion_shape"] for r in runs if r["arm"] == arm), None)
                    for arm in ARMS}
    base_sha = out["cif"]["base"]
    print("cdk2x2_512 CIF sha256 per arm (determinism within an arm, identity across arms):")
    for arm in ARMS:
        s = out["cif"][arm]
        print(f"  {arm:4s} {len(s)} distinct  {[x[:16] for x in s]}  "
              f"{'BIT-EXACT vs base' if s == base_sha else 'differs from base'}  "
              f"plddt {out['plddt'][arm]}  N_padded {out['shape'][arm]}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
