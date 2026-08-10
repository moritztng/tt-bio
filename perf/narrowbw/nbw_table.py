#!/usr/bin/env python3
"""Render the deliverable table for z-narrowbw-512 from the arm JSONs + gate JSONs.

Reads perf/narrowbw/nbw_{512,298}_qb2c0.json and gate_cap*.json, prints one row per cap:
ms/fold at each size (sum of the three narrow site walls, delta vs the bracketing `on` arms),
the hsa fraction of bound, plDDT and the CIF sha identity.
"""
import json, sys, statistics as st
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
NBW = ROOT / "perf" / "narrowbw"
SITES = ["lin|pairbias|c256@16", "lin|pwa|c256@1", "lin|template|c256@64"]


def analyse(runs):
    ok = [r for r in runs if "wall_ms" in r]
    ons = [r for r in ok if r["arm"] == "on"]
    keys = sorted({k for r in ok for k in r["wall_ms"]})
    out = {"n_on_arms": len(ons), "aa_floor_ms": {}, "deltas_ms": {}}
    for k in keys:
        v = [r["wall_ms"][k]["ms"] for r in ons if k in r["wall_ms"]]
        if not v:
            continue
        out["aa_floor_ms"][k] = {"n": len(v), "spread": round(max(v) - min(v), 2),
                                 "stdev": round(st.stdev(v), 2) if len(v) > 1 else None,
                                 "median": round(st.median(v), 2),
                                 "calls": ons[0]["wall_ms"].get(k, {}).get("calls")}
    for r in ok:
        if r["arm"] == "on":
            continue
        before = [x for x in ok if x["i"] < r["i"] and x["arm"] == "on"]
        after = [x for x in ok if x["i"] > r["i"] and x["arm"] == "on"]
        br = ([before[-1]] if before else []) + ([after[0]] if after else [])
        d = {"bracketed_by": [x["i"] for x in br], "walls": {},
             "fold_s_delta": round(r["fold_s"] - st.mean([x["fold_s"] for x in br]), 3) if br else None}
        for k in keys:
            base = [x["wall_ms"][k]["ms"] for x in br if k in x["wall_ms"]]
            if not base or k not in r["wall_ms"]:
                continue
            delta = r["wall_ms"][k]["ms"] - st.mean(base)
            floor = out["aa_floor_ms"].get(k, {}).get("spread")
            d["walls"][k] = {"off_minus_on_ms_per_fold": round(delta, 2),
                             "calls": r["wall_ms"][k]["calls"],
                             "aa_spread_ms": floor,
                             "resolved": floor is not None and abs(delta) > floor}
        d["walls"] = dict(sorted(d["walls"].items(),
                                 key=lambda kv: -abs(kv[1]["off_minus_on_ms_per_fold"])))
        out["deltas_ms"][f"{r['i']}:{r['arm']}"] = d
    return out


def load(size):
    p = NBW / f"nbw_{size}_qb2c0.json"
    if not p.exists():
        return None
    res = json.loads(p.read_text())
    res["analysis"] = res.get("analysis") or analyse(res["runs"])
    return res


def site_sum(res, arm):
    """Signed sum over the three narrow site walls of (arm - bracketing on), i.e. arm-minus-on.
    A cap that is faster than cap 1 gives a NEGATIVE delta; we report the saving as its negation."""
    for key, d in res["analysis"]["deltas_ms"].items():
        if key.split(":", 1)[1] != arm:
            continue
        tot, floor, det = 0.0, 0.0, {}
        for s in SITES:
            w = d["walls"].get(s)
            if not w:
                continue
            tot += w["off_minus_on_ms_per_fold"]
            floor += (w["aa_spread_ms"] or 0.0)
            det[s] = w
        return {"arm": arm, "delta_ms": round(tot, 2), "saved_ms": round(-tot, 2),
                "floor_sum_ms": round(floor, 2), "walls": det,
                "bracketed_by": d["bracketed_by"], "fold_s_delta": d["fold_s_delta"]}
    return None


def plddt(res, arm):
    v = [(r["plddt"], r["cif_sha256"]) for r in res["runs"] if r["arm"] == arm and "plddt" in r]
    return v[0] if v else (None, None)


def gate(cap):
    p = NBW / f"gate_cap{cap}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


for size in (512, 298):
    res = load(size)
    print(f"\n===== {size} aa =====")
    if not res:
        print("  (no results yet)")
        continue
    print(f"  ttnn={res['ttnn']} grid={res['grid']} reblock_permute={res['reblock_permute']}")
    print(f"  runs done: {len(res['runs'])}  on-arms: {res['analysis']['n_on_arms']}")
    for s in SITES + ["block:PairformerLayer", "stage:Pairformer", "stage:template", "stage:msa"]:
        f = res["analysis"]["aa_floor_ms"].get(s)
        if f:
            print(f"  A/A floor {s:<26s} n={f['n']} median={f['median']:>10.2f} "
                  f"spread={f['spread']:>8.2f} stdev={f['stdev']} calls={f['calls']}")
    for arm in ("off:narrowbw", "bw:2", "bw:4", "bw:8", "bw:16", "bw:8+pairbw1"):
        r = site_sum(res, arm)
        if not r:
            continue
        pl, sha = plddt(res, arm)
        print(f"  {arm:<14s} site-sum delta {r['delta_ms']:>9.2f} ms  saved {r['saved_ms']:>9.2f}  "
              f"floorsum {r['floor_sum_ms']:>7.2f}  fold_s_delta {r['fold_s_delta']}  "
              f"plddt {pl}  sha {str(sha)}  brack {r['bracketed_by']}")
        for s, w in r["walls"].items():
            print(f"        {s:<26s} {w['off_minus_on_ms_per_fold']:>9.2f} ms over {w['calls']:>4d} "
                  f"calls  floor {w['aa_spread_ms']}  resolved={w['resolved']}")
    pl_on = [(r["plddt"], list(r["cif_sha256"].values())) for r in res["runs"] if r["arm"] == "on"]
    print(f"  on-arm plddt/sha: {pl_on}")

print("\n===== gate =====")
for cap in (1, 2, 4, 8):
    g = gate(cap)
    print(f"  cap {cap}: {'present' if g else 'not run yet'}")
