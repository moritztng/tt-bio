#!/usr/bin/env python3
"""The pc card-0 accuracy legs, one table per rung. Absolute Angstrom is the comparison; the floor
and X/floor only qualify it, because the floor moves per arm.

Two conventions per anchored rung, both printed, because each anchor has one seed whose REFERENCE
sits in the other basin and whose device does not always follow it across: a five-seed mean there
scores which basin the arm landed in, not how accurate it is. 7ROA drops seed 0, ubq drops seed 4.
"""
import json
import statistics
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
RUNGS = [("cdk128", "cdk2_128  ALIGNED control", None),
         ("cdk298", "cdk2_298", None),
         ("roa117", "7ROA L117", "0"),
         ("ubq76", "ubq L76", "4")]


def load(tag):
    out = []
    for f in sorted(R.glob("pc_%s_*.json" % tag)):
        d = json.loads(f.read_text())
        m = d["metrics"]["kabsch_rmsd"]
        rc = d.get("route_counters") or {}
        out.append({"leg": f.stem.replace("pc_%s_" % tag, ""), "arm": d["arm"],
                    "ckc": d["arm_applied"]["resolved"]["_CKC_OVERRIDE"],
                    "site": d["arm_applied"]["resolved"]["tri_att_sdpa_hifi"],
                    "X": m["cross"]["mean"], "floor": m["floor_mean"],
                    "inside": m["within_noise_floor"], "meanD": m["dev_floor"]["mean"],
                    "ps": m["X_per_seed"], "sha": d["dev_sha"],
                    "fused": rc.get("sdpa_route_counts", {}).get("fused"),
                    "stock": rc.get("sdpa_route_counts", {}).get("stock"),
                    "hifi_calls": rc.get("sdpa_hifi_calls"),
                    "picks": rc.get("sdpa_chunk_picks")})
    return out


def drop(ps, seed):
    return statistics.mean(v for k, v in ps.items() if k != seed)


for tag, title, dseed in RUNGS:
    rows = load(tag)
    if not rows:
        continue
    hdr = "X5 (A)" if dseed is None else "X5 (A)  X-drop%s" % dseed
    print("\n== %s   (%s)" % (title, rows[0]["picks"]))
    print("    %-10s %-38s %8s %9s %8s %7s %8s" %
          ("leg", "ckc", "X5 (A)", "Xdrop", "floor", "inside", "mean D"))
    for r in rows:
        ck = r["ckc"] + (" +per-site" if r["site"] else "")
        xd = "%9.4f" % drop(r["ps"], dseed) if dseed else "%9s" % "-"
        print("    %-10s %-38s %8.4f %s %8.4f %7s %8.4f" %
              (r["leg"], ck, r["X"], xd, r["floor"], r["inside"], r["meanD"]))
        print("               per-seed  %s   fused/stock %s/%s  hifi_calls %s" %
              (" ".join("%.4f" % v for _, v in sorted(r["ps"].items())),
               r["fused"], r["stock"], r["hifi_calls"]))

    # The on-card floor: every a1 process is the SAME code on the SAME card. Any spread between
    # them is pc card 0, not an arm (`pc-card0-512aa-fold-nondeterminism`).
    a1 = [r for r in rows if r["arm"] == "a1"]
    if len(a1) >= 2:
        xs = [r["X"] for r in a1]
        spread = max(xs) - min(xs)
        dspread = (max(drop(r["ps"], dseed) for r in a1)
                   - min(drop(r["ps"], dseed) for r in a1)) if dseed else 0.0
        # how many individual seed rollouts deviated from the modal coordinates
        dev = 0
        for s in a1[0]["sha"]:
            c = Counter(r["sha"][s] for r in a1)
            dev += len(a1) - c.most_common(1)[0][1]
        print("    FLOOR (a1 x%d on this card): X5 spread %.4f A, X-drop spread %.4f A, "
              "%d/%d seed rollouts deviated from the modal coordinates"
              % (len(a1), spread, dspread, dev, len(a1) * len(a1[0]["sha"])))
        base5 = statistics.mean(xs)
        based = statistics.mean(drop(r["ps"], dseed) for r in a1) if dseed else None
        for r in rows:
            if r["arm"] == "a1":
                continue
            d5 = r["X"] - base5
            v5 = "inside floor" if abs(d5) <= spread else ("better" if d5 < 0 else "WORSE")
            line = "      %-10s X5 %+.4f -> %-12s" % (r["leg"], d5, v5)
            if dseed:
                dd = drop(r["ps"], dseed) - based
                vd = "inside floor" if abs(dd) <= max(dspread, 1e-9) else \
                     ("better" if dd < 0 else "WORSE")
                line += "  Xdrop %+.4f -> %s" % (dd, vd)
            print(line)
