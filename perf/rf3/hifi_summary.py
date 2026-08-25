#!/usr/bin/env python3
"""Regenerate the RF3 HiFi-arm tables straight out of the result JSONs. No number is typed by hand."""
import json, glob, os, statistics as st, sys

def acc(path):
    d = json.load(open(path))
    k = d["metrics"]["kabsch_rmsd"]
    ps = k["X_per_seed"]
    return dict(arm=d["arm"], X=k["cross"]["mean"], floor=k["floor_mean"],
                inside=k["within_noise_floor"], per_seed=ps,
                meanD=st.mean(k["D_pairs"]), meanR=st.mean(k["R_pairs"]),
                xof=k["cross_over_floor"], sha=d.get("dev_sha"))

def table(pat, drop):
    out = []
    for f in sorted(glob.glob(pat)):
        a = acc(f)
        kept = [v for s, v in a["per_seed"].items() if s != drop]
        a["Xdrop"] = st.mean(kept)
        out.append(a)
    return out

for rung, pat, drop in (("7ROA L117", "perf/rf3/results/hifi_roa117_a*.json", "0"),
                        ("ubq L76",   "perf/rf3/results/hifi_ubq76_a*.json",  "4")):
    print("== %s  (drop seed %s)" % (rung, drop))
    print("   arm      X(5)     X(drop)   meanD    meanR    floor    inside")
    for a in table(pat, drop):
        print("   %-8s %.4f   %.4f    %.4f   %.4f   %.4f   %s"
              % (a["arm"], a["X"], a["Xdrop"], a["meanD"], a["meanR"], a["floor"], a["inside"]))
    print()
