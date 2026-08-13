#!/usr/bin/env python3
"""Reproduce site/index.html's cost arithmetic in Python so a data edit can be checked
before it is published. Mirrors perHour / usdPerHour / perDollar / perDollarIndex.

  python3 perf/perfpage/recompute.py [--of3 44.88]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "site", "data", "perf-512aa.json")
SERVER_OF = {"p150a": "galaxy_bh", "h200": "dgx_h200", "b200": "dgx_b200"}


def per_hour(s, p):
    return 3600.0 / s * p["accelerators"] * p.get("scaling_efficiency", 1.0)


def usd_per_hour(p, years, rate):
    return p["price_usd"] / (years * 8766) + p["power_kw"] * rate


def index(d, model, key, basis, years, rate):
    plat = {p["id"]: p for p in d["platforms"]}
    p, ref = plat[SERVER_OF[key]], plat[d["cost_model"]["index_platform"]]
    ref_key = next(k for k, v in SERVER_OF.items() if v == ref["id"])

    def per_dollar(s, box):
        n = per_hour(s, box)
        return n / (box["price_usd"] if basis == "capex" else usd_per_hour(box, years, rate))

    return per_dollar(model["cells"][key]["s_per_fold"], p) / per_dollar(
        model["cells"][ref_key]["s_per_fold"], ref
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--of3", type=float, help="override OpenFold3's p150a fold time")
    a = ap.parse_args()
    d = json.load(open(DATA))
    if a.of3 is not None:
        for m in d["models"]:
            if m["id"] == "openfold3":
                m["cells"]["p150a"]["s_per_fold"] = a.of3
    cm = d["cost_model"]
    base_y, base_r = cm["amortisation_years"], cm["electricity_usd_per_kwh"]

    print(f"{'model':<12} {'p150a s':>9} {'capex':>8} {'tco':>8}")
    for m in d["models"]:
        c = index(d, m, "p150a", "capex", base_y, base_r)
        t = index(d, m, "p150a", "tco", base_y, base_r)
        print(f"{m['id']:<12} {m['cells']['p150a']['s_per_fold']:>9.3f} {c:>8.4f} {t:>8.4f}")

    print("\nTCO sensitivity, min and max across the five models")
    for label, y, r in [("4 y @ 8.71 c", base_y, base_r), ("3 y @ 8.71 c", 3, base_r),
                        ("6 y @ 8.71 c", 6, base_r), ("4 y @ 13.54 c", base_y, 0.1354)]:
        v = {m["id"]: index(d, m, "p150a", "tco", y, r) for m in d["models"]}
        lo, hi = min(v, key=v.get), max(v, key=v.get)
        print(f"  {label:<14} {v[lo]:.4f} ({lo})  ..  {v[hi]:.4f} ({hi})")


if __name__ == "__main__":
    main()
