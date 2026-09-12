#!/usr/bin/env python3
"""Is the union of the three host-to-device ports additive, sub-additive, or neither?

Each lever was measured alone against its own base, on a different card at a different commit, so
the only honest way to combine them is as RATIOS -- seconds do not transfer between whglx cards or
between commits on this campaign (state/b2z2/CONTEXT.md, 2-CORRECTION).

Every lever here is ~1 % on a fold whose own A/A floor is the same size, so the MARGIN does not
carry the claim and this script does not pretend it does. What it reports is the SIGN across
interleaved pairs (6 positive out of 6 is p = 1/64 under the null) beside the A/A floor measured
with the same instrument, in the same interleaving, on the same box under the same load.

    additivity.py --ab ab_union_512_whglx_c13.json --aa aa_floor_512_whglx_c21.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# The singles, each against the base it was actually taken on. Ratios, never seconds.
SINGLES = {
    "TT_BIO_DEVICE_CONDITIONING": (1.0333, "b2z2-host-residual-zero, whglx c6, 1.312 s of 40.667"),
    "TT_BIO_DEVICE_ZINIT": (1.0109, "b2z2-host-residual-round2, whglx c11, conditioning ALREADY on"),
    "TT_BIO_DEVICE_CONFIDENCE": (1.0128, "b2z2-conf-head-device, whglx c14, conditioning off"),
}


def summarise(path: Path) -> dict:
    d = json.loads(path.read_text())
    rows = d["ab"]["rows"]
    n = max(r["i"] for r in rows) + 1
    deltas, ratios = [], []
    for i in range(n):
        a = next(r for r in rows if r["i"] == i and r["arm"] == "A")
        b = next(r for r in rows if r["i"] == i and r["arm"] == "B")
        deltas.append(b["wall_s"] - a["wall_s"])
        ratios.append(b["wall_s"] / a["wall_s"])
    med = {arm: st.median([r["wall_s"] for r in rows if r["arm"] == arm]) for arm in "AB"}
    return {"file": path.name, "commit": d["env"].get("commit"), "card": d["env"].get("card"),
            "n_pairs": n, "median_A": med["A"], "median_B": med["B"],
            "ratio_median": med["B"] / med["A"], "ratio_paired_mean": st.mean(ratios),
            "paired_mean_s": st.mean(deltas), "paired_deltas_s": [round(x, 4) for x in deltas],
            "pairs_positive": sum(1 for x in deltas if x > 0),
            "cif_arms_differ": len({list(r["cif"].values())[0] for r in rows}) > 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", type=Path, required=True, nargs="+",
                    help="one or more A/B runs; pairs are pooled for the sign test")
    ap.add_argument("--aa", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    runs = [summarise(f) for f in a.ab]
    # Pooling is legitimate only for the SIGN: the runs share a fixture, a card and a commit, but
    # not the load the box happened to carry, so their seconds are not one sample and their
    # medians are not poolable. The sign of a paired delta is.
    pooled = [d for r in runs for d in r["paired_deltas_s"]]
    ab = dict(runs[0])
    ab["runs"] = [{"file": r["file"], "n_pairs": r["n_pairs"], "pairs_positive": r["pairs_positive"],
                   "paired_mean_s": round(r["paired_mean_s"], 4),
                   "ratio_median": round(r["ratio_median"], 5),
                   "paired_deltas_s": r["paired_deltas_s"]} for r in runs]
    ab["pooled_n_pairs"] = len(pooled)
    ab["pooled_pairs_positive"] = sum(1 for d in pooled if d > 0)
    ab["pooled_sign_p"] = 2.0 ** -len(pooled) * sum(
        __import__("math").comb(len(pooled), k)
        for k in range(ab["pooled_pairs_positive"], len(pooled) + 1))
    out = {"union": ab, "singles": {k: v[0] for k, v in SINGLES.items()}}
    product = 1.0
    for r, _ in SINGLES.values():
        product *= r
    out["product_of_singles"] = round(product, 5)
    out["discount_vs_product"] = round(ab["ratio_median"] - product, 5)
    if a.aa:
        aa = summarise(a.aa)
        out["aa_floor"] = aa
        # The floor is what a ratio of 1.0000 actually looks like on this box, both directions.
        out["aa_floor_ratio"] = round(aa["ratio_median"], 5)
        out["aa_floor_abs_pct"] = round(100 * abs(aa["ratio_median"] - 1.0), 4)
        out["union_outside_floor"] = abs(ab["ratio_median"] - 1.0) > abs(aa["ratio_median"] - 1.0)
        # The brief's pre-registered falsifier.
        out["competing_for_overlapped_host_time"] = (
            ab["ratio_median"] < product - abs(aa["ratio_median"] - 1.0))
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
