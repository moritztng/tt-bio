#!/usr/bin/env python3
"""Fold-level projection from the per-op sweep, split by whether the lever changes numerics.

Reads `renoise_all_wh_c1.json`, the paired-repeat sweep, NOT the earlier single-bracket one. The
single-bracket estimator scored an arm against the median of the two incumbent runs bracketing it,
which a host-load excursion lasting longer than one bracket walks straight through: it produced
seven A/A floors between 2.16 and 3.31 and seven wins up to 2.97x, all of which vanished when the
same instances were re-run as independent (incumbent, arm) pairs. Every ratio here is the median of
three such pairs, and the A/A control is the same estimator with the arm replaced by another
incumbent run, so the control breaks exactly what the measurement reads.

The split that matters is bit-exactness, not size:

  * `outbuf=L1` is pure placement -- same kernel, same accumulation order, different destination
    buffer -- and `l1_bitexact.py` confirms all ten winning instances are bit-exact. It needs no
    accuracy argument.
  * the fidelity and `fp32_dest_acc_en` arms all change numerics and belong behind the
    cdk2x2_298 control.

The discount is `trimul-e6-fusion-returns-third-of-deleted-cost`: a deleted byte returns 63-73 % of
its modelled cost, so per-op wins summed across a block do not add.
"""
from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
# CONTEXT.md section 6, re-measured on current main by b2z-kernel-cycle-census.
FOLD_S, DEVICE_S = 22.3142, 18.6287
PLACEMENT = {"outbuf=L1"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "bench_manifest.json"))
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--resweep", required=True, help="for the shipped incumbent us/call")
    ap.add_argument("--renoise", required=True, help="the paired-repeat ratios")
    ap.add_argument("--bitexact", default="")
    ap.add_argument("--min-ratio", type=float, default=1.03,
                    help="an arm must beat this to count; the A/A controls all land under 1.02")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    recs = {r["id"]: r for r in json.load(open(a.manifest))}
    base = {r["id"]: r for r in json.load(open(a.baseline))["rows"] if "ms_per_fold" in r}
    rsi = {i["id"]: i for i in json.load(open(a.resweep))["instances"]}
    rn = {i["id"]: i for i in json.load(open(a.renoise))["instances"]}
    bx = {r["id"]: r for r in json.load(open(a.bitexact))["rows"]} if a.bitexact else {}

    corr = {i: (rsi[i]["incumbent_us"] * recs[i]["calls_per_fold"] / 1000.0
                if i in rsi and rsi[i].get("incumbent_us") else r["ms_per_fold"])
            for i, r in base.items()}
    total = sum(corr.values())

    place, numer, aa_ctrl = [], [], []
    for iid, i in rn.items():
        ms = corr[iid]
        arms = {x["raw_knob"]: x for x in i["arms"]}
        if "" in arms and arms[""].get("ratio"):
            aa_ctrl.append(arms[""]["ratio"])

        def pick(keys):
            c = [(k, v["ratio"]) for k, v in arms.items()
                 if k in keys and v.get("ratio", 0) >= a.min_ratio]
            return max(c, key=lambda x: x[1]) if c else (None, None)

        k, r = pick(PLACEMENT)
        if k:
            place.append({"id": iid, "ms_per_fold": round(ms, 1), "knob": k, "ratio": r,
                          "saved_ms": round(ms * (1 - 1 / r), 1),
                          "shipped_out": recs[iid]["out_mem"]["buffer"],
                          "bit_exact": bx.get(iid, {}).get("bit_exact")})
        nk = {x for x in arms if x and x not in PLACEMENT and not x.startswith("grid")
              and x != "nogrid=1"}
        k2, r2 = pick(nk)
        if k2:
            numer.append({"id": iid, "ms_per_fold": round(ms, 1), "knob": k2, "ratio": r2,
                          "saved_ms": round(ms * (1 - 1 / r2), 1)})

    def fold(saved, disc):
        return FOLD_S / (FOLD_S - DEVICE_S * (saved * disc / total))

    sp = sum(r["saved_ms"] for r in place)
    sn = sum(r["saved_ms"] for r in numer)
    out = {
        "replayed_total_ms": round(total, 1), "fold_s": FOLD_S, "device_s": DEVICE_S,
        "aa_control": {"n": len(aa_ctrl), "min": min(aa_ctrl), "max": max(aa_ctrl)},
        "placement_bit_exact": {
            "saved_ms": round(sp, 1), "device_frac": round(sp / total, 5),
            "fold_x_undiscounted": round(fold(sp, 1.0), 4),
            "fold_x_at_0.73": round(fold(sp, 0.73), 4),
            "all_bit_exact": all(r["bit_exact"] for r in place) if bx else None},
        "numerics_changing": {
            "saved_ms": round(sn, 1), "fold_x_at_0.73": round(fold(sn, 0.73), 4)},
        "both": {"saved_ms": round(sp + sn, 1),
                 "fold_x_undiscounted": round(fold(sp + sn, 1.0), 4),
                 "fold_x_at_0.73": round(fold(sp + sn, 0.73), 4)},
        "placement_rows": sorted(place, key=lambda r: -r["saved_ms"]),
        "numerics_rows": sorted(numer, key=lambda r: -r["saved_ms"]),
    }
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_rows")}, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
