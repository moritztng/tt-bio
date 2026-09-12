#!/usr/bin/env python3
"""Fold-level projection from the corrected per-op sweep, with its additivity discount.

Three rules, all of them there to stop the number being flattering:

  * an arm only counts if its instance's A/A floor is under `--aa-cap`. Seven instances ran while
    the box was contended and came back with A/A above 2; their arms say nothing and are dropped
    rather than averaged in.
  * an arm only counts if its ratio EXCEEDS its own A/A floor.
  * instances whose shipped call goes through a hand-tuned program config are excluded from the
    headline. The replay cannot reproduce that config, so its incumbent is ttnn's default
    resolver and the ratio is against something the fold never runs. They are reported as an
    optimistic upper bound instead.

The discount is `trimul-e6-fusion-returns-third-of-deleted-cost`: a deleted byte returns 63-73 %
of its modelled cost, so per-op wins summed across a block do not add.
"""
from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
# CONTEXT.md section 6, re-measured on current main by b2z-kernel-cycle-census.
FOLD_S, DEVICE_S = 22.3142, 18.6287


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "bench_manifest.json"))
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--resweep", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--aa-cap", type=float, default=1.20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    recs = {r["id"]: r for r in json.load(open(a.manifest))}
    base = {r["id"]: r for r in json.load(open(a.baseline))["rows"] if "ms_per_fold" in r}
    rsi = {i["id"]: i for i in json.load(open(a.resweep))["instances"]}
    old = {i["id"]: i for i in json.load(open(a.sweep))["instances"]}

    corr = {i: (rsi[i]["incumbent_us"] * recs[i]["calls_per_fold"] / 1000.0
                if i in rsi and rsi[i].get("incumbent_us") else r["ms_per_fold"])
            for i, r in base.items()}
    total = sum(corr.values())

    picks, excluded = [], []
    pool = list(rsi.items()) + [(k, v) for k, v in old.items() if recs[k]["kind"] != "matmul"]
    for iid, i in pool:
        aa = i.get("aa_floor")
        if aa is None or aa > a.aa_cap:
            continue
        cands = [x for x in i["arms"]
                 if "ratio" in x and x["knob"] != "nogrid=1" and x["ratio"] > aa]
        if not cands:
            continue
        b = max(cands, key=lambda x: x["ratio"])
        ms = corr[iid]
        row = {"id": iid, "kind": recs[iid]["kind"], "ms_per_fold": round(ms, 1),
               "knob": b["knob"], "ratio": b["ratio"], "aa_floor": aa,
               "saved_ms": round(ms * (1 - 1 / b["ratio"]), 1),
               "bit_exact": bool(b.get("bit_exact")),
               "rel_err": b.get("rel_max_err"), "rms_err": b.get("rms_err"),
               "incumbent_rel_err": i.get("incumbent_rel_max_err"),
               "incumbent_rms_err": i.get("incumbent_rms_err")}
        (excluded if i.get("shipped_config") == "program_config" else picks).append(row)

    def fold_ratio(saved_ms, disc):
        frac = saved_ms * disc / total
        return FOLD_S / (FOLD_S - DEVICE_S * frac)

    s_head = sum(r["saved_ms"] for r in picks)
    s_opt = s_head + sum(r["saved_ms"] for r in excluded)
    out = {
        "corrected_replayed_total_ms": round(total, 1),
        "fold_s": FOLD_S, "device_s": DEVICE_S, "aa_cap": a.aa_cap,
        "headline": {"saved_ms": round(s_head, 1),
                     "device_frac": round(s_head / total, 5),
                     "fold_x_undiscounted": round(fold_ratio(s_head, 1.0), 4),
                     "fold_x_at_0.73": round(fold_ratio(s_head, 0.73), 4),
                     "fold_x_at_0.63": round(fold_ratio(s_head, 0.63), 4)},
        "optimistic_incl_program_config": {
            "saved_ms": round(s_opt, 1),
            "fold_x_at_0.73": round(fold_ratio(s_opt, 0.73), 4)},
        "picks": sorted(picks, key=lambda r: -r["saved_ms"]),
        "excluded_program_config": sorted(excluded, key=lambda r: -r["saved_ms"]),
    }
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
