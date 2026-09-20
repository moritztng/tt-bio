#!/usr/bin/env python3
"""What the two named levers buy at 640 and 768, from the OWNER SPLIT rather than one exponent.

`of3t-orchestrator`'s bracket scaled the whole non-boundary term at N^2. That is one assumption
covering several objects with different scaling, and two of them are visibly not N^2: the
weights do not depend on the crop at all, and a term that is N*c_s does not grow like one that
is N^2*c_z. `split.py` separates the owners at a crop where the backward COMPLETES, so each can
be scaled on its own, and running two rungs turns the exponent itself into a measurement.

    per owner:  b(N) = b(384) * (N / 384) ** p        p fitted on the 256 and 384 rungs
    weights:    held constant, which the fit confirms rather than assumes

Two controls this has to clear, both cheap and both fatal if missed:

  THE OBSERVATIONAL FLOOR. `of3t-l1`'s 640 run reached 34.215 GB and still failed inside the
  FIRST checkpoint recompute, so any projection under that is refuted where it stands.

  THE SUM. The owner parts plus the residual are the census total by construction, so a
  projection that does not reproduce the measured rung it was fitted on has a bug, not a finding.

The verdict thresholds are the brief's, pre-registered before any number existed: recompute at
50 % or more of the non-boundary term means halving it plus the boundary spill closes 640; 30 to
50 % is marginal; under 30 % means the two named levers do not close 640 and the row's job is to
say what a third would have to save.

    project.py --rungs perf/of3t_crop640/out/split_{256,384}.json --out .../PROJECTION.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

GB = 1 << 30
CARD_GB = 34.22
FLOOR_640_GB = 34.215          # of3t-l1: where 640 DIED, a lower bound on its requirement
OWNERS = ("boundaries", "weights", "weight_grads", "tape", "cotangents", "recompute",
          "other_leaf", "unattributed")


def bnd_closed_form(rung):
    """The boundary term as a*N^2 + b*N, read off its own shape census.

    The set is small and fully enumerated -- four distinct shapes at every crop -- and its
    counts are the ARCHITECTURE's (48 pairformer blocks, one MSA module, one template stack),
    not the crop's. So the term does not need an exponent fitted to it: substitute the crop
    into the shapes and the answer is exact. The fit is kept beside it as a cross-check, and
    the two agreeing is what says the shape census is complete.
    """
    a = b = 0
    for e in rung["boundary_shapes"]:
        dims, n = e["shape"], e["n"]
        w = 2 if "BFLOAT16" in e["dtype"] else 4
        # Every crop-sized axis is the token axis; the rest are channels, heads or MSA depth.
        tok = sum(1 for d in dims if d == rung["tokens"])
        rest = 1
        for d in dims:
            if d != rung["tokens"]:
                rest *= d
        if tok == 2:
            a += n * rest * w
        elif tok == 1:
            b += n * rest * w
    return a, b


def read(p: Path):
    d = json.loads(p.read_text())
    b = d.get("backward") or {}
    w = b.get("walk")
    if not b.get("ok") or not w:
        sys.exit(f"{p}: backward did not complete with an attribution walk")
    parts = {o: w["by_owner"][o]["dram_b"] for o in OWNERS if o != "unattributed"}
    parts["unattributed"] = w["unattributed"]["dram_b"]
    return {"path": str(p), "tokens": int(d["argv"][d["argv"].index("--tokens") + 1]),
            "peak_b": b["dram_peak_b"], "walk_b": w["dram_now_b"],
            "walk_gap_pct": b.get("walk_gap_to_peak_pct"),
            "census_dram_b": w["census_dram_b"], "parts": parts,
            "boundary_pins": b.get("ckpt_pins"), "allocs": b.get("dram_live_allocs"),
            "boundary_shapes": w["shapes"]["boundaries"],
            "card": d["env"].get("tt_visible_devices"), "commit": d["env"].get("commit"),
            "branch": d["env"].get("branch"), "host": d["env"].get("host"),
            "aiclk": d["env"].get("aiclk_line")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rungs = sorted((read(p) for p in a.rungs), key=lambda r: r["tokens"])
    base = rungs[-1]
    n0 = base["tokens"]

    # Per-owner exponent from the two largest rungs. One rung leaves the exponent an assumption;
    # two make it a reading, and the reading is reported next to what it replaces.
    expo, fitted_on = {}, None
    if len(rungs) >= 2:
        lo, hi = rungs[-2], rungs[-1]
        fitted_on = [lo["tokens"], hi["tokens"]]
        for o in OWNERS:
            a_, b_ = lo["parts"][o], hi["parts"][o]
            expo[o] = (round(math.log(b_ / a_) / math.log(hi["tokens"] / lo["tokens"]), 3)
                       if a_ > 0 and b_ > 0 else None)

    bnd_a, bnd_b = bnd_closed_form(base)
    bnd_exact = lambda n: bnd_a * n * n + bnd_b * n

    def scale(o, n):
        """The closed form for the boundaries; the measured exponent elsewhere; N^2 as the
        fallback where a rung read zero and no exponent could be fitted."""
        if o == "boundaries":
            return bnd_exact(n)
        p = expo.get(o)
        if p is None:
            p = 2.0
        return base["parts"][o] * (n / n0) ** p

    def project(n, drop=()):
        return sum(scale(o, n) for o in OWNERS if o not in drop)

    # The non-boundary term is what the SECOND lever has to reach, so the split is quoted
    # against it and not against the peak.
    nonbnd = base["peak_b"] - base["parts"]["boundaries"]
    rec_share = 100 * base["parts"]["recompute"] / nonbnd if nonbnd else 0
    verdict = ("recompute >= 50 % of the non-boundary term" if rec_share >= 50 else
               "recompute 30-50 % of the non-boundary term" if rec_share >= 30 else
               "recompute < 30 % of the non-boundary term")

    out = {"card_GB": CARD_GB, "floor_640_GB": FLOOR_640_GB,
           "rungs": rungs, "base_tokens": n0,
           "owner_exponents": expo, "exponents_fitted_on": fitted_on,
           "boundary_closed_form": {
               "form": "a*N^2 + b*N, substituted into the shape census, not fitted",
               "a_b_per_token2": bnd_a, "b_b_per_token": bnd_b,
               "reproduces_base_rung_b": bnd_exact(n0),
               "measured_base_rung_b": base["parts"]["boundaries"],
               "rel_err_pct": round(100 * (bnd_exact(n0) - base["parts"]["boundaries"])
                                    / base["parts"]["boundaries"], 4),
               "cross_check_vs_fitted_exponent": expo.get("boundaries"),
               "shapes": base["boundary_shapes"]},
           "at_base": {
               "peak_GB": round(base["peak_b"] / GB, 3),
               "boundaries_GB": round(base["parts"]["boundaries"] / GB, 3),
               "boundary_share_of_peak_pct": round(100 * base["parts"]["boundaries"]
                                                   / base["peak_b"], 2),
               "non_boundary_GB": round(nonbnd / GB, 3),
               "recompute_GB": round(base["parts"]["recompute"] / GB, 3),
               "recompute_share_of_non_boundary_pct": round(rec_share, 2),
               "by_owner_GB": {o: round(base["parts"][o] / GB, 3) for o in OWNERS},
           },
           "pre_registered_branch": verdict, "projections": {}}

    for n in (640, 768):
        full = project(n)
        # Lever 1: spill the retained (s, z) boundaries to host.
        spill = scale("boundaries", n)
        # Lever 2: recompute a block in halves, which halves the LIVE recompute working set.
        half = scale("recompute", n) / 2
        both = full - spill - half
        row = {
            "projected_peak_GB": round(full / GB, 3),
            "deficit_GB": round((full - CARD_GB * GB) / GB, 3),
            "by_owner_GB": {o: round(scale(o, n) / GB, 3) for o in OWNERS},
            "lever_boundary_spill": {
                "saves_GB": round(spill / GB, 3),
                "share_of_deficit_pct": round(100 * spill / (full - CARD_GB * GB), 1)
                if full > CARD_GB * GB else None,
                "peak_after_GB": round((full - spill) / GB, 3),
                "fits": bool(full - spill <= CARD_GB * GB)},
            "lever_halve_recompute": {
                "saves_GB": round(half / GB, 3),
                "share_of_deficit_pct": round(100 * half / (full - CARD_GB * GB), 1)
                if full > CARD_GB * GB else None,
                "peak_after_GB": round((full - half) / GB, 3),
                "fits": bool(full - half <= CARD_GB * GB)},
            "both_levers": {
                "saves_GB": round((spill + half) / GB, 3),
                "peak_after_GB": round(both / GB, 3),
                "margin_GB": round(CARD_GB - both / GB, 3),
                "fits": bool(both <= CARD_GB * GB)},
        }
        if not row["both_levers"]["fits"]:
            # What a THIRD lever would have to save, and where the bytes are once the first two
            # have run. Naming the remainder's largest owner is the whole point of the split.
            rest = {o: scale(o, n) for o in OWNERS}
            rest["boundaries"] = 0.0
            rest["recompute"] = rest["recompute"] / 2
            row["third_lever"] = {
                "must_save_GB": round((both - CARD_GB * GB) / GB, 3),
                "largest_remaining_owners_GB": {
                    o: round(v / GB, 3) for o, v in
                    sorted(rest.items(), key=lambda kv: -kv[1])[:4]},
                "as_pct_of_largest_remaining": round(
                    100 * (both - CARD_GB * GB) / max(rest.values()), 1)}
        out["projections"][n] = row

    out["controls"] = {
        "reproduces_base_rung": {
            "projected_b": round(project(n0)), "measured_b": base["peak_b"],
            "rel_err_pct": round(100 * (project(n0) - base["peak_b"]) / base["peak_b"], 4)},
        "clears_640_observational_floor": bool(out["projections"][640]["projected_peak_GB"]
                                               >= FLOOR_640_GB),
        "640_floor_GB": FLOOR_640_GB,
        "walk_gap_to_peak_pct": {r["tokens"]: r["walk_gap_pct"] for r in rungs},
    }
    if not out["controls"]["clears_640_observational_floor"]:
        out["REFUTED"] = ("the projection at 640 is under 34.215 GB, which the run already "
                          "reached and still failed. The model is wrong, not the card.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))

    print("base rung %d tokens: peak %.3f GB, boundaries %.3f GB (%.1f %%), recompute %.3f GB"
          % (n0, base["peak_b"] / GB, base["parts"]["boundaries"] / GB,
             out["at_base"]["boundary_share_of_peak_pct"],
             base["parts"]["recompute"] / GB))
    print("recompute is %.1f %% of the non-boundary term -> %s"
          % (rec_share, out["pre_registered_branch"]))
    print("exponents:", {o: expo.get(o) for o in OWNERS})
    print("boundaries closed form %d*N^2 + %d*N reproduces the %d rung to %.4f %% (fit says "
          "exponent %s)" % (bnd_a, bnd_b, n0, out["boundary_closed_form"]["rel_err_pct"],
                            expo.get("boundaries")))
    for n in (640, 768):
        r = out["projections"][n]
        print("%d: peak %.2f GB, deficit %.2f GB | spill %.2f -> %.2f | halve recompute %.2f "
              "-> %.2f | BOTH -> %.2f GB, margin %+.2f GB, fits=%s"
              % (n, r["projected_peak_GB"], r["deficit_GB"],
                 r["lever_boundary_spill"]["saves_GB"], r["lever_boundary_spill"]["peak_after_GB"],
                 r["lever_halve_recompute"]["saves_GB"],
                 r["lever_halve_recompute"]["peak_after_GB"],
                 r["both_levers"]["peak_after_GB"], r["both_levers"]["margin_GB"],
                 r["both_levers"]["fits"]))
        if "third_lever" in r:
            print("     third lever must save %.2f GB; largest remaining: %s"
                  % (r["third_lever"]["must_save_GB"],
                     r["third_lever"]["largest_remaining_owners_GB"]))
    print("control: reproduces the %d rung to %.4f %%; clears the 640 floor: %s"
          % (n0, out["controls"]["reproduces_base_rung"]["rel_err_pct"],
             out["controls"]["clears_640_observational_floor"]))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
