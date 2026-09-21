#!/usr/bin/env python3
"""Re-fit on 384->512 and project 768, with the two things the 512 rung forced.

`of3t-crop640`'s `project.py` is reused for the parts that still hold -- `read()` and the
boundary closed form come from it by import, not by copy -- but two of its assumptions did not
survive this rung and are corrected here:

  THE CARD IS 34.2255 GB, NOT 34.22 GiB. `project.py` divides bytes by 2**30 and compares the
  result to 34.22, while the figure it inherited from `of3t-l1` ("the 640 run reached 34.215 GB
  before it died") is bytes/1e9: `of3t-memory` records this p300c as "34.226 GB DRAM across 8
  banks", and at the 512 refusal the allocator itself reports 8 banks of 4,278,190,016 B =
  34,225,520,128 B. A GiB bar quoted against a GB measurement prices every lever against a card
  7.34 % larger than the one in the box. Everything below is in BYTES, reported as decimal GB.

  THE 512 PEAK IS CENSORED. The 512 backward did not complete: it was refused a 6,160,384 B
  DRAM buffer with 6,957,568 B free, 100.4 s in, still inside the first recompute. So its peak,
  34,218,562,560 B, is a LOWER BOUND on what 512 requires, and every exponent fitted to it and
  every projection built on it is a lower bound too. That is stated on each figure rather than
  rounded away, because a lower bound that excludes 768 excludes it regardless of how loose it
  is -- and that is the whole decision.

  `of3t-l1`'s 34.215 GB is ALSO the card, not a property of 640. 512 died at 34.2186 GB and 640
  died at 34.2157 GB: any crop that overflows reports the same number, so "clears the 640
  observational floor" tests that a projection exceeds the card, which every projection of an
  OOM does. It is kept as a control and read for what it is.

    project512.py --rungs perf/of3t_crop640/out/split_{256,384}.json \
                          perf/of3t_crop512/out/split_512.json \
                  --out perf/of3t_crop512/out/PROJECTION_512.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.of3t_crop640.project import bnd_closed_form                   # noqa: E402

G = 1e9
# The allocator's own figure at the 512 refusal: 8 banks x 4,278,190,016 B. Not a spec sheet.
CARD_B = 8 * 4_278_190_016
OWNERS = ("recompute", "boundaries", "tape", "unattributed", "weights", "cotangents",
          "weight_grads", "other_leaf")
# Owners the crop does not move. `weights` is weight-shaped; `unattributed` is the trunk's raw
# non-autograd handles and it FELL from 384 to 512 (1.622 -> 1.402 GB), so it has no exponent to
# fit -- holding it at the base rung is the conservative reading and it is named as an
# assumption rather than hidden in a fitted -0.506.
FLAT = ("weights", "unattributed", "other_leaf", "weight_grads")


def read_rung(p: Path):
    d = json.loads(p.read_text())
    b, w = d["backward"], d["backward"]["walk"]
    parts = {o: w["by_owner"][o]["dram_b"] for o in w["by_owner"]}
    parts["unattributed"] = w["unattributed"]["dram_b"]
    return {"path": str(p), "tokens": int(d["argv"][d["argv"].index("--tokens") + 1]),
            "peak_b": b["dram_peak_b"], "completed": bool(b.get("ok")),
            "censored": not b.get("ok"), "allocs": b.get("dram_live_allocs"),
            "walk_gap_b": b.get("walk_gap_to_peak_b"), "walks": b.get("walks"),
            "forward_peak_b": d["forward"]["dram_peak_b"], "parts": parts,
            "pins": b.get("ckpt_pins"),
            "boundaries_after_forward_b": d["after_forward"]["by_owner"]["boundaries"]["dram_b"],
            "boundary_shapes": d["after_forward"]["shapes"]["boundaries"],
            "error_head": b.get("error_head"), "free_at_failure": b.get("free"),
            "card": d["env"].get("tt_visible_devices"), "host": d["env"].get("host"),
            "branch": d["env"].get("branch"), "commit": d["env"].get("commit"),
            "aiclk": d["env"].get("aiclk_line")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rungs = sorted((read_rung(p) for p in a.rungs), key=lambda r: r["tokens"])
    by_n = {r["tokens"]: r for r in rungs}
    base = by_n[512]
    lo = by_n[384]

    # The boundary term keeps its closed form, built on 256 where the token axis does not
    # collide with c_s = 384. The 512 rung is its third test and its first out-of-sample one.
    bnd_a, bnd_b = bnd_closed_form(by_n[256] | {"tokens": 256})
    bnd = lambda n: bnd_a * n * n + bnd_b * n

    expo = {}
    for o in OWNERS:
        x, y = lo["parts"].get(o, 0), base["parts"].get(o, 0)
        expo[o] = (round(math.log(y / x) / math.log(512 / 384), 3)) if x > 0 and y > 0 else None
    tot_p = {f"{x}->{y}": round(math.log(by_n[y]["peak_b"] / by_n[x]["peak_b"])
                                / math.log(y / x), 4) for x, y in ((256, 384), (384, 512))}

    def scale(o, n):
        if o == "boundaries":
            return float(bnd(n))
        if o in FLAT or not expo.get(o):
            return float(base["parts"].get(o, 0))
        return base["parts"][o] * (n / 512) ** expo[o]

    out = {"card_b": CARD_B, "card_GB_dec": round(CARD_B / G, 4),
           "note": "every peak, exponent and projection below that involves the 512 rung is a "
                   "LOWER BOUND: the 512 backward was refused by the card and did not complete",
           "rungs": rungs, "fitted_on": [384, 512], "owner_exponents_lower_bound": expo,
           "total_peak_exponents": tot_p,
           "boundary_closed_form": {
               "form": f"{bnd_a}*N^2 + {bnd_b}*N", "built_on_tokens": 256,
               "predicts": {n: bnd(n) for n in (256, 384, 512, 640, 768)},
               "measured_after_forward": {r["tokens"]: r["boundaries_after_forward_b"]
                                          for r in rungs},
               "exact_at_every_rung": all(bnd(r["tokens"]) == r["boundaries_after_forward_b"]
                                          for r in rungs)},
           "projections": {}}

    for n in (640, 768):
        parts = {o: scale(o, n) for o in OWNERS}
        full = sum(parts.values())
        spill, half = parts["boundaries"], parts["recompute"] / 2
        both = full - spill - half
        row = {"projected_peak_b": round(full), "projected_peak_GB": round(full / G, 3),
               "by_owner_GB": {o: round(v / G, 3) for o, v in
                               sorted(parts.items(), key=lambda kv: -kv[1])},
               "spill_only_GB": round((full - spill) / G, 3),
               "halve_recompute_only_GB": round((full - half) / G, 3),
               "both_levers_GB": round(both / G, 3),
               "margin_GB": round((CARD_B - both) / G, 3),
               "fits": bool(both <= CARD_B)}
        if not row["fits"]:
            rest = {o: v for o, v in parts.items()}
            rest["boundaries"] = 0.0
            rest["recompute"] = rest["recompute"] / 2
            row["third_lever"] = {
                "must_save_GB": round((both - CARD_B) / G, 3),
                "where_the_bytes_are_GB": {o: round(v / G, 3) for o, v in
                                           sorted(rest.items(), key=lambda kv: -kv[1])[:5]},
                "as_pct_of_the_halved_recompute": round(100 * (both - CARD_B) / rest["recompute"], 1),
                "recompute_budget_b": round(CARD_B - sum(v for o, v in rest.items()
                                                         if o != "recompute")),
                "recompute_must_fall_to_pct_of_projection": round(
                    100 * (CARD_B - sum(v for o, v in rest.items() if o != "recompute"))
                    / parts["recompute"], 1)}
        out["projections"][n] = row

    # CONTROLS.
    out["controls"] = {
        "boundary_form_out_of_sample_at_512": {
            "predicted_b": bnd(512), "measured_b": base["boundaries_after_forward_b"],
            "exact": bnd(512) == base["boundaries_after_forward_b"]},
        "the_256_384_fit_projected_onto_512": {
            "predicted_b": round(lo["peak_b"] * (512 / 384) ** tot_p["256->384"]),
            "measured_lower_bound_b": base["peak_b"],
            "under_predicts_by_at_least_pct": round(
                100 * (base["peak_b"] - lo["peak_b"] * (512 / 384) ** tot_p["256->384"])
                / base["peak_b"], 3)},
        "structural_N2_projected_onto_512": {
            "predicted_b": round(lo["peak_b"] * (512 / 384) ** 2),
            "measured_lower_bound_b": base["peak_b"],
            "under_predicts_by_at_least_pct": round(
                100 * (base["peak_b"] - lo["peak_b"] * (512 / 384) ** 2) / base["peak_b"], 3),
            "verdict": "REFUTED at one rung: N^2 puts 512 below the card and 512 filled it"},
        "the_640_observational_floor_is_the_card": {
            "640_died_at_b": 34_215_730_688, "512_died_at_b": base["peak_b"],
            "card_b": CARD_B,
            "reading": "both are the card, so the floor tests that a projection exceeds the "
                       "card and cannot discriminate between projections that do"},
        "walk_gap_to_peak_b": {r["tokens"]: r["walk_gap_b"] for r in rungs},
        "regime_discriminator_boundaries_intact_at_the_peak": {
            r["tokens"]: {"at_peak_b": r["parts"]["boundaries"],
                          "after_forward_b": r["boundaries_after_forward_b"],
                          "intact_pct": round(100 * r["parts"]["boundaries"]
                                              / r["boundaries_after_forward_b"], 2)}
            for r in rungs},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))

    print(f"card {CARD_B:,} B = {CARD_B/G:.4f} GB (8 banks x 4,278,190,016 B, the allocator's own)")
    print("total-peak exponents:", tot_p, "  (384->512 is a LOWER BOUND)")
    print("owner exponents 384->512 (lower bounds):",
          {o: expo[o] for o in OWNERS if expo[o] is not None})
    for n in (640, 768):
        r = out["projections"][n]
        print(f"{n}: peak >= {r['projected_peak_GB']:8.3f} GB | spill -> "
              f"{r['spill_only_GB']:8.3f} | halve -> {r['halve_recompute_only_GB']:8.3f} | "
              f"BOTH -> {r['both_levers_GB']:8.3f} GB, margin {r['margin_GB']:+8.3f} GB, "
              f"fits={r['fits']}")
        print(f"     owners: {r['by_owner_GB']}")
        if "third_lever" in r:
            t = r["third_lever"]
            print(f"     third lever must save >= {t['must_save_GB']} GB "
                  f"({t['as_pct_of_the_halved_recompute']} % of the halved recompute); "
                  f"recompute must fall to {t['recompute_must_fall_to_pct_of_projection']} % "
                  f"of its projection")
            print(f"     where the bytes are: {t['where_the_bytes_are_GB']}")
    c = out["controls"]
    print("control boundary form out-of-sample at 512:", c["boundary_form_out_of_sample_at_512"])
    print("control 256->384 fit onto 512:", c["the_256_384_fit_projected_onto_512"])
    print("control structural N^2 onto 512:", c["structural_N2_projected_onto_512"])
    print("regime (boundaries intact at the peak):",
          c["regime_discriminator_boundaries_intact_at_the_peak"])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
