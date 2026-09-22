#!/usr/bin/env python3
"""The charter's one failing clause, evaluated on the frame-matched artifact, against the
thresholds this row pre-registered before the arm existed.

Two jobs, and the first is the control for the second.

CONTROL. `model_scope.py` composes the model headline per tensor over the union. The same number
is reachable from its own `per_section` table by the identity a reading of the form
||d|| / ||ref|| obeys: it composes with weights taken from the SAME reference it divides by, so
`total = sqrt(sum_s ref_sq_s * rel_s^2 / sum_s ref_sq_s)`. That recomposition is run against the
PUBLISHED artifact first. If it does not reproduce the published headline to rounding, this
script is wrong about the scorer's arithmetic and nothing below it may be read.

THE CLAUSE. `stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2 <= bars.A26_reachable_bar_vs_their_bf16`,
read off the rescored artifact -- the same producer, the same five other arms, the same three
pinned references, `pairformer_stack` taken from the arm on the model's own boundary.

THE MULTIPLE. The projection under test is the in-frame 2.2340903768268046, which is our trunk
against the LOCAL float64 reference over upstream's own bf16 against the same reference
(`perf/of3t_frame384/FRAME_N384.json`, `MATCHED.ratio_ours_over_floor`). The model boundary has
3.12x less trunk gradient mass, so the same ratio there is a measurement and not arithmetic.
This prints it, and prints the projection re-derived from the MEASURED multiple beside the
projection derived from the in-frame one, so a dead projection is visible rather than carried.

  clause.py --published <graded>.json [--rescored <new>.json] [--out <report>.json]
"""
from __future__ import annotations

import argparse
import json
import math
import socket
from pathlib import Path

SEC = "pairformer_stack"
IN_FRAME_MULTIPLE = 2.2340903768268046   # FRAME_N384.json MATCHED.ratio_ours_over_floor
LEVER_CEILING = 1.8563207917912123       # the pre-registered trunk lever ceiling divisor


def recompose(secs, override=None):
    """The scorer's own composition identity, over one arm's per-section table."""
    over = override or {}
    num = sum(v["ref_sq"] * (over.get(s, v["mass_weighted_rel_l2"]) or 0.0) ** 2
              for s, v in secs.items())
    den = sum(v["ref_sq"] for v in secs.values())
    return math.sqrt(num / den) if den else None


def solve(secs, target):
    """The largest the trunk section may read with every other section held, for the composed
    headline to equal `target`. None if no value of it can reach the target."""
    den = sum(v["ref_sq"] for v in secs.values())
    rest = sum(v["ref_sq"] * (v["mass_weighted_rel_l2"] or 0.0) ** 2
               for s, v in secs.items() if s != SEC)
    x2 = (target ** 2 * den - rest) / secs[SEC]["ref_sq"]
    return math.sqrt(x2) if x2 > 0 else None


def read(path):
    d = json.loads(Path(path).read_text())
    return {
        "path": str(path),
        "host_of_the_scoring_run": d.get("host"),
        "arms": d.get("arms"),
        "bar": d["bars"]["A26_reachable_bar_vs_their_bf16"],
        "headline": d["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"],
        "headline_vs_f64": d["stats"]["renorm_vs_FLOAT64"]["mass_weighted_rel_l2"],
        "secs_bf16": d["per_section"]["renorm_vs_UPSTREAM_BF16"],
        "secs_f64": d["per_section"]["renorm_vs_FLOAT64"],
        "secs_floor": d["per_section"]["UPSTREAM_BF16_vs_FLOAT64"],
        "model_sq": d["model_squared_gradient_norm_measured"],
    }


def trunk_view(a):
    """Everything about the trunk section that the clause depends on."""
    bf16, f64, floor = a["secs_bf16"][SEC], a["secs_f64"][SEC], a["secs_floor"][SEC]
    den = sum(v["ref_sq"] for v in a["secs_bf16"].values())
    rest = sum(v["ref_sq"] * (v["mass_weighted_rel_l2"] or 0.0) ** 2
               for s, v in a["secs_bf16"].items() if s != SEC)
    trunk_err = bf16["ref_sq"] * bf16["mass_weighted_rel_l2"] ** 2
    r = floor["mass_weighted_norm_ratio"]
    return {
        "reading_vs_their_bf16": bf16["mass_weighted_rel_l2"],
        "reading_vs_float64": f64["mass_weighted_rel_l2"],
        "cos_vs_float64": f64["mass_weighted_cos"],
        "norm_ratio_vs_float64": f64["mass_weighted_norm_ratio"],
        "upstreams_own_floor_here_vs_float64": floor["mass_weighted_rel_l2"],
        "upstreams_own_norm_ratio_here": r,
        "section_A26_level": math.sqrt(2.0) * floor["mass_weighted_rel_l2"] / r,
        "multiple_of_upstreams_own_bf16": (f64["mass_weighted_rel_l2"]
                                           / floor["mass_weighted_rel_l2"]),
        "pct_of_model_mass": bf16["pct_of_model_mass"],
        "share_of_the_models_error_mass": trunk_err / (trunk_err + rest),
        "worst_tensor": bf16["worst_tensor"], "worst_rel_l2": bf16["worst_rel_l2"],
        "n_over_per_tensor_bar": bf16["n_over_per_tensor_bar"], "n": bf16["n"],
        "_den": den, "_rest": rest,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", required=True, type=Path)
    ap.add_argument("--rescored", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    pub = read(a.published)
    ctrl = recompose(pub["secs_bf16"])
    control = {
        "what": "the scorer's composition identity reproduced against the published headline, "
                "before any rescored number is read",
        "published_headline": pub["headline"], "recomposed": ctrl,
        "rel_difference": abs(ctrl - pub["headline"]) / pub["headline"],
    }
    if control["rel_difference"] > 1e-12:
        raise SystemExit(f"the recomposition reads {ctrl!r} against the published "
                         f"{pub['headline']!r} -- this script is wrong about the scorer's "
                         f"arithmetic and nothing below it may be read")

    pt = trunk_view(pub)
    bar = pub["bar"]
    allowance = solve(pub["secs_bf16"], bar)
    prereg = {
        "trunk_allowance_for_the_clause_to_clear": allowance,
        "bar": bar,
        "levels": {
            "bit_exact_float64_trunk": recompose(pub["secs_bf16"], {SEC: 0.0}),
            "upstreams_own_floor_here": recompose(
                pub["secs_bf16"], {SEC: pt["upstreams_own_floor_here_vs_float64"]}),
            "section_A26_level": recompose(pub["secs_bf16"], {SEC: pt["section_A26_level"]}),
            "projection_from_the_in_frame_multiple": recompose(
                pub["secs_bf16"],
                {SEC: IN_FRAME_MULTIPLE * pt["section_A26_level"] / math.sqrt(2.0)}),
            "the_frame_mismatch_alone": recompose(pub["secs_bf16"], {SEC: 1.8416532191335382}),
        },
        "projected_trunk_reading": IN_FRAME_MULTIPLE * pt["section_A26_level"] / math.sqrt(2.0),
        "projected_trunk_reading_with_the_lever_ceiling":
            IN_FRAME_MULTIPLE * pt["section_A26_level"] / math.sqrt(2.0) / LEVER_CEILING,
        "in_frame_multiple_under_test": IN_FRAME_MULTIPLE,
    }
    for k, v in prereg["levels"].items():
        prereg["levels"][k] = {"clause_value": v, "x_bar": v / bar, "clears": v <= bar}

    out = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(),
        "control": control,
        "published": {"artifact": pub["path"], "headline": pub["headline"],
                      "x_bar": pub["headline"] / bar, "bar": bar,
                      "arms": pub["arms"], "trunk": pt},
        "preregistered": prereg,
    }

    if a.rescored:
        new = read(a.rescored)
        nt = trunk_view(new)
        nctrl = recompose(new["secs_bf16"])
        measured_multiple = nt["multiple_of_upstreams_own_bf16"]
        out["frame_matched"] = {
            "artifact": new["path"],
            "arms": new["arms"],
            "recomposition_check": {"headline": new["headline"], "recomposed": nctrl,
                                    "rel_difference": abs(nctrl - new["headline"])
                                    / new["headline"]},
            "CLAUSE": {
                "value": new["headline"], "bar": new["bar"],
                "x_bar": new["headline"] / new["bar"],
                "clears": new["headline"] <= new["bar"],
            },
            "trunk": nt,
            "MULTIPLE": {
                "measured_on_this_boundary": measured_multiple,
                "in_frame_projection_was": IN_FRAME_MULTIPLE,
                "ratio_measured_over_projected": measured_multiple / IN_FRAME_MULTIPLE,
                "projection_alive": abs(measured_multiple / IN_FRAME_MULTIPLE - 1.0) <= 0.15,
                "what_near_means": "within 15 % of 2.2340903768268046. Outside that the "
                                   "projection is dead and the campaign must stop carrying it.",
            },
            "against_the_preregistration": {
                "trunk_read": nt["reading_vs_their_bf16"],
                "trunk_allowance": allowance,
                "x_allowance": (nt["reading_vs_their_bf16"] / allowance
                                if allowance else None),
                "projected_trunk_reading": prereg["projected_trunk_reading"],
                "measured_over_projected_trunk":
                    nt["reading_vs_their_bf16"] / prereg["projected_trunk_reading"],
            },
            "MOVED": {s: {"published": pub["secs_bf16"][s]["mass_weighted_rel_l2"],
                          "frame_matched": new["secs_bf16"][s]["mass_weighted_rel_l2"]}
                      for s in sorted(new["secs_bf16"])
                      if abs((new["secs_bf16"][s]["mass_weighted_rel_l2"] or 0)
                             - (pub["secs_bf16"].get(s, {}).get("mass_weighted_rel_l2") or 0))
                      > 1e-15},
        }

    txt = json.dumps(out, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
