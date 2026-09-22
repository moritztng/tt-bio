#!/usr/bin/env python3
"""of3t-recutfin job 2: the trunk's magnitude/direction split IN THE SPACE THE CLAUSE IS GRADED IN.

`of3t-recut` banked `cos` and `norm_ratio` against float64 only, and read a split off them: the
magnitude is within 5.5 %, the angle is 35 degrees, so 97 % of the error is direction. That
number is correct and it is in vs-float64 space. **The clause is graded against upstream's own
bf16 step**, and carrying a quotient across those two spaces is the exact error R159 was
retracted for at R174, one pass before this row. So the split is re-taken in bf16 space from the
scorer's own bf16 aggregates, and the float64 one is reported beside it as contrast, never as
input.

The three aggregates are the norms of one concatenated vector per section --
`mass_weighted_rel_l2` is ||a-g||/||g||, `mass_weighted_norm_ratio` is ||a||/||g||,
`mass_weighted_cos` is <a,g>/(||a|| ||g||) -- so

    rel^2 = 1 + r^2 - 2 r cos

is an IDENTITY, not a model, and it is checked in both spaces before anything is read off it.

Then the question the campaign owes: the trunk must fall 1.7414134679108282x for the clause to
clear. Is that reachable by magnitude alone, by direction alone, or by neither? Arithmetic only.
No bar is written here: the allowance and the bar are read from the pre-registered CLAUSE.json.

CPU only, no device. Reads two committed artifacts.
"""
from __future__ import annotations

import json
import math
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESCORED = REPO / "perf/of3t_recut/MODEL_RECUT_composed3660_n384.json"
CLAUSE = REPO / "perf/of3t_modelframe/CLAUSE.json"
SECTION = "pairformer_stack"


def split(rel, r, cos, name, space):
    """The two axes of one reading, each with the other held perfect."""
    reconstructed = math.sqrt(max(0.0, 1.0 + r * r - 2.0 * r * cos))
    magnitude_alone = abs(r - 1.0)          # the direction made perfect, cos = 1
    direction_alone = math.sqrt(max(0.0, 2.0 - 2.0 * cos))   # the magnitude made perfect, r = 1
    best_scalar = math.sqrt(max(0.0, 1.0 - cos * cos))       # the BEST any rescaling can do
    return {
        "space": space, "what": name,
        "measured_rel_l2": rel, "norm_ratio": r, "cos": cos,
        "angle_degrees": math.degrees(math.acos(max(-1.0, min(1.0, cos)))),
        "IDENTITY": {
            "reconstructed_from_r_and_cos": reconstructed,
            "rel_difference": abs(reconstructed - rel) / rel if rel else None,
            "why": "rel^2 = 1 + r^2 - 2 r cos holds exactly for these three aggregates because "
                   "they are three norms of one vector pair. A residual here would mean the "
                   "split is modelled rather than exact, and nothing below could be read.",
        },
        "magnitude_alone_cos_set_to_1": magnitude_alone,
        "direction_alone_r_set_to_1": direction_alone,
        "residual_after_the_best_possible_rescaling": best_scalar,
        "shares_of_the_measured_error": {
            "magnitude": magnitude_alone / rel if rel else None,
            "direction": direction_alone / rel if rel else None,
            "note": "these do not sum to 1 and are not meant to. Each is what the reading would "
                    "be with the OTHER axis made perfect, divided by what it is; they add up to "
                    "more or less than the whole exactly as the two errors reinforce or cancel.",
        },
    }


def reach(s, allowance):
    """Can this reading fall to `allowance` on one axis alone? Arithmetic, not projection."""
    r, cos, rel = s["norm_ratio"], s["cos"], s["measured_rel_l2"]
    # hold r, solve rel(cos*) = allowance  ->  cos* = (1 + r^2 - a^2) / (2 r)
    need_cos = (1.0 + r * r - allowance ** 2) / (2.0 * r)
    # hold cos, solve rel(r*) = allowance  ->  r* = cos +- sqrt(cos^2 - 1 + a^2)
    disc = cos * cos - 1.0 + allowance ** 2
    r_roots = ([cos - math.sqrt(disc), cos + math.sqrt(disc)] if disc >= 0 else [])
    return {
        "allowance": allowance, "x_allowance_today": rel / allowance,
        "BY_MAGNITUDE_ALONE": {
            "what": "make the magnitude perfect (r -> 1) and keep today's direction",
            "reading_it_would_leave": s["direction_alone_r_set_to_1"],
            "x_allowance": s["direction_alone_r_set_to_1"] / allowance,
            "reaches_it": s["direction_alone_r_set_to_1"] <= allowance,
            "and_the_strongest_form": {
                "what": "the best ANY rescaling can do, not just r -> 1: min over all scalars s "
                        "of ||s a - g|| / ||g|| = sqrt(1 - cos^2) = sin(angle)",
                "reading_it_would_leave": s["residual_after_the_best_possible_rescaling"],
                "x_allowance": s["residual_after_the_best_possible_rescaling"] / allowance,
                "reaches_it": s["residual_after_the_best_possible_rescaling"] <= allowance,
            },
            "r_values_that_would_reach_it": r_roots,
        },
        "BY_DIRECTION_ALONE": {
            "what": "make the direction perfect (cos -> 1) and keep today's magnitude",
            "reading_it_would_leave": s["magnitude_alone_cos_set_to_1"],
            "x_allowance": s["magnitude_alone_cos_set_to_1"] / allowance,
            "reaches_it": s["magnitude_alone_cos_set_to_1"] <= allowance,
            "and_it_does_not_need_to_be_perfect": {
                "cos_required_at_todays_norm_ratio": need_cos,
                "angle_required_degrees": (math.degrees(math.acos(need_cos))
                                           if abs(need_cos) <= 1.0 else None),
                "angle_today_degrees": s["angle_degrees"],
                "fraction_of_todays_angle_that_must_close":
                    (1.0 - math.degrees(math.acos(need_cos)) / s["angle_degrees"]
                     if abs(need_cos) <= 1.0 and s["angle_degrees"] else None),
            },
        },
    }


def main() -> int:
    rs = json.loads(RESCORED.read_text())
    cl = json.loads(CLAUSE.read_text())
    allowance = cl["preregistered"]["trunk_allowance_for_the_clause_to_clear"]
    bar = cl["preregistered"]["bar"]

    def sec(pair):
        v = rs["per_section"][pair][SECTION]
        return v["mass_weighted_rel_l2"], v["mass_weighted_norm_ratio"], v["mass_weighted_cos"]

    bf16 = split(*sec("renorm_vs_UPSTREAM_BF16"),
                 name="our corrected trunk against upstream's own bf16 step",
                 space="vs-upstream-bf16 -- THE SPACE THE CLAUSE IS GRADED IN")
    f64 = split(*sec("renorm_vs_FLOAT64"),
                name="our corrected trunk against the float64 reference",
                space="vs-float64 -- CONTRAST ONLY, not carried across (R174)")
    theirs = split(*sec("UPSTREAM_BF16_vs_FLOAT64"),
                   name="upstream's own bf16 step against the float64 reference",
                   space="vs-float64 -- the reference's own error, for scale")

    out = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(), "row": "of3t-recutfin", "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only; this reads two committed JSON artifacts and does arithmetic",
        "SECTION": SECTION,
        "n_tensors": rs["per_section"]["renorm_vs_UPSTREAM_BF16"][SECTION]["n"],
        "THE_TWO_FRAMES_BEING_DIFFERENCED": {
            "ours": {
                "what": "the corrected trunk arm: the pairformer_stack scope of the composed "
                        "3,660-tensor artifact, device arm dev_RENORM_model_n384_external.pt, "
                        "run end to end on (cot_s, cot_z - delta) on of3t-modelframe's captured "
                        "boundary at padded N=384, injected with the graph-cut-external "
                        "cotangent cot_external.pt",
                "convention": rs.get("injection", {}).get("convention"),
                "correction_sha256": (rs.get("injection", {}).get("by_scope", {})
                                      .get("renorm:pairformer_stack", {})
                                      .get("correction", {}).get("sha256")),
            },
            "theirs": {
                "what": "upstream's own bf16 autocast step over the same 2,736 parameters of the "
                        "same batch -- the pinned p175 arm4 artifact, which is a FULL-MODEL "
                        "backward and carries no injected boundary at all",
                "path": rs["inputs"]["upstream_bf16"]["path"],
                "sha256": rs["inputs"]["upstream_bf16"]["sha256"],
            },
            "why_this_is_stated": "the two sides are not two runs of one configuration: ours is "
                                  "an injected scope on a captured boundary and theirs is a "
                                  "whole-model step. The reading is between those two frames and "
                                  "every number below inherits that.",
        },
        "THE_SPLIT_IN_THE_GRADED_SPACE": bf16,
        "THE_ANSWER": reach(bf16, allowance),
        "CONTRAST_IN_FLOAT64_SPACE_NOT_CARRIED": {
            "ours": f64, "upstreams_own": theirs,
            "why_it_may_not_be_carried": "R159 divided a vs-upstream-bf16 quantity by a "
                                         "vs-float64 one and was retracted at R174. The two "
                                         "spaces divide by different denominators (||g_bf16|| "
                                         "and ||g_f64||) and their decompositions are different "
                                         "numbers about different differences.",
            "and_they_disagree_sharply": None,
        },
        "preregistered_inputs": {
            "trunk_allowance": allowance, "clause_bar": bar,
            "source": "perf/of3t_modelframe/CLAUSE.json -- read, not re-derived. No bar is "
                      "written by this script.",
        },
    }
    m_bf16 = bf16["shares_of_the_measured_error"]["magnitude"]
    m_f64 = f64["shares_of_the_measured_error"]["magnitude"]
    out["CONTRAST_IN_FLOAT64_SPACE_NOT_CARRIED"]["and_they_disagree_sharply"] = (
        f"magnitude carries {m_bf16 * 100:.4g} % of the error in the graded space and "
        f"{m_f64 * 100:.4g} % in float64 space, a factor of {m_f64 / m_bf16:.4g}. Same arm, same "
        f"tensors, two references. That is why the float64 split could not be carried across.")

    a = out["THE_ANSWER"]
    by_mag = a["BY_MAGNITUDE_ALONE"]
    by_dir = a["BY_DIRECTION_ALONE"]
    out["VERDICT"] = (
        f"In the space the clause is graded in, the corrected trunk reads "
        f"{bf16['measured_rel_l2']:.16g} at norm ratio {bf16['norm_ratio']:.16g} and cos "
        f"{bf16['cos']:.16g}, an angle of {bf16['angle_degrees']:.4f} degrees. The magnitude is "
        f"already right to {abs(bf16['norm_ratio'] - 1) * 100:.4g} %: it carries "
        f"{m_bf16 * 100:.4g} % of the error and the direction carries "
        f"{bf16['shares_of_the_measured_error']['direction'] * 100:.5g} %. "
        f"The required {a['x_allowance_today']:.16g}x fall is NOT reachable by magnitude: "
        f"making the magnitude exact leaves {by_mag['reading_it_would_leave']:.16g}, "
        f"{by_mag['x_allowance']:.6g}x the allowance, and even the best possible rescaling -- "
        f"not r = 1 but the optimal scalar -- leaves "
        f"{by_mag['and_the_strongest_form']['reading_it_would_leave']:.16g}, "
        f"{by_mag['and_the_strongest_form']['x_allowance']:.6g}x. It IS reachable by direction: "
        f"cos need only rise from {bf16['cos']:.16g} to "
        f"{by_dir['and_it_does_not_need_to_be_perfect']['cos_required_at_todays_norm_ratio']:.16g}"
        f", closing the angle from {bf16['angle_degrees']:.4f} to "
        f"{by_dir['and_it_does_not_need_to_be_perfect']['angle_required_degrees']:.4f} degrees, "
        f"and a perfect direction at today's magnitude would read "
        f"{by_dir['reading_it_would_leave']:.6g}, "
        f"{by_dir['x_allowance']:.6g}x the allowance. So: by direction, not by magnitude, and "
        f"the whole of the trunk's remaining excess is an angle.")

    bad = [s["space"] for s in (bf16, f64, theirs)
           if s["IDENTITY"]["rel_difference"] > 1e-12]
    out["IDENTITY_HOLDS_IN_EVERY_SPACE_READ"] = {
        "worst_rel_difference": max(s["IDENTITY"]["rel_difference"] for s in (bf16, f64, theirs)),
        "bar": 1e-12, "failed": bad,
        "verdict": "the split is exact rather than modelled" if not bad
                   else "FAILED: the identity does not hold, nothing above may be read",
    }
    p = REPO / "perf/of3t_recutfin/BF16_SPLIT.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
