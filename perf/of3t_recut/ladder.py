#!/usr/bin/env python3
"""of3t-recut job 4: where the corrected trunk reading lands on of3t-modelframe's ladder.

The five levels were pre-registered in `perf/of3t_modelframe/CLAUSE.json` before this arm
existed and NOTHING here re-derives one. This script does three things:

  1. reproduces each level's clause value by substituting that level's trunk reading into the
     RESCORED artifact's own section table. The levels were computed on the published table, so
     if they come back unchanged the ladder applies to this artifact and is not being bent to it;
  2. places the corrected trunk reading among them;
  3. reads the verdict off that placement.

It also reports what the pre-registered trunk lever ceiling would do to the corrected reading.
That ceiling is a projection this campaign already carries, not a measured lever, and it is
labelled as such.
"""
from __future__ import annotations

import json
import math
import socket
from pathlib import Path

SEC = "pairformer_stack"
CLAUSE = Path("perf/of3t_recut/CLAUSE_RECUT.json")
RESCORED = Path("perf/of3t_recut/MODEL_RECUT_composed3660_n384.json")
PUBLISHED_LADDER = Path("perf/of3t_modelframe/CLAUSE.json")
LEVER_CEILING = 1.8563207917912123          # clause.py's pre-registered trunk lever divisor
IN_FRAME_MULTIPLE = 2.2340903768268046      # clause.py's, FRAME_N384 ratio_ours_over_floor


def recompose(secs, override=None):
    over = override or {}
    num = sum(v["ref_sq"] * (over.get(s, v["mass_weighted_rel_l2"]) or 0.0) ** 2
              for s, v in secs.items())
    den = sum(v["ref_sq"] for v in secs.values())
    return math.sqrt(num / den) if den else None


def main() -> int:
    cl = json.loads(CLAUSE.read_text())
    rs = json.loads(RESCORED.read_text())
    secs = rs["per_section"]["renorm_vs_UPSTREAM_BF16"]
    bar = cl["preregistered"]["bar"]
    allowance = cl["preregistered"]["trunk_allowance_for_the_clause_to_clear"]
    fm = cl["frame_matched"]
    t = fm["trunk"]
    measured = t["reading_vs_their_bf16"]

    # the trunk reading each pre-registered level stands for, from clause.py's own definitions
    readings = {
        "bit_exact_float64_trunk": 0.0,
        "upstreams_own_floor_here": t["upstreams_own_floor_here_vs_float64"],
        "section_A26_level": t["section_A26_level"],
        "projection_from_the_in_frame_multiple":
            IN_FRAME_MULTIPLE * t["section_A26_level"] / math.sqrt(2.0),
        "the_frame_mismatch_alone": 1.8416532191335382,
    }
    prereg = json.loads(PUBLISHED_LADDER.read_text())["preregistered"]["levels"]

    levels = {}
    for k, r in readings.items():
        v = recompose(secs, {SEC: r})
        levels[k] = {
            "trunk_reading_it_stands_for": r,
            "clause_value_reproduced_here": v,
            "clause_value_as_preregistered": prereg[k]["clause_value"],
            "rel_difference": abs(v - prereg[k]["clause_value"]) / prereg[k]["clause_value"],
            "x_bar": v / bar,
            "clears": v <= bar,
            "corrected_trunk_is_at_or_below_it": measured <= r,
        }
    worst = max(x["rel_difference"] for x in levels.values())

    ctrl = recompose(secs)
    out = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(), "row": "of3t-recut", "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only; this reads two committed artifacts",
        "LADDER_APPLIES_UNCHANGED": {
            "what": "each pre-registered level, re-derived on the RESCORED artifact's own "
                    "section table. Only pairformer_stack moved between the two artifacts, so "
                    "the levels must come back unchanged -- if they do not, the ladder is being "
                    "read on a different composition than the one it was registered on",
            "worst_rel_difference_over_the_five_levels": worst,
            "verdict": ("the ladder applies unchanged" if worst < 1e-12 else
                        "the ladder does NOT reproduce on this artifact"),
        },
        "RECOMPOSITION_CONTROL": {
            "headline": rs["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"],
            "recomposed": ctrl,
            "rel_difference": abs(ctrl - rs["stats"]["renorm_vs_UPSTREAM_BF16"]
                                  ["mass_weighted_rel_l2"])
            / rs["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"],
        },
        "CORRECTED": {
            "trunk_vs_upstream_bf16": measured,
            "trunk_vs_float64": t["reading_vs_float64"],
            "multiple_of_upstreams_own_bf16": t["multiple_of_upstreams_own_bf16"],
            "cos_vs_float64": t["cos_vs_float64"],
            "norm_ratio_vs_float64": t["norm_ratio_vs_float64"],
            "clause": fm["CLAUSE"],
            "trunk_allowance": allowance,
            "x_allowance": measured / allowance,
            "share_of_the_models_error_mass": t["share_of_the_models_error_mass"],
        },
        "ON_THE_DOUBLE_COUNTED_FUNCTIONAL_IT_READ": {
            "trunk_vs_upstream_bf16": 0.9969599833682794,
            "trunk_vs_float64": 0.9349175217825587,
            "multiple_of_upstreams_own_bf16": 2.970162431380236,
            "clause_x_bar": 1.7814428090278143,
            "source": "perf/of3t_modelframe/CLAUSE.json, withdrawn at of3t-frameself pass 389 "
                      "and explained at of3t-orchestrator pass 395",
        },
        "LEVELS": levels,
        "MULTIPLE": fm["MULTIPLE"],
        "THE_PREREGISTERED_LEVER_CEILING": {
            "divisor": LEVER_CEILING,
            "what": "clause.py carries it as `projected_trunk_reading_with_the_lever_ceiling`. "
                    "It is a projection this campaign already registered, NOT a measured lever, "
                    "and no arm here exercises it",
            "corrected_trunk_under_it": measured / LEVER_CEILING,
            "clause_there": recompose(secs, {SEC: measured / LEVER_CEILING}),
            "x_bar_there": recompose(secs, {SEC: measured / LEVER_CEILING}) / bar,
            "clears_there": recompose(secs, {SEC: measured / LEVER_CEILING}) <= bar,
        },
    }

    cleared = [k for k, v in levels.items() if v["clears"] and measured <= v[
        "trunk_reading_it_stands_for"]]
    was = 1.7814428090278143
    now = fm["CLAUSE"]["x_bar"]
    out["WHAT_THE_REPAIR_MOVED"] = {
        "clause_x_bar": {"double_counted": was, "corrected": now},
        "excess_over_the_bar": {"double_counted": was - 1.0, "corrected": now - 1.0,
                                "fraction_removed": (was - now) / (was - 1.0)},
        "trunk_vs_upstream_bf16": {"double_counted": 0.9969599833682794, "corrected": measured,
                                   "fraction_removed": 1.0 - measured / 0.9969599833682794},
        "trunk_vs_float64": {"double_counted": 0.9349175217825587,
                             "corrected": t["reading_vs_float64"],
                             "fraction_removed":
                                 1.0 - t["reading_vs_float64"] / 0.9349175217825587},
    }
    out["VERDICT"] = {
        "clause_clears": fm["CLAUSE"]["clears"],
        "x_bar": fm["CLAUSE"]["x_bar"],
        "highest_clearing_level_the_corrected_trunk_reaches": cleared[-1] if cleared else None,
        "reading": (
            f"the corrected trunk reads {measured:.16g} against upstream's own bf16, which is "
            f"{measured / allowance:.6g}x the pre-registered allowance of {allowance:.16g}. It "
            f"sits between the projection level "
            f"({readings['projection_from_the_in_frame_multiple']:.6g}, which already fails at "
            f"{levels['projection_from_the_in_frame_multiple']['x_bar']:.6g}x bar) and the frame "
            f"mismatch alone ({readings['the_frame_mismatch_alone']:.6g}). The clause reads "
            f"{fm['CLAUSE']['value']:.16g} at {fm['CLAUSE']['x_bar']:.6g}x the "
            f"{bar:.16g} bar and does not clear. The repair took it from "
            f"{was:.16g}x, removing "
            f"{out['WHAT_THE_REPAIR_MOVED']['excess_over_the_bar']['fraction_removed'] * 100:.4g} "
            f"% of the excess over the bar; the rest is ours."),
    }
    p = Path("perf/of3t_recut/LADDER_READ.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
