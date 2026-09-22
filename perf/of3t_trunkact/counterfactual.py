#!/usr/bin/env python3
"""What model scope reads if the pairformer trunk is put at a given level, the other nine
sections held where they are.

This is ARITHMETIC ON MEASURED SECTIONS, not a new arm, and it is exact rather than
approximate: `model_scope.py` pools sections as
    total = sqrt( sum_s ref_sq_s * rel_s^2 / sum_s ref_sq_s )
and `MODEL_withtrunk_n384.json`'s own `reconciliation` records that this composition reproduces
its assembled headline with `rel_difference_exact: 0.0`. The control below re-derives the
unswapped headline before any rung is quoted; if that does not come back, no rung is admissible.
"""
from __future__ import annotations

import argparse
import json
import math

SECTION = "pairformer_stack"


def pool(secs, override=None):
    num = den = 0.0
    for s, v in secs.items():
        rel = override if (override is not None and s == SECTION) else v["mass_weighted_rel_l2"]
        num += v["ref_sq"] * (rel or 0.0) ** 2
        den += v["ref_sq"]
    return math.sqrt(num / den) if den else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="renorm_vs_UPSTREAM_BF16")
    a = ap.parse_args()
    d = json.load(open(a.artifact))
    secs = d["per_section"][a.arm]
    bar = d["bars"]["A26_reachable_bar_vs_their_bf16"]
    headline = d["stats"][a.arm]["mass_weighted_rel_l2"]
    floor = d["per_section"]["UPSTREAM_BF16_vs_FLOAT64"][SECTION]["mass_weighted_rel_l2"]
    ours = secs[SECTION]["mass_weighted_rel_l2"]

    ctl = pool(secs)
    R = {"what": __doc__.strip().splitlines()[0], "artifact": a.artifact, "arm": a.arm,
         "bar_A26_reachable_vs_their_bf16": bar,
         "control_repool_of_the_unswapped_sections": {
             "assembled_headline_in_the_artifact": headline,
             "repooled_here": ctl,
             "rel_difference": abs(ctl - headline) / headline,
             "exact_enough": abs(ctl - headline) / headline < 1e-12},
         "trunk_today": {"mass_weighted_rel_l2": ours,
                         "pct_of_model_mass": secs[SECTION]["pct_of_model_mass"],
                         "ref_sq": secs[SECTION]["ref_sq"],
                         "upstream_own_bf16_on_this_section_vs_float64": floor},
         "rungs": {}}
    rungs = {
        "trunk_at_upstreams_own_bf16_level": (floor,
            "the brief's counterfactual: the section reads what upstream's own bf16 step reads "
            "against float64 on the same 2,736 tensors"),
        "trunk_exactly_reproducing_upstreams_bf16": (0.0,
            "the unreachable limit, quoted so the rung above can be read as an interpolation "
            "and not as a ceiling"),
        "trunk_at_the_A26_bar_itself": (bar,
            "every section at the bar, which is what a uniformly bar-clearing port would read"),
        "trunk_unchanged": (ours, "today, as a check that the swap machinery is inert at 1.0x"),
    }
    for nm, (rel, why) in rungs.items():
        t = pool(secs, rel)
        R["rungs"][nm] = {"trunk_section_rel": rel, "model_scope": t,
                          "over_the_bar": t / bar, "passes": t <= bar, "why": why}
    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps(R["control_repool_of_the_unswapped_sections"], indent=1))
    for nm, v in R["rungs"].items():
        print(f"{nm:44s} trunk={v['trunk_section_rel']:.10f} model={v['model_scope']:.10f} "
              f"{v['over_the_bar']:.4f}x bar  passes={v['passes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
