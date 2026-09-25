#!/usr/bin/env python3
"""of3t-trunkopclass: the CEILING each op class can buy, by arithmetic on the census.

An ablation removes SOME of a class's error. Setting that class's error to exactly ZERO is the
upper bound on every possible ablation of it, so this screens the four classes before any device
time is spent: a class whose perfect arm still fails the clause cannot be closed by any arm.

The trunk figure the GRADIENTS clause reads is `ours_vs_bf16auto` over 2,736 tensors. Removing a
class's error mass leaves

    rel' = sqrt( (sum_err_sq - err_sq_class) / sum_ref_sq )

with `sum_ref_sq` unchanged, because an ablation changes the arm and not the reference.

Model scope is then re-pooled exactly the way `perf/of3t_wholemodel/model_scope.py` pools it,
    scope = sqrt( sum_s ref_sq_s * rel_s^2 / sum_s ref_sq_s ),
with the nine non-trunk sections held at their graded values. The control is that substituting
the trunk's own published figure reproduces the graded headline 0.5201243840984896.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

BAR = 0.14735268326440318           # A26 model-scope bar
CLAUSE_NEEDS = 0.4361680548         # what the trunk must read for GRADIENTS to close
GRADED_SCOPE = 0.5201243840984896   # the published model-scope headline
TRUNK_TODAY_CROSSFRAME = 2.0151163033431088   # the trunk section as the graded artifact has it
TRUNK_TODAY_INFRAME = 1.029395337772341       # ours vs upstream's own bf16, same 2,736 tensors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--model", required=True, help="MODEL_withtrunk_n384.json, for section mass")
    ap.add_argument("--section-arm", default="renorm_vs_UPSTREAM_BF16",
                    help="which arm's per-section table is the graded one")
    ap.add_argument("--label", default="ours_vs_bf16auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    C = json.load(open(a.census))
    cls = C["by_class"][a.label]
    tot_err_sq = sum(v["abs_err_sq"] for v in cls.values())
    tot_ref_sq = sum(v["ref_sq"] for v in cls.values())
    base = math.sqrt(tot_err_sq / tot_ref_sq)

    # --- the model-scope pooling, read out of the graded artifact -------------------------
    M = json.load(open(a.model))
    sections = M["per_section"][a.section_arm]

    def sec_fields(v):
        rs = v.get("ref_sq", v.get("reference_squared_norm"))
        rl = v.get("mass_weighted_rel_l2", v.get("rel_l2"))
        return rs, rl

    pool = {}
    for k, v in sections.items():
        rs, rl = sec_fields(v)
        if rs is None or rl is None:
            continue
        pool[k] = (float(rs), float(rl))
    if "pairformer_stack" not in pool:
        raise SystemExit(f"no pairformer_stack section; have {sorted(pool)}")

    def scope_with(trunk_rel):
        num = sum(rs * (trunk_rel if k == "pairformer_stack" else rl) ** 2
                  for k, (rs, rl) in pool.items())
        den = sum(rs for rs, _ in pool.values())
        return math.sqrt(num / den)

    ctrl = scope_with(pool["pairformer_stack"][1])
    R = {"what": __doc__.strip().splitlines()[0],
         "label": a.label,
         "trunk_base_in_frame": base,
         "trunk_published_in_frame": TRUNK_TODAY_INFRAME,
         "base_reproduces_published": abs(base - TRUNK_TODAY_INFRAME) / TRUNK_TODAY_INFRAME,
         "clause_needs_trunk_at_or_below": CLAUSE_NEEDS,
         "factor_still_to_find": base / CLAUSE_NEEDS,
         "model_scope_control": {
             "recomputed": ctrl, "published": GRADED_SCOPE,
             "rel_difference": abs(ctrl - GRADED_SCOPE) / GRADED_SCOPE,
             "trunk_section_value_used": pool["pairformer_stack"][1],
             "n_sections": len(pool)},
         "sections_held": {k: {"ref_sq": rs, "rel_l2": rl} for k, (rs, rl) in pool.items()},
         "ceilings": {}}

    # WHICH SUBSTITUTION, and it is not a free choice. The published clause threshold
    # 0.4361680548 was derived by putting the IN-FRAME trunk figure straight into the pool in
    # place of the graded section, so a ceiling that scaled it by the frame factor
    # (graded 2.0151163033 / in-frame 1.0293953378 = 1.9576) would answer a different question
    # with the same threshold. The control below is what says the convention is the right one:
    # substituting 0.4361680548 must land on the A26 bar itself.
    frame_factor = 1.0
    bar_check = scope_with(CLAUSE_NEEDS)
    R["clause_threshold_control"] = {
        "trunk_substituted": CLAUSE_NEEDS, "model_scope": bar_check, "A26_bar": BAR,
        "rel_difference": abs(bar_check - BAR) / BAR,
        "what": "the published clause threshold put through this pooling must land on the A26 "
                "bar; if it does not, this row is using a different frame from the one the "
                "threshold was derived in and no ceiling below is comparable to it"}
    R["frame_factor_graded_over_inframe"] = frame_factor
    print("clause-threshold control: trunk %.10f -> scope %.16f vs bar %.16f  rel diff %.3e"
          % (CLAUSE_NEEDS, bar_check, BAR, R["clause_threshold_control"]["rel_difference"]))

    names = sorted(cls)
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            rem = tot_err_sq - sum(cls[c]["abs_err_sq"] for c in combo)
            rel = math.sqrt(max(rem, 0.0) / tot_ref_sq)
            sc = scope_with(rel * frame_factor)
            R["ceilings"]["+".join(combo)] = {
                "classes_perfect": list(combo),
                "trunk_in_frame": rel,
                "factor_bought": base / rel if rel else None,
                "clause_passes": rel <= CLAUSE_NEEDS,
                "over_clause": rel / CLAUSE_NEEDS,
                "model_scope": sc,
                "over_A26_bar": sc / BAR,
                "scope_passes": sc <= BAR}

    json.dump(R, open(a.out, "w"), indent=2)
    print("base in frame %.10f  (published %.10f, rel diff %.2e)" % (
        base, TRUNK_TODAY_INFRAME, R["base_reproduces_published"]))
    print("model-scope control %.16f vs published %.16f  rel diff %.3e over %d sections" % (
        ctrl, GRADED_SCOPE, R["model_scope_control"]["rel_difference"], len(pool)))
    print("frame factor graded/in-frame = %.10f" % frame_factor)
    print()
    print("%-26s %14s %9s %9s %12s %9s %s" % (
        "classes made PERFECT", "trunk in frame", "factor", "/clause", "model scope",
        "/A26 bar", "verdict"))
    for k, v in sorted(R["ceilings"].items(), key=lambda kv: kv[1]["trunk_in_frame"]):
        print("%-26s %14.7f %9.4f %9.4f %12.7f %9.4f %s" % (
            k, v["trunk_in_frame"], v["factor_bought"] or 0, v["over_clause"],
            v["model_scope"], v["over_A26_bar"],
            "SCOPE PASSES" if v["scope_passes"] else ("clause ok" if v["clause_passes"]
                                                      else "fails")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
