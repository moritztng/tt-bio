#!/usr/bin/env python3
"""of3t-modelever: the clause and the magnitude/direction split for the exact-softmax trunk arm.

Reads, does not re-derive. The bar, the trunk allowance and the five levels come from the
pre-registered `perf/of3t_modelframe/CLAUSE.json`; the split and the reach arithmetic are
`perf/of3t_recutfin/bf16_split.py`'s own functions, imported, so the SHIP arm here reproduces
BF16_SPLIT.json by construction and the EXACT arm is read by the same code.

Order of reading, and each step refuses the next if it fails:
  1. A42: every device arm cites the model frame's own cot_external.pt digest, and the two
     composed artifacts carry the same correction digest. Asserted, not shape-checked.
  2. A/A floor: SHIP_A vs the banked arm and SHIP_A vs SHIP_B, stated beside the lever's own
     separation (EXACT vs SHIP_A) in the same statistic.
  3. The clause, `x the bar`, and where it sits on the five levels.
  4. The split in both spaces with the identity residual beside every angle (A43).
CPU only, no device.
"""
from __future__ import annotations

import importlib.util
import json
import math
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
H = REPO / "perf/of3t_modelever"
SEC = "pairformer_stack"
MODEL_COT = "1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4"

spec = importlib.util.spec_from_file_location("bf16_split", REPO / "perf/of3t_recutfin/bf16_split.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def J(p):
    return json.loads(Path(p).read_text())


def sec(doc, pair):
    v = doc["per_section"][pair][SEC]
    return v["mass_weighted_rel_l2"], v["mass_weighted_norm_ratio"], v["mass_weighted_cos"]


def corr_sha(doc):
    return doc["injection"]["by_scope"]["renorm:pairformer_stack"]["correction"]["sha256"]


def main() -> int:
    cl = J(REPO / "perf/of3t_modelframe/CLAUSE.json")["preregistered"]
    bar, allowance, levels = cl["bar"], cl["trunk_allowance_for_the_clause_to_clear"], cl["levels"]
    dev = {t: J(H / f"DEV_{t}.json") for t in ("SHIP_A", "SHIP_B", "EXACT")}
    ship, exact = J(H / "MODEL_BANKED_composed3660_n384.json"), J(H / "MODEL_EXACT_composed3660_n384.json")

    # 1. A42 -------------------------------------------------------------------------------
    cited = {t: d["provenance"]["injection"]["cotangent_sha256"] for t, d in dev.items()}
    cited["banked_recut_external"] = J(REPO / "perf/of3t_recut/DEV_RENORM_MODEL_N384_EXTERNAL.json")["cotangent_sha256"]
    composed = {"SHIP(banked)": corr_sha(ship), "EXACT": corr_sha(exact)}
    a42 = {"model_frame_cot_external_sha256": MODEL_COT, "cited_by_each_device_arm": cited,
           "carried_by_each_composed_artifact": composed,
           "all_equal": len(set(cited.values()) | set(composed.values()) | {MODEL_COT}) == 1}
    assert a42["all_equal"], a42

    # 2. floor beside separation --------------------------------------------------------------
    pd = {k: J(H / f"{k}.json") for k in ("AA_SHIPA_vs_BANKED", "AA_SHIPA_vs_SHIPB", "SEP_EXACT_vs_SHIPA")}
    floor = max(pd["AA_SHIPA_vs_BANKED"]["concatenated_rel_a_vs_b"],
                pd["AA_SHIPA_vs_SHIPB"]["concatenated_rel_a_vs_b"])
    sep = pd["SEP_EXACT_vs_SHIPA"]["concatenated_rel_a_vs_b"]
    aa = {k: {f: v[f] for f in ("compared", "bit_identical", "differing", "all_bit_identical",
                                "concatenated_rel_a_vs_b", "cos")} for k, v in pd.items()}
    aa["arms_origin_D155"] = {k: v["arms"] for k, v in pd.items()}
    b = pd["AA_SHIPA_vs_BANKED"]
    aa["CARRYABLE_ONTO_THE_CLAUSE"] = {
        "holds": b["all_bit_identical"],
        "ship_a_sha256": b["arms"]["a"]["sha256"],
        "clause_arm_sha256": b["arms"]["b"]["sha256"],
        "clause_arm": b["arms"]["b"]["pt"],
        "why": "SHIP_A is bit-identical, 2736/2736, to of3t-recut's external arm, the artifact "
               "the repointed clause was scored on. So the SHIP side of this lever pair IS the "
               "clause's arm by digest, and the EXACT reading is a lever on the clause's own "
               "arm, not on a comparable frame.",
    }
    aa["floor"] = floor
    aa["separation"] = sep
    aa["separation_over_floor"] = (sep / floor) if floor else "floor is exactly 0: any nonzero separation is the lever"

    # 3. the clause -------------------------------------------------------------------------
    head = {n: d["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"] for n, d in
            (("SHIP", ship), ("EXACT", exact))}
    trunk = {n: sec(d, "renorm_vs_UPSTREAM_BF16")[0] for n, d in (("SHIP", ship), ("EXACT", exact))}
    place = sorted(((v["clause_value"], k, v["clears"]) for k, v in levels.items()))
    below = [k for cv, k, _ in place if cv < head["EXACT"]]
    above = [k for cv, k, _ in place if cv >= head["EXACT"]]
    clause = {
        "bar": bar, "trunk_allowance": allowance, "source": "perf/of3t_modelframe/CLAUSE.json, read not re-derived",
        "clause": head, "x_bar": {n: v / bar for n, v in head.items()},
        "clears": head["EXACT"] <= bar,
        "trunk_vs_their_bf16": trunk, "trunk_x_allowance": {n: v / allowance for n, v in trunk.items()},
        "excess_over_bar_removed_fraction": (head["SHIP"] - head["EXACT"]) / (head["SHIP"] - bar),
        "levels_below_the_exact_arm": below, "levels_at_or_above_it": above,
        "levels": levels,
    }

    # 4. the split, both spaces ---------------------------------------------------------------
    def both(doc, who):
        return {"vs_upstream_bf16_GRADED": bs.split(*sec(doc, "renorm_vs_UPSTREAM_BF16"), name=who,
                                                    space="vs-upstream-bf16"),
                "vs_float64_CONTRAST": bs.split(*sec(doc, "renorm_vs_FLOAT64"), name=who,
                                                space="vs-float64, contrast only")}
    sp = {"SHIP": both(ship, "model-frame trunk, shipped device softmax"),
          "EXACT": both(exact, "model-frame trunk, tt_bio.autograd.exact_softmax()")}
    move = {}
    for space in ("vs_upstream_bf16_GRADED", "vs_float64_CONTRAST"):
        s, e = sp["SHIP"][space], sp["EXACT"][space]
        rel = lambda r, c: math.sqrt(max(0.0, 1 + r * r - 2 * r * c))
        d_rel = s["measured_rel_l2"] - e["measured_rel_l2"]
        move[space] = {
            "cos": [s["cos"], e["cos"]], "norm_ratio": [s["norm_ratio"], e["norm_ratio"]],
            "angle_degrees": [s["angle_degrees"], e["angle_degrees"]],
            "angle_closed_degrees": s["angle_degrees"] - e["angle_degrees"],
            "fraction_of_the_angle_closed": 1 - e["angle_degrees"] / s["angle_degrees"],
            "identity_residual": [s["IDENTITY"]["rel_difference"], e["IDENTITY"]["rel_difference"]],
            "counterfactual": {
                "lever_norm_ratio_at_shipped_cos": {"rel": rel(e["norm_ratio"], s["cos"]),
                    "share_of_the_move": (s["measured_rel_l2"] - rel(e["norm_ratio"], s["cos"])) / d_rel if d_rel else None},
                "lever_cos_at_shipped_norm_ratio": {"rel": rel(s["norm_ratio"], e["cos"]),
                    "share_of_the_move": (s["measured_rel_l2"] - rel(s["norm_ratio"], e["cos"])) / d_rel if d_rel else None},
            },
        }
    g = move["vs_upstream_bf16_GRADED"]
    need = bs.reach(sp["SHIP"]["vs_upstream_bf16_GRADED"], allowance)["BY_DIRECTION_ALONE"]["and_it_does_not_need_to_be_perfect"]
    g["against_the_requirement"] = {
        "cos_required_at_shipped_norm_ratio": need["cos_required_at_todays_norm_ratio"],
        "angle_required_degrees": need["angle_required_degrees"],
        "fraction_of_the_angle_that_had_to_close": need["fraction_of_todays_angle_that_must_close"],
        "fraction_closed": g["fraction_of_the_angle_closed"],
        "share_of_the_required_closure_delivered":
            g["angle_closed_degrees"] / (g["angle_degrees"][0] - need["angle_required_degrees"]),
        "source": "perf/of3t_recutfin/bf16_split.py reach(), on the SHIP arm of this same frame",
    }
    worst = max(max(m["identity_residual"]) for m in move.values())

    out = {
        "what": __doc__.strip().splitlines()[0], "row": "of3t-modelever", "host": socket.gethostname(),
        "device_involved": False, "why_no_aiclk": "the READ is CPU only. Every number in it is "
            "about device arms, whose host, board, card and DURING AICLK are under "
            "AA_FLOOR_AND_SEPARATION.arms_origin_D155 and `clocks`",
        "arms_host": sorted({(v["host"], v["board"], v["card"]) for x in pd.values()
                             for v in x["arms"].values()}),
        "A42": a42, "AA_FLOOR_AND_SEPARATION": aa, "CLAUSE": clause,
        "SPLIT": sp, "MOVE": move,
        "A43": {"definition": "concatenated: the three aggregates are norms of one vector pair "
                              "per section (model_scope.py), so rel^2 = 1 + r^2 - 2 r cos is exact",
                "worst_identity_residual": worst, "bar": 1e-12, "holds": worst <= 1e-12},
        "clocks": {t: d["provenance"]["aiclk_mhz_sampled_DURING"] for t, d in dev.items()},
        "host_quiet": {t: d["provenance"]["host_quiet"].splitlines()[-1] for t, d in dev.items()},
    }
    (H / "EVER_READ.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"A42": a42["all_equal"], "AA": aa, "CLAUSE": {k: clause[k] for k in
          ("clause", "x_bar", "clears", "trunk_x_allowance", "excess_over_bar_removed_fraction",
           "levels_below_the_exact_arm")}, "MOVE": move, "A43": out["A43"]}, indent=1))
    return 0 if out["A43"]["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
