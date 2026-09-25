#!/usr/bin/env python3
"""of3t-stackexact: the monotonicity ladder, read against the pre-registration.

Reads, does not re-derive. The bar, the trunk allowance and the five levels come from
`perf/of3t_modelframe/CLAUSE.json`; the split is `perf/of3t_recutfin/bf16_split.py`'s own
function, as in of3t-modelever's read.py. Order, each step refusing the next:

  1. A42: every rung's device arm cites the model frame's own cot_external.pt, and every
     composed artifact carries the same correction digest.
  2. A/A: SHIP_A bit-identical to the banked recut arm and to SHIP_B; S bit-identical to
     of3t-modelever's EXACT arm; S's clause equal to 1.3037867474869442x. Otherwise STOP.
  3. Per rung: clause x bar, excess, closure, trunk split in both spaces with the A43 residual.
  4. f(L), dL, I, axis A, axis B and the verdict, by PREREGISTERED.md's rules as committed.
Boundary version for every gradient figure: upstream OpenFold3 0.4.3. CPU only.
"""
from __future__ import annotations

import importlib.util
import json
import re
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
H = REPO / "perf/of3t_stackexact"
SEC = "pairformer_stack"
MODEL_COT = "1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4"
RUNG2 = 1.3037867474869442
E_SHIP = 0.4511706984958472
T = 0.02
BOUNDARY_VERSION = "upstream OpenFold3 0.4.3"
RUNGS = [("SHIP", "SHIP_A", "none"), ("S", "S", "softmax"), ("L", "L", "layer_norm"),
         ("SL", "SL", "softmax,layer_norm")]

spec = importlib.util.spec_from_file_location("bf16_split", REPO / "perf/of3t_recutfin/bf16_split.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def J(p):
    return json.loads(Path(p).read_text())


def sec(doc, pair):
    v = doc["per_section"][pair][SEC]
    return v["mass_weighted_rel_l2"], v["mass_weighted_norm_ratio"], v["mass_weighted_cos"]


def ln_backward_reach(tag):
    """dev_cot's own LN counter, from the arm's raw log: 0 backward calls when exactln holds the verb."""
    for line in reversed(open(f"/home/ttuser/of3t_stackexact/raw_{tag}.log").read().splitlines()):
        if '"layer_norm_backward_reach"' in line:
            return json.loads(line[line.index("{"):])["layer_norm_backward_reach"]
    return None


def main() -> int:
    cl = J(REPO / "perf/of3t_modelframe/CLAUSE.json")["preregistered"]
    bar, allowance, levels = cl["bar"], cl["trunk_allowance_for_the_clause_to_clear"], cl["levels"]
    have = [r for r in RUNGS if (H / f"MODEL_{r[1]}_composed3660_n384.json").exists()]
    dev = {t: J(H / f"DEV_{t}.json") for t in ("SHIP_A", "SHIP_B", "S", "L", "SL")
           if (H / f"DEV_{t}.json").exists()}
    comp = {n: J(H / f"MODEL_{t}_composed3660_n384.json") for n, t, _ in have}

    # 1. A42
    cited = {t: d["provenance"]["injection"]["cotangent_sha256"] for t, d in dev.items()}
    carried = {n: d["injection"]["by_scope"]["renorm:pairformer_stack"]["correction"]["sha256"]
               for n, d in comp.items()}
    a42 = {"model_frame_cot_external_sha256": MODEL_COT, "cited_by_each_device_arm": cited,
           "carried_by_each_composed_artifact": carried,
           "all_equal": set(cited.values()) | set(carried.values()) == {MODEL_COT}}
    assert a42["all_equal"], a42

    # 2. A/A and reproduction
    pd = {p.stem: J(p) for p in sorted(H.glob("PD_*.json"))}
    keep = ("compared", "bit_identical", "differing", "all_bit_identical",
            "concatenated_rel_a_vs_b", "cos")
    aa = {k: {f: v[f] for f in keep} | {"a_sha256": v["arms"]["a"]["sha256"],
                                         "b_sha256": v["arms"]["b"]["sha256"],
                                         "arms_origin_D155": v["arms"]} for k, v in pd.items()}
    gate = {"SHIP_A_is_the_clause_arm": pd["PD_AA_SHIPA_vs_BANKED"]["all_bit_identical"],
            "SHIP_A_vs_SHIP_B": pd.get("PD_AA_SHIPA_vs_SHIPB", {}).get("all_bit_identical"),
            "S_is_modelever_EXACT": pd["PD_REPRO_S_vs_MODELEVER_EXACT"]["all_bit_identical"]}
    head = {n: d["stats"]["renorm_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"] for n, d in comp.items()}
    xbar = {n: v / bar for n, v in head.items()}
    gate["S_clause_x_bar"] = xbar.get("S")
    gate["S_reproduces_1.3037867474869442x"] = xbar.get("S") == RUNG2
    gate["ladder_licensed"] = bool(gate["SHIP_A_is_the_clause_arm"]
                                   and gate["S_reproduces_1.3037867474869442x"])

    # 3. per rung
    rungs = {}
    for n, t, scopes in have:
        d = comp[n]
        g = bs.split(*sec(d, "renorm_vs_UPSTREAM_BF16"), name=n, space="vs-upstream-bf16")
        f = bs.split(*sec(d, "renorm_vs_FLOAT64"), name=n, space="vs-float64, contrast only")
        e = xbar[n] - 1.0
        st = J(H / f"STACK_EXACT_{t}.json")
        rungs[n] = {
            "tag": t, "exact_scopes": scopes, "boundary_version": BOUNDARY_VERSION,
            "clause": head[n], "x_bar": xbar[n], "excess": e, "closure_f": 1.0 - e / E_SHIP,
            "clears": head[n] <= bar,
            "trunk_vs_their_bf16_GRADED": {k: g[k] for k in ("measured_rel_l2", "norm_ratio", "cos",
                                                            "angle_degrees")}
                                         | {"identity_residual": g["IDENTITY"]["rel_difference"]},
            "trunk_vs_float64_CONTRAST": {k: f[k] for k in ("measured_rel_l2", "norm_ratio", "cos",
                                                           "angle_degrees")}
                                         | {"identity_residual": f["IDENTITY"]["rel_difference"]},
            "model_vs_float64": d["stats"]["renorm_vs_FLOAT64"]["mass_weighted_rel_l2"],
            "trunk_x_allowance": sec(d, "renorm_vs_UPSTREAM_BF16")[0] / allowance,
            "reach": st["counters"], "installed": st["installed"],
            "dev_cot_layer_norm_backward_reach": ln_backward_reach(t),
            "device": {k: dev[t]["provenance"][k] for k in
                       ("host", "board_class", "card", "aiclk_mhz_sampled_DURING")}
                      | {"host_quiet": dev[t]["provenance"]["host_quiet"].splitlines()[-1]},
        }
    worst = max(max(r["trunk_vs_their_bf16_GRADED"]["identity_residual"],
                    r["trunk_vs_float64_CONTRAST"]["identity_residual"]) for r in rungs.values())

    # 4. the shape, by the committed rules
    shape = {"T": T, "rules": "perf/of3t_stackexact/PREREGISTERED.md, as committed at 0425448d4"}
    if gate["ladder_licensed"] and all(k in rungs for k in ("S", "L", "SL")):
        fS, fL, fSL = (rungs[k]["closure_f"] for k in ("S", "L", "SL"))
        dL = fSL - fS
        I = fSL - fS - fL
        if fL >= T and dL >= T and dL >= 0.5 * fL:
            A = "A-COMPOSES"
        elif dL < -T or (fL >= T and dL < 0.5 * fL):
            A = "A-CANCELS"
        elif abs(fL) < T and abs(dL) < T:
            A = "A-INERT"
        else:
            A = "A-OTHER"
        dthf = (rungs["SL"]["trunk_vs_float64_CONTRAST"]["angle_degrees"]
                - rungs["S"]["trunk_vs_float64_CONTRAST"]["angle_degrees"])
        B = "B-AWAY" if dthf > 0.5 else ("B-TOWARD" if dthf < -0.5 else "B-FLAT")
        if A == "A-COMPOSES" and B != "B-AWAY":
            verdict = "COMPOSES"
        elif A == "A-CANCELS" or (A == "A-COMPOSES" and B == "B-AWAY"):
            verdict = "CANCELS"
        elif A == "A-INERT":
            verdict = "INERT"
        else:
            verdict = "SPLIT"
        shape |= {"f_S": fS, "f_L": fL, "f_SL": fSL, "dL": dL, "I": I,
                  "dL_over_fL": (dL / fL) if fL else None,
                  "theta_f_SL_minus_S_degrees": dthf,
                  "theta_g_SL_minus_S_degrees": (
                      rungs["SL"]["trunk_vs_their_bf16_GRADED"]["angle_degrees"]
                      - rungs["S"]["trunk_vs_their_bf16_GRADED"]["angle_degrees"]),
                  "axis_A": A, "axis_B": B, "verdict": verdict}

    place = {n: {"below": sorted(k for k, v in levels.items() if v["clause_value"] < head[n]),
                 "at_or_above": sorted(k for k, v in levels.items() if v["clause_value"] >= head[n])}
             for n in head}
    out = {"what": __doc__.strip().splitlines()[0], "row": "of3t-stackexact",
           "host": socket.gethostname(), "device_involved": False,
           "why_no_aiclk": "the READ is CPU only; each rung's device host, board, card and "
                           "DURING AICLK are under rungs.<n>.device, stamped by arm.sh",
           "boundary_version": BOUNDARY_VERSION, "bar": bar, "trunk_allowance": allowance,
           "levels_source": "perf/of3t_modelframe/CLAUSE.json, read not re-derived",
           "A42": a42, "AA_AND_REPRODUCTION": {"gate": gate, "pairs": aa},
           "RUNGS": rungs, "LEVEL_PLACEMENT": place, "SHAPE": shape,
           "A43": {"definition": "concatenated", "worst_identity_residual": worst,
                   "bar": 1e-12, "holds": worst <= 1e-12}}
    (H / "LADDER.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"gate": gate, "x_bar": xbar, "SHAPE": shape, "A43": out["A43"]}, indent=1))
    return 0 if out["A43"]["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
