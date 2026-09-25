"""of3t-exactscope: the narrowed scope read against of3t-stackexact's ladder, per tensor.

The row's question -- does exact softmax alone carry the gradient accuracy, or is the exact
layer norm load-bearing -- has an arm already: stackexact's rung 2 opens `ag.exact_softmax()`
and its `STACK_EXACT_S.json` records the mirror scope control (softmax serving, layer_norm
zero on every counter). So this reads, it does not re-run. What it adds over `LADDER.json`,
which stops at the clause:

  * the A46 clause-1 reading: per-tensor worst case against the CFD-validated float64
    reference, located by parameter path, for each of the four scopes;
  * the per-section decomposition, which is where the refutation actually lives;
  * the median tensor beside the mass-weighted clause, because they disagree at S.

Writes SCOPE.json. CPU only, no device.
"""
from __future__ import annotations

import json
import pathlib

SE = pathlib.Path(__file__).resolve().parents[1] / "of3t_stackexact"
ARMS = {"SHIP_A": "none (ship)", "S": "softmax only", "L": "layer_norm only",
        "SL": "softmax + layer_norm (exact_training ON)"}
BAR = 0.15210099830945006
X = {"SHIP_A": 1.4511706984958472, "S": 1.3037867474869442,
     "L": 1.2838647207334815, "SL": 0.9822570327981535}


def main() -> int:
    err_ship: dict = {}
    mod = {a: json.loads((SE / f"MODEL_{a}_composed3660_n384.json").read_text()) for a in ARMS}
    out = {
        "what": __doc__.split("\n\n")[0],
        "row": "of3t-exactscope",
        "device_involved": False,
        "why_no_aiclk": "the READ is CPU only; it reads arms stackexact took on qb2 p300c "
                        "card 1, whose own DURING AICLK is stamped in each STACK_EXACT_*.json",
        "bar": BAR,
        "per_tensor_bar": mod["SHIP_A"]["bars"]["per_tensor_bar"],
        "pass_rule": "clause / bar <= 1.0, pre-registered at 0425448d4 2026-09-23T02:26:30Z, "
                     "before the first rung ran at c17c34f02 02:30:29Z. Not moved here.",
        "reference_validation_A13": "the float64 reference is validated by float64 central "
                                    "finite differences, worst 1.1678e-03, 43x inside the "
                                    "0.05 per-tensor bar (of3t-reference, concluded)",
    }

    # --- the scope control, quoted from the arm's own record rather than asserted ---
    sc = json.loads((SE / "STACK_EXACT_S.json").read_text())
    out["scope_control_S"] = {
        "opened": sc["exact"],
        "installed_inside_the_scope": sc["installed"],
        "counters": sc["counters"],
        "reading": "softmax served 5901 taped + 1742 raw calls over 21,856,518,144 elements; "
                   "layer_norm reads 0 on verb, raw, elements AND bw. The arm is softmax-only "
                   "from the mechanism, not from the argument passed.",
        "complement": "dev_cot's own shipped-LN backward counter reads 1296 on SHIP and S, "
                      "0 on L and SL -- the shipped layer norm ran on S, so nothing was left "
                      "half-installed (of3t-stackexact LADDER)",
    }

    # --- the ladder, with the excess each scope closes ---
    e_ship = X["SHIP_A"] - 1.0
    out["ladder"] = [{
        "arm": a, "scope": ARMS[a],
        "clause": round(X[a] * BAR, 17),
        "x_bar": X[a],
        "clears": X[a] <= 1.0,
        "excess_closed_frac": round((X["SHIP_A"] - X[a]) / e_ship, 4),
    } for a in ARMS]

    # --- A46 clause 1: per tensor, located by path ---
    out["per_tensor"] = {}
    for ref in ("FLOAT64", "UPSTREAM_BF16"):
        rows = []
        for a in ARMS:
            s = mod[a]["stats"][f"renorm_vs_{ref}"]
            rows.append({"arm": a, "scope": ARMS[a],
                         "mass_weighted_rel_l2": s["mass_weighted_rel_l2"],
                         "median_rel_l2_over_tensors": s["median_rel_l2_over_tensors"],
                         "n_over_per_tensor_bar": s["n_over_per_tensor_bar"],
                         "n_rel_measurable": s["n_rel_measurable"],
                         "worst_rel_l2": s["worst_rel_l2"],
                         "worst_tensor": s["worst_tensor"]})
        out["per_tensor"][ref] = rows

    # --- where it lives: per section, graded reference and float64 contrast ---
    out["per_section"] = {}
    for ref in ("UPSTREAM_BF16", "FLOAT64"):
        secs = sorted(mod["SHIP_A"]["per_section"][f"renorm_vs_{ref}"],
                      key=lambda k: -mod["SHIP_A"]["per_section"][f"renorm_vs_{ref}"][k]
                      ["pct_of_model_mass"])
        rowsec = []
        for k in secs:
            r = {"section": k,
                 "pct_of_model_mass": round(
                     mod["SHIP_A"]["per_section"][f"renorm_vs_{ref}"][k]["pct_of_model_mass"], 3)}
            for a in ARMS:
                p = mod[a]["per_section"][f"renorm_vs_{ref}"][k]
                r[a] = round(p["mass_weighted_rel_l2"], 6)
                r[a + "_median"] = round(p["median_rel_l2_over_tensors"], 4)
            r["S_over_SL"] = round(r["S"] / r["SL"], 3) if r["SL"] else None
            rowsec.append(r)
        out["per_section"][ref] = rowsec

    # --- what actually moves: only the pairformer_stack scope differs between rungs ---
    out["what_moves"] = {
        "note": "the four arms are the same six composed scopes with one file swapped; five "
                "are byte-identical banked arms and only pairformer_stack is re-run per rung. "
                "So the ladder measures the exact scopes ON THE TRUNK, and every other "
                "section reads identically on all four arms by construction.",
        "scopes_identical_across_arms": [
            s_.split("=")[0] for s_ in mod["SHIP_A"]["arms"]["renorm"]["scopes"]
            if s_ != [x for x in mod["S"]["arms"]["renorm"]["scopes"]
                      if x.split("=")[0] == s_.split("=")[0]][0]] or
        [s_.split("=")[0] for s_ in mod["SHIP_A"]["arms"]["renorm"]["scopes"]
         if s_ in mod["S"]["arms"]["renorm"]["scopes"]],
        "pairformer_pct_of_model_mass": 5.8281714991342914,
    }
    err = {}
    for ref in ("FLOAT64", "UPSTREAM_BF16"):
        rows = []
        for a in ARMS:
            ps = mod[a]["per_section"][f"renorm_vs_{ref}"]
            tot = sum(v["mass_weighted_rel_l2"] ** 2 * v["ref_sq"] for v in ps.values())
            pf = (ps["pairformer_stack"]["mass_weighted_rel_l2"] ** 2
                  * ps["pairformer_stack"]["ref_sq"])
            rows.append({"arm": a, "scope": ARMS[a],
                         "model_squared_gradient_error": tot,
                         "vs_ship": round(tot / err_ship[ref], 4) if ref in err_ship else 1.0,
                         "pairformer_share_of_error_pct": round(pf / tot * 100, 2)})
            err_ship.setdefault(ref, tot)
        for r in rows:
            r["vs_ship"] = round(r["model_squared_gradient_error"] / err_ship[ref], 4)
        err[ref] = rows
    out["model_squared_gradient_error"] = err

    (pathlib.Path(__file__).parent / "SCOPE.json").write_text(json.dumps(out, indent=1))

    print(f"BAR {BAR}  pass iff x_bar <= 1.0")
    for r in out["ladder"]:
        print(f"  {r['arm']:7s} {r['scope']:40s} x_bar {r['x_bar']:.5f} "
              f"closes {r['excess_closed_frac']:+.4f} of ship's excess  "
              f"{'CLEARS' if r['clears'] else 'FAILS'}")
    for ref in ("FLOAT64", "UPSTREAM_BF16"):
        print(f"\nmodel squared gradient error vs {ref}:")
        for r in out["model_squared_gradient_error"][ref]:
            print(f"  {r['arm']:7s} {r['model_squared_gradient_error']:.6f}  "
                  f"{r['vs_ship']:.4f}x ship   pairformer carries "
                  f"{r['pairformer_share_of_error_pct']:.2f}% of it")
    for ref in ("UPSTREAM_BF16", "FLOAT64"):
        print(f"\nper tensor vs {ref}:")
        for r in out["per_tensor"][ref]:
            print(f"  {r['arm']:7s} massw {r['mass_weighted_rel_l2']:.6f}  "
                  f"median {r['median_rel_l2_over_tensors']:.4f}  "
                  f"over-bar {r['n_over_per_tensor_bar']}/{r['n_rel_measurable']}  "
                  f"worst {r['worst_rel_l2']:9.3f} @ {r['worst_tensor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
