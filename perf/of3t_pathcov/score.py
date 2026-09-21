#!/usr/bin/env python3
"""Score a census: sites by class, and parameters by the mass no executed site reaches.

    score.py CENSUS_diffusion.json [...] --out COVERAGE.json

The denominator is the campaign's own: `REF_MASS.json`, the per-tensor squared float64 norm of
`grads_f64_043.pt`, whose total reproduces `SECTION_MASS_MEASURED.json`'s 10.27964 to 15 digits.
Scoring the coverage there is what makes it compose with the 92.1568 % rather than sit beside it.

Three classes, and the line between the second and the third is measured rather than argued:

  EXECUTED                     the site's line fired inside a tape.
  REACHABLE, NEVER EXECUTED    it did not, but its enclosing function DID run inside the tape.
                               The branch is on the path and this arm did not take it.
  NOT REACHED IN THIS ARM      the enclosing function never ran. Inside a shimmed module that
                               is a branch this arm does not enter; outside one it is another
                               model's file and the OF3 training step cannot reach it at all.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os


def _load(path):
    o = gzip.open if str(path).endswith(".gz") else open
    with o(path, "rt") as f:
        return json.load(f)


ap = argparse.ArgumentParser()
ap.add_argument("census", nargs="+")
ap.add_argument("--ref-mass", default="perf/of3t_pathcov/REF_MASS.json")
ap.add_argument("--sections", default="perf/of3t_orchestrator/SECTION_MASS_MEASURED.json")
ap.add_argument("--out", default="perf/of3t_pathcov/COVERAGE.json")
a = ap.parse_args()

ref = _load(a.ref_mass)
MASS, TOTAL = ref["mass"], ref["total_sq_norm"]
SEC = _load(a.sections)["sections_pct_of_model"]


def section_of(name):
    best = ""
    for s in SEC:
        if name == s or name.startswith(s + "."):
            if len(s) > len(best):
                best = s
    return best or name.split(".")[0]


def ref_name(leaf, scope_prefix):
    """A census leaf carries the instrument's checkpoint name, which is relative to the scope
    the instrument ran. The reference is keyed model-wide, so the arm's prefix is restored and
    the match is required rather than assumed."""
    for cand in ((scope_prefix + "." + leaf) if scope_prefix else leaf, leaf):
        if cand in MASS:
            return cand
    return None


SCOPE = {"CENSUS_diffusion.json": "diffusion_module",
         "CENSUS_cond.json": "diffusion_module.diffusion_conditioning",
         "CENSUS_msa.json": "msa_module", "CENSUS_aux.json": "aux_heads"}

out = {"denominator": {"file": a.ref_mass, "total_sq_norm": TOTAL,
                       "n_reference_tensors": ref["n_tensors"]},
       "arms": [], "sites_union": {}, "uncovered": {}}

reached_names, arm_rows = set(), []
for path in a.census:
    c = _load(path)
    prefix = SCOPE.get(os.path.basename(path).removesuffix(".gz"), "")
    sites = c["sites"]
    shim_files = {m.split(".")[-1] + ".py" for m in c["shimmed_modules"]}

    def fname(s):
        return os.path.basename(s["file"])

    def cls(s):
        if s["executed_in_tape"]:
            return "executed"
        if c["func_entered"].get(f"{s['file']}::{s['func']}"):
            return "reachable_never_executed"
        return "function_never_entered" if fname(s) in shim_files else "module_not_on_arm"

    by = collections.Counter(cls(s) for s in sites if s["taped_verb"])
    by_all = collections.Counter(cls(s) for s in sites)

    # parameters reached by an executed site. A leaf can carry TWO checkpoint names when the
    # device weight is a fused pair, and both halves own the gradient that flows through it.
    raw_names = c.get("leaf_names_all") or [[x] if x else [] for x in c["leaf_names"]]
    names = c["leaf_names"]
    leaf_ref = [sorted({r for nm in row if (r := ref_name(nm, prefix))}) for row in raw_names]
    unmatched = [nm for row in raw_names for nm in row if ref_name(nm, prefix) is None]
    reached_ref = set()
    for s in sites:
        if s["executed_in_tape"]:
            for b in s["params_idx"]:
                reached_ref.update(leaf_ref[b])
    reached_names |= reached_ref

    arm_names = {r for row in leaf_ref for r in row}
    arm_mass = sum(MASS[r] for r in arm_names)
    reached_mass = sum(MASS[r] for r in reached_ref)

    # the site table, by the mass each site's output reaches
    top = []
    for s in sites:
        if not s["executed_in_tape"] or not s["params_idx"]:
            continue
        got = set()
        for b in s["params_idx"]:
            got.update(leaf_ref[b])
        m = sum(MASS[r] for r in got)
        top.append((m, s["file"], s["line"], s["qual"], s["taped_calls"],
                    s["n_params_reached"]))
    top.sort(reverse=True)

    arm_rows.append({
        "census": path, "scope_prefix": prefix, "card": c.get("card"),
        "elapsed_s": c.get("elapsed_s"), "tape_opens": c["n_tape_opens"],
        "n_sites_total": c["n_sites"],
        "taped_verb_sites": {"executed": by["executed"],
                             "reachable_never_executed": by["reachable_never_executed"],
                             "function_never_entered": by["function_never_entered"],
                             "module_not_on_arm": by["module_not_on_arm"]},
        "all_sites": dict(by_all),
        "n_shimmed_modules": len(c["shimmed_modules"]),
        "raw_ttnn_calls_in_tape": {
            f: sum(s["raw_calls_in_tape"] for s in sites if s["file"] == f)
            for f in sorted({s["file"] for s in sites if s["raw_calls_in_tape"]})},
        "n_leaves": c["n_leaves"], "n_leaves_named": c["n_leaves_named"],
        "n_leaf_names_unmatched_in_reference": len(unmatched),
        "leaf_names_unmatched": unmatched[:10],
        "arm_mass_pct_of_model": 100.0 * arm_mass / TOTAL,
        "reached_mass_pct_of_model": 100.0 * reached_mass / TOTAL,
        "reached_pct_of_arm_mass": 100.0 * reached_mass / arm_mass if arm_mass else 0.0,
        "n_params_reached_by_an_executed_site": len(reached_ref),
        "top_sites_by_reached_mass": [
            {"mass_pct_of_model": 100.0 * m / TOTAL, "file": f, "line": l, "verb": q,
             "taped_calls": t, "n_params": n} for m, f, l, q, t, n in top[:15]],
    })

# model-wide: what no executed site in any supplied arm reaches
per_sec = collections.defaultdict(lambda: [0, 0.0, 0, 0.0])      # n, mass, n_cov, mass_cov
for name, m in MASS.items():
    s = section_of(name)
    per_sec[s][0] += 1
    per_sec[s][1] += m
    if name in reached_names:
        per_sec[s][2] += 1
        per_sec[s][3] += m
rows = []
for s, (n, m, nc, mc) in per_sec.items():
    rows.append({"section": s, "n_tensors": n, "n_reached": nc,
                 "pct_of_model": 100.0 * m / TOTAL,
                 "pct_of_model_reached": 100.0 * mc / TOTAL,
                 "pct_of_model_uncovered": 100.0 * (m - mc) / TOTAL})
rows.sort(key=lambda r: -r["pct_of_model_uncovered"])

cov = sum(MASS[n] for n in reached_names)
out["arms"] = arm_rows
out["model"] = {
    "n_reference_tensors": ref["n_tensors"],
    "n_reached_by_an_executed_site": len(reached_names),
    "pct_of_model_sq_grad_norm_reached": 100.0 * cov / TOTAL,
    "pct_of_model_sq_grad_norm_not_reached": 100.0 * (TOTAL - cov) / TOTAL,
    "by_section": rows,
    "worst_uncovered_section": rows[0]["section"] if rows else None,
}
json.dump(out, open(a.out, "w"), indent=1)

print(f"wrote {a.out}")
for r in arm_rows:
    t = r["taped_verb_sites"]
    print(f"\n{r['census']}  ({r['tape_opens']} tape opens, {r['elapsed_s']}s)")
    print(f"  taped-verb sites: {t['executed']} executed, "
          f"{t['reachable_never_executed']} reachable and never executed, "
          f"{t['function_never_entered']} in a function this arm never entered, "
          f"{t['module_not_on_arm']} in a module the arm does not load")
    print(f"  raw ttnn calls inside a tape: {r['raw_ttnn_calls_in_tape']}")
    print(f"  arm mass {r['arm_mass_pct_of_model']:.4f} % of model; reached by an executed "
          f"site {r['reached_mass_pct_of_model']:.4f} % "
          f"({r['reached_pct_of_arm_mass']:.4f} % of the arm)")
print(f"\nMODEL: {out['model']['pct_of_model_sq_grad_norm_reached']:.4f} % of the squared "
      f"gradient norm is reached by an executed site; "
      f"{out['model']['pct_of_model_sq_grad_norm_not_reached']:.4f} % is not")
for r in rows[:8]:
    print(f"  {r['pct_of_model_uncovered']:8.4f} %  uncovered  {r['section']}  "
          f"({r['n_reached']}/{r['n_tensors']} tensors reached)")
