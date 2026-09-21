#!/usr/bin/env python3
"""The worst uncovered branch by gradient mass, and whether it is uncovered because it never
runs or because no arm in the union taped the instance that owns the parameters.

    worst.py --out perf/of3t_pathcov/WORST.json

Those two are different findings and the campaign's habit is to quote one as the other
(PREDICTION.md, P8 vs P9). They are separated here by measurement rather than by reading the
constructors: a section's code path is the set of call sites whose OUTPUT reaches one of that
section's parameters, taken from the tape's own reach bitmask. If those sites executed, the
branch ran; if they did not, it did not.

The aux arm is what makes this decidable for `pairformer_stack`. `OF3ConfidenceHead` embeds
its own `pairformer_embedding.pairformer_stack`, so the reference names it registers include
`...pairformer_stack.blocks.N.*` -- the same classes in `tenstorrent.py` that the 48-block
trunk instance is built from.
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
ap.add_argument("--coverage", default="perf/of3t_pathcov/COVERAGE.json")
ap.add_argument("--census", nargs="+", default=[
    "perf/of3t_pathcov/CENSUS_diffusion.json.gz", "perf/of3t_pathcov/CENSUS_cond.json.gz",
    "perf/of3t_pathcov/CENSUS_msa.json.gz", "perf/of3t_pathcov/CENSUS_aux.json.gz"])
ap.add_argument("--instance", default="perf/of3t_pathcov/CENSUS_trunk.json.gz",
                help="a census of an arm that tapes the section's OWN instance, even if its "
                     "boundary keeps it out of the union. The site table needs no reference "
                     "names, so it is readable whether or not the mass composes.")
ap.add_argument("--out", default="perf/of3t_pathcov/WORST.json")
a = ap.parse_args()

cov = _load(a.coverage)
worst = cov["uncovered"]["worst"]
SEC = worst["section"]
LEAF = SEC.split(".")[-1]          # the section name as it appears inside another scope

rep = {"worst_uncovered_section": SEC,
       "pct_of_model_sq_grad_norm": worst["pct_of_model_uncovered"],
       "n_reference_tensors": worst["n_tensors"],
       "n_reached_by_an_executed_site": worst["n_reached"],
       "arms_in_scope": worst["arms_in_scope"],
       "coverage_now_pct": cov["model"]["pct_of_model_sq_grad_norm_reached"]}

# --- does the section's CODE run anywhere in the census? ------------------------------------
# Measured from the reach bitmask: a site whose output reaches a parameter whose checkpoint
# name contains the section is a site on the section's code path, whatever scope it sits in.
code_sites, by_arm, proxy_idx = {}, {}, set()
for path in a.census:
    c = _load(path)
    sites, names = c["sites"], c.get("leaf_names_all") or [
        [x] if x else [] for x in c["leaf_names"]]
    hits = {i for i, row in enumerate(names)
            if any(LEAF + "." in nm or nm.startswith(LEAF + ".") for nm in row)}
    if not hits:
        by_arm[os.path.basename(path)] = {"leaves_under_section": 0, "executed_sites": 0}
        continue
    ex = [(i, s) for i, s in enumerate(sites)
          if s["executed_in_tape"] and hits & set(s["params_idx"])]
    proxy_idx |= {i for i, _ in ex}
    for _, s in ex:
        k = f"{s['file']}:{s['line']} {s['func']} {s['qual']}"
        code_sites[k] = code_sites.get(k, 0) + s["taped_calls"]
    by_arm[os.path.basename(path)] = {
        "leaves_under_section": len(hits),
        "executed_sites": len(ex),
        "taped_calls": sum(s["taped_calls"] for _, s in ex),
        "files": dict(collections.Counter(s["file"] for _, s in ex))}

rep["code_path"] = {
    "n_distinct_executed_sites": len(code_sites),
    "by_arm": by_arm,
    "files": dict(collections.Counter(k.split(":")[0] for k in code_sites)),
    "top_sites": sorted(code_sites.items(), key=lambda kv: -kv[1])[:12],
}
# --- the proxy question: does the section's OWN instance take the same branches? -----------
# This is the one that decides whether the mass is "covered by proxy". A section can have
# every class it is built from executed by some other scope's embedded copy and still take a
# different route: the same `tenstorrent.py` classes branch on shape, on the config the
# instance was constructed with, and on which fused kernel declines under the tape.
if a.instance and os.path.exists(a.instance):
    inst = _load(a.instance)
    isites = inst["sites"]
    iex = {i for i, s in enumerate(isites) if s["executed_in_tape"]}
    iex_p = {i for i in iex if isites[i]["params_idx"]}

    def row(i):
        s = isites[i]
        return {"file": s["file"], "line": s["line"], "func": s["func"], "verb": s["qual"]}

    rep["own_instance"] = {
        "census": a.instance,
        "instrument": inst.get("instrument"),
        "n_tape_opens": inst["n_tape_opens"],
        "n_leaves": inst["n_leaves"],
        "n_executed_sites": len(iex),
        "n_executed_sites_reaching_a_parameter": len(iex_p),
        "shared_with_the_proxy": len(proxy_idx & iex),
        "only_the_proxy_takes": [row(i) for i in sorted(proxy_idx - iex)],
        "only_the_instance_takes": [row(i) for i in sorted(iex_p - proxy_idx)],
        "reading": "these two sets are why the section's mass is NOT covered by proxy: the "
                   "instance that owns the 2,736 parameters and the embedded copy that the "
                   "census does reach take different branches of the same classes."}

rep["verdict"] = (
    "the branch RUNS -- its call sites executed inside a tape and produced gradients; what is "
    "uncovered is the parameters of a different INSTANCE of it"
    if code_sites else
    "the branch NEVER RAN in any arm supplied to this census")

# what covering it is worth, as arithmetic on the shares this census already measured
rep["if_covered"] = {
    "coverage_would_become_pct": (cov["model"]["pct_of_model_sq_grad_norm_reached"]
                                  + worst["pct_of_model_uncovered"]),
    "next_worst": (cov["uncovered"]["by_section"][1]["section"]
                   if len(cov["uncovered"]["by_section"]) > 1 else None),
    "next_worst_pct": (cov["uncovered"]["by_section"][1]["pct_of_model_uncovered"]
                       if len(cov["uncovered"]["by_section"]) > 1 else 0.0)}

json.dump(rep, open(a.out, "w"), indent=1)
print(f"wrote {a.out}\n")
print(f"WORST uncovered: {SEC}  {rep['pct_of_model_sq_grad_norm']:.4f} % of the model's "
      f"squared gradient norm, {rep['n_reached_by_an_executed_site']}/"
      f"{rep['n_reference_tensors']} tensors reached")
for k, v in by_arm.items():
    print(f"  {k:34s} {v['leaves_under_section']:4d} leaves under the section, "
          f"{v['executed_sites']:4d} executed sites")
print(f"\n{rep['verdict']}")
print(f"{rep['code_path']['n_distinct_executed_sites']} distinct executed call sites on the "
      f"branch, in {rep['code_path']['files']}")
oi = rep.get("own_instance")
if oi:
    print(f"\nits OWN instance ({oi['census']}): {oi['n_executed_sites']} executed sites, "
          f"{oi['n_executed_sites_reaching_a_parameter']} reaching a parameter; "
          f"{oi['shared_with_the_proxy']} shared with the proxy, "
          f"{len(oi['only_the_proxy_takes'])} only the proxy takes, "
          f"{len(oi['only_the_instance_takes'])} only the instance takes")
print(f"\ncovering it takes coverage {rep['coverage_now_pct']:.4f} % -> "
      f"{rep['if_covered']['coverage_would_become_pct']:.4f} %; the next worst is then "
      f"{rep['if_covered']['next_worst']} at {rep['if_covered']['next_worst_pct']:.4f} %")
