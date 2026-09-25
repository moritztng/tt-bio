#!/usr/bin/env python3
"""SS6's census on the term whose weighting changed. No card.

The protocol's bar for a term is non-zero weight AND non-zero gradient seed, floor 1e-12 --
"a term whose weight is non-zero but whose gradient contribution is zero has not been
covered; it has been skipped with extra steps". `mse` is the only term this row touched, so
it is the only one that has to clear the bar again, but the whole eight are reported because
a change that silenced a neighbour would otherwise be invisible.

Run four ways on the real frozen batch: native flags (how OpenFold3 arrives), the same fact
as a `mol_type` column in each convention (how the other three arrive), and bare. Reuses
`of3t-gradients`' own `census`, not a second copy of it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "perf/of3t_updaterule")
sys.path.insert(0, "perf/of3t_gradients")

import mse_entity as U                                          # noqa: E402

OUT = "perf/of3t_entity"
FLOOR = 1e-12
ENTITY = ("is_dna", "is_rna", "is_ligand")


def main() -> int:
    from coverage_census import census, labels_from
    from entity_delta import _mol_type_from_flags
    from tt_bio.train.losses import of3_loss_weights

    got = U.sha256(U.BATCH)
    want = {a["file"]: a.get("sha256") for a in json.load(open(U.MANIFEST))["artifacts"]}
    if got != want["batch_step003.pt"]:
        print(f"batch sha256 {got} != declared; refusing to measure"); return 1
    batch = torch.load(U.BATCH, map_location="cpu", weights_only=False)
    labels, outputs, meta = labels_from(batch, np)
    weights = of3_loss_weights("initial_training", "weighted-pdb")

    bare = {k: v for k, v in labels.items() if k not in ENTITY}
    arms = {"native_flags": labels,
            "mol_type_af3": {**bare, "mol_type": _mol_type_from_flags(labels, "af3"),
                             "mol_type_convention": "af3"},
            "mol_type_boltz": {**bare, "mol_type": _mol_type_from_flags(labels, "boltz"),
                               "mol_type_convention": "boltz"},
            "mol_type_unnamed": {**bare,
                                 "mol_type": _mol_type_from_flags(labels, "af3")},
            "no_entity_labels": bare}

    rep = {"instrument": "PROTOCOL SS6 coverage census on the changed term",
           "floor": FLOOR, "stage_dataset": "initial_training/weighted-pdb",
           "batch": {"file": U.BATCH, "sha256": got, "pdb_id": batch["pdb_id"], **meta},
           "arms": {}}
    for name, lab in arms.items():
        total, rows = census(lab, outputs, weights, np)
        rep["arms"][name] = {"total": total, "terms": rows,
                             "terms_that_fired": sorted(k for k, v in rows.items()
                                                        if v["fired"]),
                             "terms_that_did_not": sorted(k for k, v in rows.items()
                                                          if not v["fired"])}

    n_terms = len(rep["arms"]["native_flags"]["terms"])
    base = rep["arms"]["native_flags"]
    same_set = all(rep["arms"][a]["terms_that_fired"] == base["terms_that_fired"]
                   for a in arms)
    mse_rows = {a: rep["arms"][a]["terms"]["mse"] for a in arms}
    rep["mse_across_arms"] = {
        a: {"weight": r["weight"], "value": r["value"], "seed_norm": r["seed_norm"],
            "fired": r["fired"]} for a, r in mse_rows.items()}
    # The derived arms must reproduce the native one exactly; the bare arm must not.
    rep["derived_matches_native"] = {
        a: bool(mse_rows[a]["value"] == mse_rows["native_flags"]["value"]
                and mse_rows[a]["seed_norm"] == mse_rows["native_flags"]["seed_norm"])
        for a in ("mol_type_af3", "mol_type_boltz", "mol_type_unnamed")}
    rep["bare_differs_from_native"] = bool(
        mse_rows["no_entity_labels"]["value"] != mse_rows["native_flags"]["value"])

    for a in arms:
        r = mse_rows[a]
        print(f"[SS6] {a:20s} mse weight {r['weight']:.1f}, value {r['value']:.6f}, "
              f"seed norm {r['seed_norm']:.6e}, fired {r['fired']}; "
              f"{len(rep['arms'][a]['terms_that_fired'])} of {n_terms} terms fired")
    print(f"[SS6] the same term set fires in every arm: {same_set}; derived reproduces "
          f"native exactly: {rep['derived_matches_native']}; bare differs: "
          f"{rep['bare_differs_from_native']}")

    ok = (same_set
          and all(mse_rows[a]["fired"] for a in arms)
          and all(mse_rows[a]["seed_norm"] > FLOOR for a in arms)
          and all(rep["derived_matches_native"].values())
          and rep["bare_differs_from_native"])
    rep["pass"] = bool(ok)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/census.json", "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(f"\n{'PASS' if ok else 'FAIL'} -- wrote {OUT}/census.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
