#!/usr/bin/env python3
"""The other four models, one measurement each, for both fixes. No card.

`recipes.py` and `objectives.py` are shared, so "UNIFIED, NEVER PER-MODEL" makes the fixes
one change for five models -- and whether each model is AFFECTED is a measurement, not an
inference from that. Three per model, all executable:

  1. TRAINING REACH (D11). Every `AdamW(...)` construction that passes a `schedule` anywhere
     in the shipped package, by AST, with the recipes that reach it. One wiring site means
     one fix; two would mean the fix is incomplete and this is how that would show.
  2. FEATURE VOCABULARY (D12). The molecule-type feature keys each model's own shipped code
     subscripts out of its feature dict, by AST over that model's modules. `losses.mse`
     reads `is_dna` / `is_rna` / `is_ligand`; a model whose featuriser calls the same fact
     `mol_type` hands the adapter nothing and the entity weighting stays dropped for it.
  3. RUNTIME BEHAVIOUR (D12). `af3_loss` run on a batch built from each model's vocabulary,
     reporting the breakdown's `without`. This is the check that has to fire, so it is run
     rather than argued.

Measurement 2 is a census of the code as shipped, not of a fold: running five featurisers
needs MSAs, CCD data and a card's worth of patience, and the question here is which NAME each
model uses, which the code answers exactly.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())

OUT = "perf/of3t_updaterule"
ENTITY_KEYS = ("is_dna", "is_rna", "is_ligand")
MOLTYPE_KEYS = ENTITY_KEYS + ("is_protein", "mol_type", "entity_type", "molecule_type",
                              "is_atomized", "token_type", "atom_type")

# Each model's own shipped modules. Inference entry first, so a reader can check the list is
# the model rather than a grep pattern that happened to match.
MODELS = {
    "openfold3": ["tt_bio/openfold3_data.py", "tt_bio/openfold3_host_prep.py",
                  "tt_bio/openfold3_fold.py", "tt_bio/openfold3_confidence.py"],
    "protenix-v2": ["tt_bio/protenix.py", "tt_bio/protenix_data.py",
                    "tt_bio/protenix_template.py", "tt_bio/protenix_weights.py"],
    "boltz-2": ["tt_bio/boltz2.py"],
    "boltzgen": ["tt_bio/boltzgen"],
    "af2": ["tt_bio/af2.py", "tt_bio/af2_data.py", "tt_bio/af2_confidence.py",
            "tt_bio/af2_weights.py"],
}
# The model-side modules that read the frozen batch's own names, for openfold3. Its features
# are built by the vendored upstream pipeline, so the names are upstream's.
OF3_FEATURE_READER = "tt_bio/worker.py"


def py_files(spec: str):
    p = Path(spec)
    return sorted(p.rglob("*.py")) if p.is_dir() else [p]


def string_keys(paths):
    """Every string literal used as a dict key or a subscript, with where it was found.

    Both directions, deliberately. A featuriser WRITES its keys into a dict literal and the
    model READS them back by subscript, so looking at only one of the two answers a
    different question from "what does this model call a ligand".
    """
    seen = {}
    for p in paths:
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            lits = []
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                lits = [node.slice.value]
            elif isinstance(node, ast.Dict):
                lits = [k.value for k in node.keys
                        if isinstance(k, ast.Constant)]
            elif isinstance(node, ast.Call):
                lits = [k.arg for k in node.keywords if k.arg]
            for v in lits:
                if isinstance(v, str):
                    seen.setdefault(v, []).append(f"{p}:{node.lineno}")
    return seen


def arm_training_reach():
    """Every scheduled-optimizer construction in the shipped package, by AST."""
    sites = []
    for p in sorted(Path("tt_bio").rglob("*.py")):
        if "_vendor" in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "AdamW":
                sites.append({"file": str(p), "line": node.lineno,
                              "kwargs": sorted(k.arg for k in node.keywords if k.arg)})
    # `cli` carries the recipe names precisely so reading them does not import the tape,
    # and a shipped test pins the two lists together. Importing `recipes` here would need a
    # ttnn install for a question about which recipes exist.
    from tt_bio.train.cli import ADAPTABLE, RECIPE_NAMES
    from tt_bio.train.optim import AdamW
    src = AdamW.step.__code__
    return {"adamw_construction_sites": sites,
            "sites_passing_a_schedule": [s for s in sites if "schedule" in s["kwargs"]],
            "recipes": list(RECIPE_NAMES),
            "models_the_cli_admits": list(ADAPTABLE),
            "step_code_object": f"{src.co_filename}:{src.co_firstlineno}",
            "note": "one recipe, one scheduled AdamW, so every model's training step reads "
                    "the schedule at the same line; the fix cannot be per-model and the "
                    "census is what makes that a measurement rather than an assumption"}


def arm_vocabulary(frozen_of3_keys):
    """What each model calls a ligand, and whether the adapter's three names are among them.

    OpenFold3 is settled by the artifact rather than by the census: its features are built by
    the vendored upstream pipeline, so the definitive answer is which keys the frozen batch
    actually carries. The other four have no training batch in existence, so their own
    shipped code is the evidence available and it is read both ways.
    """
    out = {}
    for model, specs in MODELS.items():
        paths = [f for spec in specs for f in py_files(spec)]
        if model == "openfold3":
            paths += py_files(OF3_FEATURE_READER)
        keys = string_keys(paths)
        hit = {k: {"uses": len(keys[k]), "first": keys[k][0]} for k in MOLTYPE_KEYS
               if k in keys}
        if model == "openfold3":
            present = [k for k in ENTITY_KEYS if k in frozen_of3_keys]
            evidence = "of3t-reference's frozen batch, the real artifact"
        else:
            present = [k for k in ENTITY_KEYS if k in keys]
            evidence = f"{len(paths)} shipped modules, string keys read and written"
        out[model] = {
            "modules": len(paths), "distinct_string_keys": len(keys),
            "molecule_type_keys": hit, "evidence": evidence,
            "entity_keys_present": present,
            "entity_weighting_reaches_losses_mse": len(present) == len(ENTITY_KEYS),
            "own_name_for_the_same_fact": sorted(
                k for k in hit if k not in ENTITY_KEYS and k != "is_protein") or None,
        }
    return out


def arm_runtime(vocab):
    """`af3_loss` on a batch built from each model's vocabulary. The `without` field fires."""
    from tt_bio.train import losses, objectives
    rng = np.random.default_rng(7)
    n = 24
    true_xyz = rng.standard_normal((n, 3)) * 5.0
    pred_xyz = true_xyz + rng.standard_normal((n, 3))
    cm = np.ones(n)
    lig = np.zeros(n); lig[-2:] = 1.0          # two ligand tokens, as 5nw3 has
    weights = {"mse": 4.0}
    out = {}
    for model, v in vocab.items():
        batch = {"true_xyz": true_xyz, "coord_mask": cm}
        if v["entity_weighting_reaches_losses_mse"]:
            batch |= {"is_dna": np.zeros(n), "is_rna": np.zeros(n), "is_ligand": lig}
        else:                                   # the model's own name for the same fact
            for k in (v["own_name_for_the_same_fact"] or ["mol_type"]):
                batch[k] = lig
        total, breakdown, seeds = objectives.af3_loss(batch, {"pred_xyz": pred_xyz}, weights)
        out[model] = {"mse_value": breakdown["mse"]["value"],
                      "without": breakdown["mse"].get("without"),
                      "seed_norm": float(np.linalg.norm(
                          np.asarray(seeds["pred_xyz"], np.float64)))}
    # Both endpoints on the SAME synthetic batch, so the numbers above can be placed.
    b_full = {"true_xyz": true_xyz, "coord_mask": cm, "is_dna": np.zeros(n),
              "is_rna": np.zeros(n), "is_ligand": lig}
    b_none = {"true_xyz": true_xyz, "coord_mask": cm}
    vf, gf = losses.mse(pred_xyz, true_xyz, cm, is_dna=b_full["is_dna"],
                        is_rna=b_full["is_rna"], is_ligand=lig)
    vp, gp = losses.mse(pred_xyz, true_xyz, cm)
    out["_reference_endpoints"] = {
        "value_weighted": vf, "value_unweighted": vp,
        "rel_value": abs(vf - vp) / abs(vf),
        "rel_seed": float(np.linalg.norm(gf - gp) / np.linalg.norm(gf)),
        "batch": f"{n} tokens, 2 of them ligand, synthetic"}
    return out


def arm_inference_reach():
    """Does importing each model's inference module pull the changed modules in?

    A release-gated change to `tt_bio/train/` must not be able to move an inference result.
    Run in a fresh interpreter per model, because a previous import in this one would make
    every answer yes.
    """
    probe = ("import importlib,sys,json;"
             "importlib.import_module(sys.argv[1]);"
             "print(json.dumps(sorted(m for m in sys.modules if m.startswith('tt_bio.train'))))")
    mods = {"openfold3": "tt_bio.openfold3_fold", "protenix-v2": "tt_bio.protenix",
            "boltz-2": "tt_bio.boltz2", "boltzgen": "tt_bio.boltzgen",
            "af2": "tt_bio.af2"}
    out = {}
    for model, mod in mods.items():
        r = subprocess.run([sys.executable, "-c", probe, mod], capture_output=True,
                           text=True, cwd=os.getcwd(), timeout=600)
        if r.returncode != 0:
            out[model] = {"module": mod, "import_failed": r.stderr.strip().splitlines()[-1:]}
            continue
        loaded = json.loads(r.stdout.strip().splitlines()[-1])
        out[model] = {"module": mod, "tt_bio_train_modules_loaded": loaded,
                      "reaches_the_changed_code": any(
                          m in loaded for m in ("tt_bio.train.optim",
                                                "tt_bio.train.objectives"))}
    return out


def main() -> int:
    rep = {"what": "the other four models, one measurement each, for D11 and D12",
           "changed": ["tt_bio/train/optim.py AdamW.step", "tt_bio/train/objectives.py "
                       "_TERMS['mse'] and _OPTIONAL"]}

    rep["training_reach"] = t = arm_training_reach()
    print(f"[D11] {len(t['sites_passing_a_schedule'])} scheduled AdamW construction site(s) "
          f"in the shipped package: "
          f"{[s['file'] + ':' + str(s['line']) for s in t['sites_passing_a_schedule']]}; "
          f"recipes {t['recipes']}; CLI admits {t['models_the_cli_admits']}")

    import torch
    frozen = torch.load("/tmp/of3t/of3t-updaterule/batch_step003.pt", map_location="cpu",
                        weights_only=False)
    frozen_keys = sorted(frozen)
    rep["frozen_of3_batch_keys"] = {"n": len(frozen_keys),
                                    "entity_keys": [k for k in ENTITY_KEYS
                                                    if k in frozen_keys]}
    del frozen
    rep["feature_vocabulary"] = v = arm_vocabulary(frozen_keys)
    for m, d in v.items():
        print(f"[D12] {m:12s} molecule-type keys "
              f"{sorted(d['molecule_type_keys']) or '[]'} ({d['evidence']}); entity "
              f"weighting reaches losses.mse: "
              f"{d['entity_weighting_reaches_losses_mse']}")

    rep["runtime"] = r = arm_runtime(v)
    for m in v:
        print(f"[D12] {m:12s} af3_loss reports without={r[m]['without']}, "
              f"mse {r[m]['mse_value']:.6f}")
    e = r["_reference_endpoints"]
    print(f"[D12] on that batch the weighting is worth {e['rel_value']:.4f} on the value and "
          f"{e['rel_seed']:.4f} on the seed")

    rep["inference_reach"] = i = arm_inference_reach()
    for m, d in i.items():
        print(f"[gate] {m:12s} inference import -> train modules "
              f"{d.get('tt_bio_train_modules_loaded', d.get('import_failed'))}")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/models.json", "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(f"\nwrote {OUT}/models.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
