#!/usr/bin/env python3
"""PROTOCOL SS6: the coverage census, at runtime, on the frozen batch.

SS6's requirement is the UNION over (stage, dataset), not "all terms in one run": their loss
weights are per-dataset overrides inside each stage config and arrive per example as
`batch["loss_weights"]` (`runner.py:448`), so no single stage fires every term. For each term
this names the `(stage, dataset)` that fires it and shows a non-zero weight AND a non-zero
gradient contribution. A path that cannot fire is listed NOT COVERED with its reason.

WHAT IS REAL HERE AND WHAT IS NOT, stated before the numbers rather than after.
  REAL: the batch. `of3t-reference`'s frozen `batch_step003.pt`, sha256-verified against
        MANIFEST.json before it is opened -- 5nw3, 56 tokens, 422 atoms, cropped by their own
        WeightedPDBDataset. Every label below is derived from its `ground_truth`.
  REAL: the weights. `of3_loss_weights(stage, dataset)` for all eleven shipped (stage, dataset)
        pairs, and the frozen batch's OWN `loss_weights` as an independent check of that table.
  REAL: the loss. `tt_bio.train.objectives.af3_loss` and `tt_bio.train.losses`, the shipped code
        a training step calls, and the gradient each term seeds into the model outputs.
  NOT REAL: the prediction. tt-bio has no OF3 training forward -- nothing in `tt_bio/train/`
        produces these eight outputs from an OF3 batch -- so the prediction is a deterministic
        perturbation of the ground truth and the logits are a seeded draw. That makes this a
        census of the OBJECTIVE's coverage, not of the model's, and the model half is listed
        NOT COVERED with that reason rather than implied by the terms firing.

THE NEGATIVE CONTROL (SS3e) has to break what the check READS. The check reads "weight non-zero
AND seed non-zero", so the control sets the prediction EQUAL to the ground truth: the geometry
terms then have nothing to correct, their seeds collapse, and a census that still called them
covered would be reading the weight alone. Reported whatever it does.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())

OUT = "perf/of3t_gradients"
BUNDLE = "/home/ttuser/of3t/bundle_min"
SEED = 20260919
# A seed norm has to clear a FLOOR, not merely be unequal to zero, and this number was not
# chosen in advance -- it was forced by the control. With the prediction set equal to the
# ground truth, `mse`'s exact gradient is algebraically zero and reads 1.5206e-15 in float64.
# The first version of this census asked `gn > 0.0`, so the control left `mse` reported as
# still firing and rejected only `smooth_lddt`, whose gradient happens to cancel exactly. The
# check was passing on numerical dust. 1e-12 sits nine orders under the smallest real seed
# here (1.7753e-06 on `pae`) and three under the dust, so it separates them without coming
# near any live term.
SEED_FLOOR = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def in_function_ttnn_imports(root="tt_bio"):
    """SS6's file-level check: `import ttnn` inside a function body.

    `tape()` installs itself by rebinding the module-global name `ttnn`, so a function that
    imports it at CALL time binds the real module into its own scope and runs untaped. Nothing
    raises, because the raw handles it is then given are not `autograd.Tensor`s. A static scan
    is the right instrument for this one thing: the defect is in where the import statement
    sits, and that is a syntactic property.
    """
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(p, encoding="utf-8", errors="replace").read(), p)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Import) and any(
                            al.name == "ttnn" or al.name.startswith("ttnn.")
                            for al in sub.names):
                        hits.append({"file": p, "line": sub.lineno, "function": node.name,
                                     "form": "import ttnn"})
                    elif isinstance(sub, ast.ImportFrom) and (sub.module or "") == "ttnn":
                        hits.append({"file": p, "line": sub.lineno, "function": node.name,
                                     "form": "from ttnn import ..."})
    return hits


def labels_from(batch, np):
    """Every label the eight terms need, derived from the frozen batch's ground truth.

    ONE COORDINATE SET, at token scope. `af3_loss` hands the same `pred_xyz`/`true_xyz` to
    `mse` and to the three pair terms, and `losses.distogram` documents its logits as
    "[N, N, 64] on representative atoms" -- so the objective's N is the token count and the
    coordinate set is the representative atom per token. 56 tokens here, not 422 atoms.

    The derivations are listed because a census whose labels are wrong reports coverage of a
    function nobody computes:
      representative atom -- each token's `ground_truth.start_atom_index`, its first atom.
        Upstream selects a named per-residue representative; which atom it is decides the
        VALUE, never whether a term fires, and the choice is recorded rather than hidden.
      frames -- the token triple (i-1, i, i+1) clamped to the ends. Upstream builds frames
        from named backbone atoms, which a token-scope coordinate set does not carry; same
        caveat as above.
      nucleotide / polymer / ligand -- straight from `ground_truth.is_dna|is_rna|is_ligand`.
    """
    from tt_bio.train import losses
    gt = batch["ground_truth"]
    t = lambda x: np.asarray(x[0].to("cpu").float().numpy(), np.float64)
    i = lambda x: np.asarray(x[0].to("cpu").numpy())
    ntok = int(gt["token_mask"].sum())
    atom_xyz = t(gt["atom_positions"])
    atom_res = t(gt["atom_resolved_mask"])
    nat = atom_xyz.shape[0]
    start = i(gt["start_atom_index"])[:ntok].astype(np.int64)
    natoms = i(gt["num_atoms_per_token"])[:ntok].astype(np.int64)
    rep = np.clip(start, 0, nat - 1)

    true_xyz = atom_xyz[rep]                       # [N_token, 3]
    coord_mask = atom_res[rep]
    is_dna = i(gt["is_dna"])[:ntok].astype(np.float64)
    is_rna = i(gt["is_rna"])[:ntok].astype(np.float64)
    is_lig = i(gt["is_ligand"])[:ntok].astype(np.float64)
    is_nuc = (is_dna + is_rna) > 0
    is_poly = i(gt["is_protein"])[:ntok].astype(bool) | is_nuc
    idx = np.arange(ntok)
    frames = np.stack([np.clip(idx - 1, 0, ntok - 1), idx,
                       np.clip(idx + 1, 0, ntok - 1)], axis=-1)

    rng = np.random.default_rng(SEED)
    pred_xyz = true_xyz + rng.standard_normal(true_xyz.shape) * 1.0
    true_dist = losses._pdist(true_xyz)
    pred_dist = losses._pdist(pred_xyz)
    pair_mask = coord_mask[:, None] * coord_mask[None, :]
    lddt_pair_mask = losses.lddt_mask(true_dist, pair_mask, is_nuc)
    bonds = i(batch["token_bonds"])[:ntok, :ntok].astype(np.float64)
    lddt, lddt_w = losses.atom_bespoke_lddt(pred_xyz, true_xyz, is_nuc, is_poly,
                                            coord_mask.astype(bool))

    lg = lambda *s: rng.standard_normal(s) * 0.5
    labels = {"true_xyz": true_xyz, "coord_mask": coord_mask, "true_dist": true_dist,
              "lddt_pair_mask": lddt_pair_mask, "bond_mask": bonds,
              "per_atom_lddt": lddt, "per_atom_weight": lddt_w,
              "frame_atom_index": frames,
              "is_dna": is_dna, "is_rna": is_rna, "is_ligand": is_lig}
    outputs = {"pred_xyz": pred_xyz, "pred_dist": pred_dist,
               "distogram_logits": lg(ntok, ntok, 64), "pde_logits": lg(ntok, ntok, 64),
               "pae_logits": lg(ntok, ntok, 64), "plddt_logits": lg(ntok, 50),
               "resolved_logits": lg(ntok, 2)}
    meta = {"n_tokens": ntok, "n_atoms": nat,
            "n_resolved_atoms": int(atom_res.sum()),
            "n_resolved_tokens": int(coord_mask.sum()),
            "n_token_bonds": int(bonds.sum()),
            "n_nucleotide_tokens": int(is_nuc.sum()),
            "n_ligand_tokens": int(is_lig.sum()),
            "scope": "token, one representative atom per token",
            "representative_atom": "ground_truth.start_atom_index, each token's first atom",
            "frames": "the token triple (i-1, i, i+1), clamped at the ends"}
    return labels, outputs, meta


def census(labels, outputs, weights, np):
    from tt_bio.train.objectives import af3_loss
    total, breakdown, seeds = af3_loss(labels, outputs, weights)
    rows = {}
    seed_of = {"mse": "pred_xyz", "smooth_lddt": "pred_dist", "bond": "pred_dist",
               "distogram": "distogram_logits", "plddt": "plddt_logits", "pde": "pde_logits",
               "pae": "pae_logits", "resolved": "resolved_logits"}
    for term, d in breakdown.items():
        # The seed dict is summed per OUTPUT, so a term sharing an output with another term
        # cannot be read off it. Each term is therefore re-run alone to get ITS seed, which is
        # the only way `bond` and `smooth_lddt` -- both seeding `pred_dist` -- stay separable.
        alone = {k: (weights[k] if k == term else 0.0) for k in weights}
        _t, b1, s1 = af3_loss(labels, outputs, alone)
        g = s1.get(seed_of[term])
        gn = float(np.linalg.norm(np.asarray(g, np.float64))) if g is not None else 0.0
        rows[term] = {"weight": d["weight"], "value": d["value"],
                      "contribution": d["contribution"], "skipped": d.get("skipped"),
                      "seeds": seed_of[term], "seed_norm": gn,
                      "fired": bool(d["weight"] != 0.0 and d["value"] is not None
                                    and gn > SEED_FLOOR)}
    return total, rows


def main() -> int:
    import numpy as np
    import torch
    from tt_bio.train.losses import OF3_LOSS_OVERRIDES, OF3_CROP_TOKENS, of3_loss_weights

    rep = {"instrument": "PROTOCOL SS6 coverage census, runtime, on the frozen batch",
           "bundle": BUNDLE}

    # ---- the bundle, hashed before it is read ------------------------------------------------
    man = json.load(open(os.path.join(BUNDLE, "MANIFEST.json")))
    art = {}
    for a in man["artifacts"]:
        p = os.path.join(BUNDLE, a["file"])
        present = os.path.isfile(p)
        art[a["file"]] = {"declared_sha256": a["sha256"], "declared_on_qb2": a.get("on_qb2"),
                          "present": present,
                          "sha256": sha256(p) if present else None}
        art[a["file"]]["match"] = bool(present and art[a["file"]]["sha256"] == a["sha256"])
    rep["bundle_artifacts"] = art
    bp = os.path.join(BUNDLE, "batch_step003.pt")
    if not art["batch_step003.pt"]["match"]:
        print("batch_step003.pt does not match its declared sha256; refusing to compare")
        return 1
    batch = torch.load(bp, map_location="cpu", weights_only=False)

    # ---- their per-example weights against our table -----------------------------------------
    theirs = {k: float(v[0]) for k, v in batch["loss_weights"].items()}
    ours = of3_loss_weights("initial_training", "weighted-pdb")
    alias = {"experimentally_resolved": "resolved"}
    renamed = {alias.get(k, k): v for k, v in theirs.items()}
    rep["weights_table_check"] = {
        "batch_carries": theirs, "our_table_initial_training_weighted_pdb": ours,
        "agree": all(abs(ours[k] - v) <= 1e-9 * max(abs(v), 1.0) for k, v in renamed.items()),
        "max_abs_diff": max(abs(ours[k] - v) for k, v in renamed.items()),
        "their_LossWeights_field_count": len(theirs),
        "note": ("their `LossWeights` (dataset_config_components.py:164-172) declares EIGHT "
                 "fields and this batch carries eight, so a claim of eleven loss terms does "
                 "not match the artifact; `bond` is 0.0 on it, leaving seven non-zero")}

    labels, outputs, meta = labels_from(batch, np)
    rep["batch"] = {"pdb_id": batch["pdb_id"], **meta,
                    "sha256": art["batch_step003.pt"]["sha256"]}

    # ---- the union over (stage, dataset) ------------------------------------------------------
    pairs, by_term = {}, {}
    for stage, datasets in OF3_LOSS_OVERRIDES.items():
        for ds in sorted(datasets):
            w = of3_loss_weights(stage, ds)
            total, rows = census(labels, outputs, w, np)
            key = f"{stage}/{ds}"
            pairs[key] = {"crop_tokens": OF3_CROP_TOKENS[stage], "loss": total, "terms": rows}
            for term, r in rows.items():
                if r["fired"]:
                    by_term.setdefault(term, []).append(key)
    rep["per_stage_dataset"] = pairs
    terms = sorted(next(iter(pairs.values()))["terms"])
    rep["union"] = {t: {"covered": t in by_term,
                        "carried_by": (by_term.get(t) or [])[:3],
                        "n_pairs_firing": len(by_term.get(t, []))} for t in terms}
    rep["union_summary"] = {"terms": len(terms), "covered": sum(
        1 for t in terms if t in by_term), "pairs": len(pairs)}

    # ---- the mse adapter drops upstream's per-entity weighting -------------------------------
    # `objectives._TERMS["mse"]` calls `losses.mse(pred, true, coord_mask,
    # per_sample_scale=...)` and passes neither `is_dna`, `is_rna` nor `is_ligand`, though
    # `losses.mse` implements all three at upstream's 5.0 / 5.0 / 10.0. A census that only
    # asked whether `mse` fired would never see it: it fires either way, at the wrong weight.
    from tt_bio.train import losses as _L
    v_plain, g_plain = _L.mse(outputs["pred_xyz"], labels["true_xyz"], labels["coord_mask"])
    v_full, g_full = _L.mse(outputs["pred_xyz"], labels["true_xyz"], labels["coord_mask"],
                            is_dna=labels["is_dna"], is_rna=labels["is_rna"],
                            is_ligand=labels["is_ligand"])
    rep["mse_entity_weighting"] = {
        "adapter_passes_entity_flags": False,
        "site": "tt_bio/train/objectives.py:_TERMS['mse']",
        "value_as_shipped": v_plain, "value_with_upstream_weighting": v_full,
        "rel_value": abs(v_full - v_plain) / (abs(v_full) + 1e-30),
        "rel_seed": float(np.linalg.norm(g_full - g_plain) /
                          (np.linalg.norm(g_full) + 1e-30)),
        "ligand_tokens_on_this_batch": meta["n_ligand_tokens"],
        "upstream": "loss.py:1063-1207, w_dna 5.0 / w_rna 5.0 / w_ligand 10.0"}

    # ---- the negative control: break what the check reads --------------------------------------
    ctl_out = dict(outputs)
    ctl_out["pred_xyz"] = labels["true_xyz"].copy()
    ctl_out["pred_dist"] = labels["true_dist"].copy()
    _t, ctl_rows = census(labels, ctl_out, of3_loss_weights("initial_training", "weighted-pdb"),
                          np)
    base_rows = pairs["initial_training/weighted-pdb"]["terms"]
    rep["negative_control"] = {
        "what": "prediction set EQUAL to the ground truth, which is what the check reads",
        "per_term": {t: {"baseline_seed": base_rows[t]["seed_norm"],
                         "control_seed": ctl_rows[t]["seed_norm"],
                         "baseline_value": base_rows[t]["value"],
                         "control_value": ctl_rows[t]["value"],
                         "still_reported_fired": ctl_rows[t]["fired"]}
                     for t in terms},
        "rejected": sorted(t for t in terms
                           if base_rows[t]["fired"] and not ctl_rows[t]["fired"]),
        "answers": ("which check fails if our model is replaced by zeros? the seed norm does: "
                    "a term whose gradient contribution is zero is reported NOT fired even "
                    "though its weight is unchanged, which is the whole point of requiring "
                    "both")}

    # ---- SS6's file-level check ----------------------------------------------------------------
    rep["in_function_ttnn_imports"] = {
        "hits": in_function_ttnn_imports("tt_bio"),
        "why": ("tape() rebinds the module-global name `ttnn`, so a function that imports it at "
                "call time runs untaped and nothing raises")}
    hits = rep["in_function_ttnn_imports"]["hits"]
    rep["in_function_ttnn_imports"]["count"] = len(hits)
    rep["in_function_ttnn_imports"]["files"] = sorted({h["file"] for h in hits})
    # SS6 names two sites by line: openfold3_confidence.py:163 and openfold3_host_prep.py:219.
    # Whether they are still there is a fact about the tree today, not about the protocol.
    rep["in_function_ttnn_imports"]["on_the_of3_path"] = sorted(
        {h["file"] for h in hits if "openfold3" in os.path.basename(h["file"])})

    # ---- conditional paths, on THIS batch -------------------------------------------------------
    rep["conditional_paths"] = {
        "msa": {"covered": bool(float(batch["msa_mask"].sum()) > 0),
                "evidence": f"msa {tuple(batch['msa'].shape)}, mask sum "
                            f"{float(batch['msa_mask'].sum())}",
                "note": "2 MSA rows on this batch, so the path carries a tensor but not depth"},
        "templates": {"covered": bool(float(batch["template_pseudo_beta_mask"].sum()) > 0),
                      "evidence": f"template_pseudo_beta_mask sum "
                                  f"{float(batch['template_pseudo_beta_mask'].sum())} over "
                                  f"{tuple(batch['template_restype'].shape)}",
                      "reason_if_not": "the four template slots are all mask-zero on 5nw3, so "
                                       "the template path cannot fire on this batch whatever "
                                       "the stage"},
        "bond": {"covered": bool(meta["n_token_bonds"] > 0),
                 "evidence": f"token_bonds sum {meta['n_token_bonds']}",
                 "reason_if_not": "no inter-token bond on 5nw3, so `bond` has an empty mask "
                                  "even in finetune_1/2 where its weight is 4.0"},
        "nucleotide": {"covered": bool(meta["n_nucleotide_tokens"] > 0),
                       "evidence": f"{meta['n_nucleotide_tokens']} nucleotide tokens",
                       "reason_if_not": "5nw3 is one protein chain plus Fe and Na, so the DNA "
                                        "and RNA weightings in `mse` and the nucleotide lddt "
                                        "radius are never selected"},
        "ligand": {"covered": bool(meta["n_ligand_tokens"] > 0),
                   "evidence": f"{meta['n_ligand_tokens']} ligand tokens"},
        "confidence_heads": {"covered": True,
                             "evidence": "plddt, pde, pae and resolved all fire on "
                                         "initial_training/weighted-pdb"},
        "disabled_parameters": {
            "covered": False,
            "reason": "R6: their runner disables the confidence-head parameters when a "
                      "sample's summed confidence weight is zero, which initial_training does "
                      "on its four distillation datasets. The frozen batch is weighted-pdb, "
                      "the one dataset that does NOT zero them, so the disabled path has no "
                      "batch here. of3t-reference reports the same gap from its own side."},
        "diffusion_rollout": {
            "covered": False,
            "reason": "the rollout is a differentiated path through the diffusion module and "
                      "needs a model forward, which this census does not have"},
        "multichain_permutation": {
            "covered": False,
            "reason": "upstream raises KeyError 'ref_space_uid_to_perm' and falls back to "
                      "naive alignment"},
        "model_forward": {
            "covered": False,
            "reason": "tt-bio wires no OF3 training forward: nothing in tt_bio/train/ produces "
                      "the eight outputs from an OF3 batch, so no term's gradient was carried "
                      "into a parameter here. The routed-call census of a real taped forward "
                      "is of3t-tape's, cited rather than rebuilt."},
        "routed_call_census": {
            "covered": True,
            "source": "perf/of3t_tape/coverage_trunk_c1_b48_strict_bw.json",
            "evidence": "trunk, 1 cycle, 48 blocks: 30107 calls taped, 221 untaped, 14 nested"},
    }

    ok = (rep["weights_table_check"]["agree"]
          and rep["union_summary"]["covered"] >= 1
          and bool(rep["negative_control"]["rejected"]))
    rep["verdict"] = "PARTIAL" if ok else "FAIL"
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "coverage_census.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=str)

    print(f"weights table agrees with the batch: {rep['weights_table_check']['agree']} "
          f"(max diff {rep['weights_table_check']['max_abs_diff']:.3g})")
    print(f"union: {rep['union_summary']['covered']}/{rep['union_summary']['terms']} terms "
          f"fired over {rep['union_summary']['pairs']} (stage, dataset) pairs")
    for t in terms:
        u = rep["union"][t]
        print(f"   {t:14s} {'COVERED  ' if u['covered'] else 'NOT COVERED'} "
              f"{u['n_pairs_firing']:2d} pairs  {(u['carried_by'] or ['-'])[0]}")
    print(f"control rejected: {rep['negative_control']['rejected']}")
    print(f"mse entity weighting dropped by the adapter: value "
          f"{rep['mse_entity_weighting']['rel_value']:.3f} rel, seed "
          f"{rep['mse_entity_weighting']['rel_seed']:.3f} rel")
    print("weight non-zero but gradient contribution zero (SS6: skipped with extra steps):")
    for k, v in rep["per_stage_dataset"].items():
        for t, d in v["terms"].items():
            if d["weight"] != 0.0 and not d["fired"]:
                print(f"   {k:44s} {t:12s} w={d['weight']} seed={d['seed_norm']:.3e}")
    print(f"in-function `import ttnn`: {rep['in_function_ttnn_imports']['count']} hits")
    for h in rep["in_function_ttnn_imports"]["hits"][:12]:
        print(f"   {h['file']}:{h['line']} in {h['function']}()")
    print(f"\nVERDICT: {rep['verdict']}  ->  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
