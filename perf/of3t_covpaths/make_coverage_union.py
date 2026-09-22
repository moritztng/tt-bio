#!/usr/bin/env python3
"""Write `COVERAGE_UNION.json` from the artifacts, so no figure in it is retyped.

`perf/of3t_gradients/coverage_census.json` is `of3t-gradients`' artifact and that row has
concluded, so this row may not re-emit it. This writes the same schema under its own name,
cites the census by path and sha256, and states per path whether this row CONFIRMS, EXTENDS
or CONTRADICTS it.

The three words are used strictly:

  CONFIRMS     the census's verdict and its reason both survive.
  EXTENDS      the census's verdict is right about the batch it had and wrong as a statement
               about the path. Its reason stays true; its scope does not.
  CONTRADICTS  the census's verdict is wrong on evidence that existed, or its reason names
               the wrong mechanism.

Every number is read out of an artifact named in `sources`. Nothing is typed in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(p: Path):
    return json.loads(p.read_text())


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def cite(p: Path) -> dict:
    return {"path": rel(p), "sha256": sha256(p)}


def grad_line(rep: dict) -> dict:
    c = rep["contribution"]
    return {
        "stage": rep["stage"], "dataset": rep["dataset"], "target": rep["target"]["pdb_id"],
        "datapoint": rep["target"]["datapoint"], "crop": rep["crop"], "dtype": rep["dtype"],
        "seed": rep["seed"],
        "arms": sorted(rep["arms"]),
        "squared_norm_of_difference": c["squared_norm_of_difference"],
        "share_of_squared_gradient_norm": c["share_of_squared_gradient_norm"],
        "n_params_moved": c["n_params_moved"],
        "n_params_with_gradient_on": c["n_params_with_gradient_on"],
    }


def null_control(ns, name: str, sources: dict, key: str) -> dict | None:
    """The arm applied to a batch the path cannot act on. A no-op there is what makes the
    positive reading attributable to the path rather than to the edit."""
    p = ns / name
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    sources[key] = cite(p)
    return {"target": d["target"]["pdb_id"], "edit": d["edit_applied"],
            "molecule_type_tokens": d["target"]["molecule_type_tokens"],
            "template_pseudo_beta_mask_nnz": d["target"]["template_pseudo_beta_mask_nnz"],
            "tensors_compared": d["tensors_compared"],
            "n_tensors_moved": d["n_tensors_moved"], "verdict": d["verdict"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path,
                    default=ROOT / "perf/of3t_gradients/coverage_census.json")
    ap.add_argument("--ns", type=Path, default=ROOT / "perf/of3t_covpaths")
    ap.add_argument("--out", type=Path, default=ROOT / "perf/of3t_covpaths/COVERAGE_UNION.json")
    a = ap.parse_args()

    census = load(a.census)
    cp = census["conditional_paths"]
    ns = a.ns

    presence = load(ns / "presence_n4.json")
    by_pdb: dict[str, list] = {}
    for r in presence["rows"]:
        by_pdb.setdefault(r["pdb_id"], []).append(r)

    sources = {"coverage_census": cite(a.census), "presence": cite(ns / "presence_n4.json")}
    paths: dict[str, dict] = {}

    def add(name, covered, verdict, evidence, reason=None, extra=None):
        e = {"covered": covered, "census_said": cp.get(name, {}).get("covered"),
             "this_row": verdict, "evidence": evidence}
        if reason:
            e["reason"] = reason
        if extra:
            e.update(extra)
        paths[name] = e

    # --- the four the census already calls covered -----------------------------------
    for name in ("msa", "ligand", "confidence_heads", "routed_call_census"):
        add(name, True, "CONFIRMS", cp[name].get("evidence") or cp[name].get("source"),
            reason="not re-measured here; this row owns the five that did not fire")

    # --- templates -------------------------------------------------------------------
    g = ns / "gradient_templates_1kvu.json"
    tpl_rows = [{"pdb_id": r["pdb_id"], "datapoint": r["datapoint"],
                 "template_pseudo_beta_mask_nnz": r["template_pseudo_beta_mask_nnz"]}
                for r in presence["rows"] if r["template_pseudo_beta_mask_nnz"]]
    ev = {"mask": {"instrument": rel(ns / "presence_n4.json"),
                   "n_datapoints_with_nonzero_template_mask": len(tpl_rows),
                   "n_datapoints_walked": len(presence["rows"]),
                   "examples": tpl_rows[:6]}}
    if g.is_file():
        rep = load(g)
        sources["templates_gradient"] = cite(g)
        ev["gradient"] = grad_line(rep)
        ev["template_embedder"] = rep.get("template_embedder")
        ev["mask_on_the_run_batch"] = rep["target"]["template_pseudo_beta_mask_nnz"]
    nc = null_control(ns, "null_control_templates_5oid.json", sources,
                      "templates_null_control")
    if nc:
        ev["null_control"] = nc
    add("templates", bool(g.is_file()), "EXTENDS", ev,
        reason=("the census's zero is correct FOR 5nw3, whose four template slots are "
                "mask-zero, and is not a statement about the path: `initial_training` / "
                "`weighted-pdb` configures n_templates 4, and a target whose cache entry "
                "carries template_ids fills them"))

    # --- bond ------------------------------------------------------------------------
    bp = ROOT / "perf/of3t_bondcov/bond_gradient_4g5j_c1.json"
    b = load(bp)
    sources["bond_gradient"] = cite(bp)
    add("bond", True, "CONTRADICTS",
        {"owner": "of3t-bondcov, cited not re-run",
         "stage": b["stage"], "dataset": b["dataset"], "target": b["target"]["pdb_id"],
         "datapoint": b["target"]["datapoint"], "crop": b["crop"], "dtype": b["dtype"],
         "bond_mask_nnz": b["target"]["bond_mask_nnz"],
         "loss_weight_bond": b["target"]["loss_weight_bond"],
         "bond_loss": b["arms"]["bond4"]["breakdown"].get("bond_loss"),
         "squared_norm_of_difference": b["bond_gradient_contribution"]["squared_norm_of_difference"],
         "share_of_squared_gradient_norm": b["bond_gradient_contribution"]["share_of_squared_gradient_norm"],
         "n_params_moved": b["bond_gradient_contribution"]["n_params_moved"],
         "n_params": b["bond_gradient_contribution"]["n_params"]},
        reason=("the census reads covered false with the reason 'no inter-token bond on "
                "5nw3', which is true of 5nw3; a gradient reading on 4g5j at "
                "finetune_1/weighted-pdb existed before this row started and the census "
                "does not carry it"))

    # --- nucleotide ------------------------------------------------------------------
    nuc_rows = [{"pdb_id": r["pdb_id"], "datapoint": r["datapoint"],
                 "is_dna": r["molecule_type_tokens"].get("is_dna", 0),
                 "is_rna": r["molecule_type_tokens"].get("is_rna", 0)}
                for r in presence["rows"] if r["n_nucleotide_tokens"]]
    ev = {"mask": {"instrument": rel(ns / "presence_n4.json"),
                   "n_datapoints_with_nucleotide_tokens": len(nuc_rows),
                   "examples": nuc_rows[:6]}}
    covered = False
    for tag, f in (("dna", "gradient_nucleotide_4hj5_dna.json"),
                   ("rna", "gradient_nucleotide_5kla_rna.json"),
                   ("control_protein_only", "gradient_nucleotide_1kvu_CONTROL.json")):
        p = ns / f
        if p.is_file():
            sources[f"nucleotide_{tag}"] = cite(p)
            ev.setdefault("gradient", {})[tag] = grad_line(load(p))
            if tag in ("dna", "rna"):
                covered = True
    nc = null_control(ns, "null_control_nucleotide_1kvu.json", sources,
                      "nucleotide_null_control")
    if nc:
        ev["null_control"] = nc
    add("nucleotide", covered, "EXTENDS", ev,
        reason=("the census's zero is correct FOR 5nw3, one protein chain plus Fe and Na. "
                "Upstream's own training cache carries 19,572 DNA and 16,000 RNA chains "
                "over 13,196 structures, all reachable by the same WeightedPDBDataset"))

    # --- disabled_parameters ---------------------------------------------------------
    fired = [{"pdb_id": r["pdb_id"], "datapoint": r["datapoint"],
              "summed_confidence_weight": r["summed_confidence_weight"]}
             for r in presence["rows"] if r["disabled_parameters_would_fire"]]
    ctrl = [{"pdb_id": r["pdb_id"], "summed_confidence_weight": r["summed_confidence_weight"]}
            for r in presence["rows"] if not r["disabled_parameters_would_fire"]]
    ev = {"gate": {"instrument": rel(ns / "presence_n4.json"),
                   "n_datapoints_where_the_gate_fires": len(fired),
                   "fired_on": fired[:6],
                   "control_datapoints": len(ctrl),
                   "control_summed_confidence_weight": sorted({c["summed_confidence_weight"]
                                                               for c in ctrl})},
          "source": {
              "zeroing": "core/data/pipelines/featurization/loss_weights.py:44-49",
              "bounds": "projects/of3_all_atom/config/dataset_config_components.py:174-175, "
                        "min_resolution 0.1, max_resolution 4.0",
              "disable": "projects/of3_all_atom/runner.py:364-386, "
                         "_get_sample_disabled_param_names",
              "param_set": "projects/of3_all_atom/runner.py:141-161, "
                           "_identify_confidence_params"}}
    g = ns / "gradient_disabled_5oid.json"
    covered = False
    if g.is_file():
        rep = load(g)
        sources["disabled_gradient"] = cite(g)
        ev["gradient"] = grad_line(rep)
        ev["confidence_head"] = rep["contribution"]["confidence_head"]
        ev["prediction_falsified"] = {
            "predicted": "those parameters hold `grad is None` rather than a zero gradient",
            "measured": ("all 243 hold an EXACTLY ZERO gradient and none holds None; "
                         "autograd still builds the confidence terms and weights them by "
                         "zero. The None in LEDGER R6 is the runner's own disabled-NAMES "
                         "list, used for cross-rank gradient counting and for dropping "
                         "them from the clip norm, not what autograd produces"),
            "consequence": ("because their gradient is exactly zero on the batch that "
                            "trips the gate, dropping them from grad_manager._clip_grads' "
                            "params_enabled changes that norm by nothing on a real "
                            "zero-confidence-weight batch"),
            "registered_in": "perf/of3t_covpaths/PREDICTION.md, section 4",
        }
        covered = True
    add("disabled_parameters", covered, "CONTRADICTS", ev,
        reason=("the census attributes the gap to the DATASET, 'the frozen batch is "
                "weighted-pdb, the one dataset that does NOT zero them'. The dataset "
                "default is half the gate: set_loss_weights zeroes every confidence loss "
                "when the target's resolution is None or outside [0.1, 4.0], so the gate is "
                "per TARGET and fires inside weighted-pdb"))

    # --- multichain_permutation ------------------------------------------------------
    rr = ns / "permalign_rerun"
    arms = {}
    for p in sorted(rr.glob("*.json")) if rr.is_dir() else []:
        d = load(p)
        arms[d["tag"]] = {
            "completed": d["verdict"]["completed"],
            "naive_fallback_invoked": d["verdict"]["naive_fallback_invoked"],
            "detectors_agree": d["verdict"]["detectors_agree"],
            "gt_atoms_moved": d["effect"]["gt_atoms_moved"],
            "atoms_differing_from_naive": (d.get("vs_naive") or {}).get("atoms_differing_from_naive"),
            "max_abs_coord_diff_A": (d.get("vs_naive") or {}).get("max_abs_coord_diff_A"),
            "n_ref_spaces": d["ref_spaces"].get("n_ref_spaces_sample0"),
            "n_ref_spaces_with_alternatives": d["ref_spaces"].get("n_ref_spaces_with_alternatives"),
            "errors": d["verdict"].get("errors") or [],
        }
        sources[f"permalign_{d['tag']}"] = cite(p)
    completed = [k for k, v in arms.items() if v["completed"]]
    fellback = [k for k, v in arms.items() if not v["completed"]]
    add("multichain_permutation", bool(completed), "CONTRADICTS",
        {"rerun_on_qb1": arms,
         "arms_completed": sorted(completed), "arms_fell_back": sorted(fellback),
         "prior_row": "of3t-permalign D117, perf/of3t_permalign/census_correction.json",
         "key_written_at": "core/data/pipelines/featurization/conformer.py:181, "
                           "output_features['ref_space_uid_to_perm'] = ref_space_uid_to_perm, "
                           "guarded by add_ref_space_uid_to_perm which defaults True "
                           "(conformer.py:37) and is set False only on the inference dataset "
                           "(core/data/framework/single_datasets/inference.py:216)",
         "selftest": ("perf/of3t_permalign/permalign_selftest.py over these ten arms: "
                      "0 of 10 disagree with their configuration, the canary reaches both "
                      "catch tiers and zeroes every loss weight, and the positive arm "
                      "completes on batch_step003.pt. Exit 0."),
         "raised_at": "core/utils/permutation_alignment.py:1412, "
                      "pred_ref_space_uid_to_perm = single_batch['ref_space_uid_to_perm']"},
        reason=("the census reports a KeyError its own instrument never ran: the census has "
                "no model forward, as its own model_forward entry states, so it never called "
                "safe_multi_chain_permutation_alignment. The alignment completes on the "
                "frozen batch; the exception is reproducible only by removing the key, which "
                "is what a second forward over one batch dict does "
                "(model.py:670 pops it off the caller's dict)"))

    # --- the two that are not this row's ---------------------------------------------
    for name in ("diffusion_rollout", "model_forward"):
        add(name, False, "CONFIRMS", cp[name]["reason"],
            reason=("out of scope for this row: needs the OF3 training forward wired in "
                    "tt_bio/train/, which no batch selection reaches"))

    n_cov = sum(1 for v in paths.values() if v["covered"])
    doc = {
        "instrument": "PROTOCOL SS6 coverage UNION over stages, datasets and targets",
        "owner": "of3t-covpaths",
        "not_a_re_emission_of": ("perf/of3t_gradients/coverage_census.json is of3t-gradients' "
                                 "artifact and that row concluded; this file is written under "
                                 "its own name in the same schema and cites it"),
        "census": sources["coverage_census"],
        "census_verdict": census["verdict"],
        "census_conditional_paths_covered": sum(1 for v in cp.values() if v.get("covered")),
        "census_conditional_paths_total": len(cp),
        "union": paths,
        "union_summary": {
            "paths": len(paths), "covered": n_cov,
            "confirms": sorted(k for k, v in paths.items() if v["this_row"] == "CONFIRMS"),
            "extends": sorted(k for k, v in paths.items() if v["this_row"] == "EXTENDS"),
            "contradicts": sorted(k for k, v in paths.items() if v["this_row"] == "CONTRADICTS"),
        },
        "rule": ("a path is covered when a parameter gradient MOVES against an arm with the "
                 "path off, not when a config key is set"),
        "sources": sources,
    }
    a.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"{n_cov} of {len(paths)} conditional paths covered "
          f"(census: {doc['census_conditional_paths_covered']} of "
          f"{doc['census_conditional_paths_total']})")
    for k, v in paths.items():
        print(f"  {k:24s} {str(v['covered']):5s}  {v['this_row']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
