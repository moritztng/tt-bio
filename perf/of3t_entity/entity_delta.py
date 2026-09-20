#!/usr/bin/env python3
"""D16 arms 2-5: the delta per model, the SS3e control, the other four, and inference. No card.

Reuses `perf/of3t_updaterule/mse_entity.py` rather than writing a second harness: `rel`,
`sha256` and the whole float64 arm (ours against UPSTREAM'S OWN `mse_loss`, their gradient
validated by central differences first) are imported from it, and the D12 numbers reported
here are that module's, re-run.

WHAT IS REAL AND WHAT IS SYNTHETIC, said once. OpenFold3 is the only model with a training
batch in existence (`of3t-reference`'s frozen 5nw3, sha256-checked before it is opened), so
its delta is a measurement. The other three have no training batch and no registered training
adapter -- `tt_bio.train.catalogue` ships empty -- so their coordinates are synthetic. Their
`mol_type` column is NOT: it is the real column their own featuriser emitted in
`mapping.py`, which is the half of the batch this defect is about.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "perf/of3t_updaterule")
sys.path.insert(0, "perf/of3t_gradients")

import mse_entity as U                                          # noqa: E402

OUT = "perf/of3t_entity"
SCRATCH = "/tmp/of3t/of3t-entity"
SEED = U.SEED
ENTITY = ("is_dna", "is_rna", "is_ligand")
# Upstream's own weights, `core/loss/diffusion.py:138-141` at `model_config.py:496-498`.
W_DNA, W_RNA, W_LIG = 5.0, 5.0, 10.0


def _mol_type_from_flags(flags, convention) -> np.ndarray:
    """The real entity flags of a real batch, re-expressed as a stack's integer column."""
    from tt_bio.train.objectives import MOL_TYPE_CONVENTIONS
    t = MOL_TYPE_CONVENTIONS[convention]
    mt = np.full(len(flags["is_dna"]), t["protein"], dtype=np.int64)
    for k in ENTITY:
        mt[np.asarray(flags[k]) > 0] = t[k[3:]]
    return mt


def _run(labels, outputs, weights):
    from tt_bio.train import objectives
    total, breakdown, seeds = objectives.af3_loss(labels, outputs, weights)
    return {"total": total, "value": breakdown["mse"]["value"],
            "without": breakdown["mse"].get("without"),
            "derived": breakdown["mse"].get("derived"),
            "seed": np.asarray(seeds["pred_xyz"], np.float64)}


def _delta(before, after) -> dict:
    return {"value_before": before["value"], "value_after": after["value"],
            "rel_value": abs(after["value"] - before["value"])
                         / (abs(after["value"]) + 1e-30),
            "rel_seed": U.rel(before["seed"], after["seed"]),
            "rel_total": abs(after["total"] - before["total"])
                         / (abs(after["total"]) + 1e-30),
            "without_before": before["without"], "without_after": after["without"],
            "derived_after": after["derived"]}


# ----------------------------------------------------------------- arm 2a: the real batch

def arm_delta_openfold3(labels, outputs, weights) -> dict:
    """REAL. The adapter on the frozen 5nw3 batch, against its own native flags.

    OpenFold3 does not need the adapter -- its featuriser emits the three flags and D12
    already routed them. That makes it the one place the adapter can be checked against a
    known-correct answer on real data: strip the flags, hand it the same fact as a
    `mol_type` column in each convention, and the loss and the gradient seed must come back
    BIT-IDENTICAL, not merely close.
    """
    native = _run(labels, outputs, weights)
    bare = {k: v for k, v in labels.items() if k not in ENTITY}
    dropped = _run(bare, outputs, weights)

    per_convention = {}
    for conv in ("af3", "boltz"):
        mt = _mol_type_from_flags(labels, conv)
        got = _run({**bare, "mol_type": mt, "mol_type_convention": conv}, outputs, weights)
        per_convention[conv] = {
            "value": got["value"], "derived": got["derived"],
            "bit_identical_value": got["value"] == native["value"],
            "bit_identical_seed": bool(np.array_equal(got["seed"], native["seed"])),
            "rel_value_vs_native": abs(got["value"] - native["value"])
                                   / (abs(native["value"]) + 1e-30),
            "rel_seed_vs_native": U.rel(got["seed"], native["seed"])}
    # And unnamed, which is the state every featuriser we ship is in.
    unnamed = _run({**bare, "mol_type": _mol_type_from_flags(labels, "af3")},
                   outputs, weights)

    return {"label": "REAL -- of3t-reference's frozen 5nw3 batch, token scope",
            "native_flags": {"value": native["value"], "without": native["without"]},
            "flags_dropped": _delta(dropped, native),
            "derived_from_mol_type": per_convention,
            "unnamed_convention": {
                "value": unnamed["value"],
                "bit_identical_to_native": unnamed["value"] == native["value"]
                                           and bool(np.array_equal(unnamed["seed"],
                                                                   native["seed"])),
                "derived": unnamed["derived"]},
            "pass": bool(all(c["bit_identical_value"] and c["bit_identical_seed"]
                             for c in per_convention.values()))}


# --------------------------------------------------- arm 2b: the three mol_type stacks

def _synthetic_coords(n, seed):
    rng = np.random.default_rng(seed)
    true_xyz = rng.standard_normal((n, 3)) * 5.0
    return true_xyz, true_xyz + rng.standard_normal((n, 3))


def arm_delta_moltype_stacks(mapping, weights) -> dict:
    """Per stack, on that stack's OWN measured mol_type column."""
    out, columns = {}, {}
    for stack, conv in (("protenix-v2", "af3"), ("boltz-2", "boltz"),
                        ("boltzgen", "boltz")):
        m = mapping["stacks"][stack]
        obs, n = m["observed"], m["n_tokens"]
        # Reconstruct the exact column mapping.py measured: the spans are in its report and
        # the class ids are that stack's own.
        counts = m["span_counts"]
        mt = np.concatenate([np.full(c, obs[cls], dtype=np.int64)
                             for cls, c in counts])
        assert len(mt) == n, (len(mt), n)
        n_lig = int((mt == obs["ligand"]).sum())
        assert n_lig > 0, f"{stack}: a control batch with no ligand proves nothing"
        true_xyz, pred_xyz = _synthetic_coords(n, SEED)
        base = {"true_xyz": true_xyz, "coord_mask": np.ones(n)}
        outputs = {"pred_xyz": pred_xyz}

        shipped = _run(base, outputs, weights)                   # mol_type absent entirely
        with_mt = _run({**base, "mol_type": mt}, outputs, weights)
        named = _run({**base, "mol_type": mt, "mol_type_convention": conv},
                     outputs, weights)
        out[stack] = {
            "label": "REAL mol_type column from this stack's own featuriser, SYNTHETIC "
                     "coordinates -- no training batch exists for this model and "
                     "tt_bio.train.catalogue ships empty",
            "convention": conv, "n_tokens": n, "n_dna": int((mt == obs["dna"]).sum()),
            "n_rna": int((mt == obs["rna"]).sum()), "n_ligand": n_lig,
            "delta": _delta(shipped, with_mt),
            "naming_the_convention_changes_nothing": bool(
                named["value"] == with_mt["value"]
                and np.array_equal(named["seed"], with_mt["seed"])),
            "resolved_when_named": named["derived"]["resolved"],
            "resolved_when_unnamed": with_mt["derived"]["resolved"]}
        columns[stack] = mt
    # The disagreement, in the data rather than in the two tables. Same complex, same chain
    # order, and protenix-v2's column differs from boltz-2's on exactly the nucleic-acid
    # tokens -- yet the loss and the seed are bit-identical, because upstream weights dna
    # and rna the same. That equality is the invariance the unnamed-convention default
    # rests on, measured here on real columns instead of asserted.
    a, b = columns["protenix-v2"], columns["boltz-2"]
    n = min(len(a), len(b))
    out["_convention_disagreement_in_the_data"] = {
        "columns_elementwise_equal": bool(np.array_equal(a[:n], b[:n])),
        "n_tokens_that_differ": int((a[:n] != b[:n]).sum()),
        "they_differ_only_on_nucleic_acid_tokens": bool(
            set(np.unique(np.concatenate([a[:n][a[:n] != b[:n]],
                                          b[:n][a[:n] != b[:n]]])).tolist()) <= {1, 2}),
        "loss_is_identical_anyway": bool(
            out["protenix-v2"]["delta"]["value_after"]
            == out["boltz-2"]["delta"]["value_after"]),
        "why": "upstream weights dna and rna both at 5.0, so swapping them leaves w "
               "unchanged. The ligand class, which is the one weighted differently at "
               "10.0, is at 3 in both conventions"}
    return out


# -------------------------------------------------------------------- arm 3: the control

def arm_control(mapping, weights) -> dict:
    """SS3e, and the perturbation is chosen so it CANNOT pass for the wrong reason.

    The trap `of3t-updaterule` established: flags present-and-identically-zero are
    bit-identical to flags absent, in both value and seed. So a numeric check cannot tell "no
    ligand in this batch" from "this featuriser never produced the flags", and a control run
    on a protein-only batch passes while seeing nothing. This one runs on a batch with 31
    ligand tokens and asserts it.

    The perturbation breaks what the check READS: the MAPPING, not the plumbing. Ligand is
    re-pointed at the rna class, so `losses.mse` weights those tokens 1+5 instead of 1+10.
    """
    from tt_bio.train import losses, objectives
    m = mapping["stacks"]["boltz-2"]
    obs = m["observed"]
    mt = np.concatenate([np.full(c, obs[cls], dtype=np.int64)
                         for cls, c in m["span_counts"]])
    n = len(mt)
    n_lig = int((mt == obs["ligand"]).sum())
    assert n_lig > 0, "SS3e control on a batch with no ligand is the K54 failure"
    true_xyz, pred_xyz = _synthetic_coords(n, SEED)
    cm = np.ones(n)
    batch = {"true_xyz": true_xyz, "coord_mask": cm, "mol_type": mt,
             "mol_type_convention": "boltz"}
    outputs = {"pred_xyz": pred_xyz}

    good = objectives.entity_flags(mt, "boltz")["flags"]
    # The perturbation: ligand read as rna. One class id changed, nothing else.
    bad = dict(good)
    bad["is_rna"] = np.maximum(good["is_rna"], good["is_ligand"])
    bad["is_ligand"] = np.zeros_like(good["is_ligand"])

    w_good = 1.0 + W_DNA * good["is_dna"] + W_RNA * good["is_rna"] + W_LIG * good["is_ligand"]
    w_bad = 1.0 + W_DNA * bad["is_dna"] + W_RNA * bad["is_rna"] + W_LIG * bad["is_ligand"]

    # The prediction is made EXACT by freezing the alignment. `losses.mse` aligns with a
    # weighted Kabsch, so the weights move the rotation too and the total move is not a pure
    # weight ratio. Frozen, the loss is sum(w_i |d_i|^2) / denom and the seed is
    # proportional to w_i d_i, so both predictions are closed form and either matches to
    # float64 roundoff or the check is wrong.
    real_align = losses.weighted_rigid_align
    frozen = {}

    def freeze(x, x_target, atom_weight):
        if "a" not in frozen:
            frozen["a"] = real_align(x, x_target, atom_weight)
        return frozen["a"]

    losses.weighted_rigid_align = freeze
    try:
        v_good, g_good = losses.mse(pred_xyz, true_xyz, cm, **good)
        v_bad, g_bad = losses.mse(pred_xyz, true_xyz, cm, **bad)
        aligned = frozen["a"][0]
    finally:
        losses.weighted_rigid_align = real_align
    diff = pred_xyz - aligned
    sq = (diff ** 2).sum(-1)
    denom = float(cm.sum()) + U.EPS
    pred_v_good = (1 / 3) * float((w_good * cm * sq).sum()) / denom
    pred_v_bad = (1 / 3) * float((w_bad * cm * sq).sum()) / denom
    pred_ratio = pred_v_bad / pred_v_good
    # The seed ratio is the weight ratio, per token, exactly.
    seed_ratio = np.divide(np.linalg.norm(g_bad, axis=-1),
                           np.linalg.norm(g_good, axis=-1),
                           out=np.ones(n), where=np.linalg.norm(g_good, axis=-1) > 0)
    want_ratio = (w_bad * cm) / (w_good * cm)

    # The absent-flag detector, which is the half no number can do.
    bare = {"true_xyz": true_xyz, "coord_mask": cm}
    b_absent = _run(bare, outputs, weights)
    zeroed = {**bare, "is_dna": np.zeros(n), "is_rna": np.zeros(n),
              "is_ligand": np.zeros(n)}
    b_zeroed = _run(zeroed, outputs, weights)
    b_derived = _run(batch, outputs, weights)

    return {
        "batch": f"boltz-2's own measured mol_type column, {n} tokens, {n_lig} ligand, "
                 f"synthetic coordinates",
        "n_ligand_tokens": n_lig,
        "perturbation": {
            "what": "ligand re-pointed at the rna class, so those tokens weigh 1+5.0 "
                    "instead of 1+10.0. The MAPPING is broken, not the plumbing",
            "alignment": "frozen, so the prediction is closed form; unfrozen the weighted "
                         "Kabsch moves too and the total is not a pure weight ratio",
            "value_correct_mapping": v_good, "value_broken_mapping": v_bad,
            "measured_value_ratio": v_bad / v_good, "predicted_value_ratio": pred_ratio,
            "value_prediction_rel_error": abs((v_bad / v_good) - pred_ratio)
                                          / abs(pred_ratio),
            "measured_value_rel_move": abs(v_bad - v_good) / abs(v_good),
            "seed_rel_move": U.rel(g_bad, g_good),
            "per_token_seed_ratio_matches_weight_ratio": bool(
                np.allclose(seed_ratio, want_ratio, rtol=0, atol=1e-12)),
            "worst_per_token_seed_ratio_error": float(
                np.max(np.abs(seed_ratio - want_ratio))),
            "weight_ratio_on_a_ligand_token": float(
                (1 + W_RNA) / (1 + W_LIG))},
        "absent_flag_detector": {
            "what": "the check no number can do -- a protein-only batch and a featuriser "
                    "that never produced the flags are bit-identical",
            "absent": {"value": b_absent["value"], "without": b_absent["without"]},
            "zeroed": {"value": b_zeroed["value"], "without": b_zeroed["without"]},
            "zeroed_is_bit_identical_to_absent": bool(
                b_zeroed["value"] == b_absent["value"]),
            "derived_from_mol_type": {"value": b_derived["value"],
                                      "without": b_derived["without"],
                                      "derived": b_derived["derived"]},
            "what_separates_them": "`without` names the absent labels and fires only when "
                                   "neither the flags nor mol_type are there; `derived` "
                                   "fires only when the flags came from a mol_type column, "
                                   "so a native batch, a derived one and a blind one are "
                                   "three distinguishable reports and not one number"},
        "answers": "which check fails if the mapping is wrong? the value moves "
                   f"{abs(v_bad - v_good) / abs(v_good):.4f} and the seed "
                   f"{U.rel(g_bad, g_good):.4f}, both by the ratio the weights predict. "
                   "Which fails if the model is replaced by zeros? the value collapses to "
                   "the zero-prediction loss and the seed norm with it, measured in "
                   "of3t-updaterule's own control. Which fails if the featuriser produced "
                   "nothing? ONLY `without` -- no number moves, and that is why the "
                   "reporting half is not cosmetic.",
        "pass": bool(
            abs((v_bad / v_good) - pred_ratio) / abs(pred_ratio) < 1e-12
            and np.allclose(seed_ratio, want_ratio, rtol=0, atol=1e-12)
            and v_bad != v_good
            and b_absent["without"] == list(ENTITY)
            and b_zeroed["without"] is None
            and b_zeroed["value"] == b_absent["value"]
            and b_derived["without"] is None and b_derived["derived"] is not None)}


# --------------------------------------------------------------------------- arm 4: af2

def arm_af2(weights) -> dict:
    """AF2 folds proteins, so the weighting is VACUOUS. Measured, not asserted.

    Vacuous has a testable meaning: every AF2 token is protein, so an all-protein `mol_type`
    column derives three identically-zero flags, and a zero flag contributes nothing to
    `w = 1 + 5 is_dna + 5 is_rna + 10 is_ligand`. The loss and the gradient seed must come
    back BIT-IDENTICAL to the bare batch. `without` firing on the bare batch is then the
    correct report rather than a miss -- it says the labels are absent, which they are.
    """
    n = 64
    true_xyz, pred_xyz = _synthetic_coords(n, SEED)
    base = {"true_xyz": true_xyz, "coord_mask": np.ones(n)}
    outputs = {"pred_xyz": pred_xyz}
    bare = _run(base, outputs, weights)
    all_protein = _run({**base, "mol_type": np.zeros(n, dtype=np.int64)}, outputs, weights)
    zeroed = _run({**base, "is_dna": np.zeros(n), "is_rna": np.zeros(n),
                   "is_ligand": np.zeros(n)}, outputs, weights)
    return {
        "label": "SYNTHETIC coordinates; AF2 carries no molecule-type feature at all, so "
                 "there is no real column to take",
        "n_tokens": n,
        "bare": {"value": bare["value"], "without": bare["without"]},
        "all_protein_mol_type": {"value": all_protein["value"],
                                 "derived": all_protein["derived"],
                                 "bit_identical_to_bare": bool(
                                     all_protein["value"] == bare["value"]
                                     and np.array_equal(all_protein["seed"],
                                                        bare["seed"]))},
        "flags_present_and_zero": {"value": zeroed["value"],
                                   "without": zeroed["without"],
                                   "bit_identical_to_bare": bool(
                                       zeroed["value"] == bare["value"]
                                       and np.array_equal(zeroed["seed"], bare["seed"]))},
        "weighting_is_vacuous": bool(all_protein["value"] == bare["value"]
                                     and zeroed["value"] == bare["value"]),
        "without_firing_is_correct": bare["without"] == list(ENTITY),
        "pass": bool(all_protein["value"] == bare["value"]
                     and np.array_equal(all_protein["seed"], bare["seed"])
                     and bare["without"] == list(ENTITY))}


# ------------------------------------------------------------------ arm 5: inference reach

def arm_inference_reach() -> dict:
    """A fresh interpreter per model. A previous import in this one makes every answer yes."""
    probe = ("import importlib,sys,json;"
             "importlib.import_module(sys.argv[1]);"
             "print(json.dumps(sorted(m for m in sys.modules "
             "if m.startswith('tt_bio.train'))))")
    mods = {"openfold3": "tt_bio.openfold3_fold", "protenix-v2": "tt_bio.protenix",
            "boltz-2": "tt_bio.boltz2", "boltzgen": "tt_bio.boltzgen",
            "af2": "tt_bio.af2"}
    out = {}
    for model, mod in mods.items():
        r = subprocess.run([sys.executable, "-c", probe, mod], capture_output=True,
                           text=True, cwd=os.getcwd(), timeout=900)
        if r.returncode != 0:
            out[model] = {"module": mod,
                          "import_failed": r.stderr.strip().splitlines()[-1:]}
            continue
        loaded = json.loads(r.stdout.strip().splitlines()[-1])
        out[model] = {"module": mod, "tt_bio_train_modules_loaded": loaded,
                      "n_loaded": len(loaded),
                      "reaches_objectives": "tt_bio.train.objectives" in loaded}
    return out


def main() -> int:
    from coverage_census import labels_from
    from tt_bio.train.losses import of3_loss_weights

    sys.path.insert(0, U.UPSTREAM)
    mapping = json.load(open(f"{OUT}/mapping.json"))
    if not mapping["pass"]:
        print("mapping.json did not pass; run perf/of3t_entity/mapping.py first"); return 1

    got = U.sha256(U.BATCH)
    want = {a["file"]: a.get("sha256") for a in json.load(open(U.MANIFEST))["artifacts"]}
    if got != want["batch_step003.pt"]:
        print(f"batch sha256 {got} != declared; refusing to measure"); return 1
    batch = torch.load(U.BATCH, map_location="cpu", weights_only=False)
    labels, outputs, meta = labels_from(batch, np)
    weights = of3_loss_weights("initial_training", "weighted-pdb")

    rep = {"defect": "D16 -- protenix-v2, boltz-2 and boltzgen emit `mol_type`, so the "
                     "entity weighting D12 restored never reached them",
           "fix": "tt_bio/train/objectives.py: one mol_type -> three-flag derivation in "
                  "af3_loss, the site every model's batch already takes",
           "upstream_weights": {"dna": W_DNA, "rna": W_RNA, "ligand": W_LIG},
           "batch": {"file": U.BATCH, "sha256": got, "pdb_id": batch["pdb_id"], **meta},
           "torch": torch.__version__}

    rep["arm_reference"] = r = U.arm_reference(batch)
    f64_ok = (r["rel_value"] <= 1e-12 and r["rel_grad"] <= 1e-12
              and r["fd"]["worst_rel"] <= U.FD_BAR)
    r["pass"] = bool(f64_ok)
    print(f"[{'PASS' if f64_ok else 'FAIL'}] float64 reference (of3t-updaterule's arm, "
          f"re-run): value rel {r['rel_value']:.3e}, gradient rel {r['rel_grad']:.3e}, "
          f"worst element {r['max_elementwise_rel_grad']:.3e}")

    rep["delta_openfold3"] = o = arm_delta_openfold3(labels, outputs, weights)
    for conv, d in o["derived_from_mol_type"].items():
        print(f"[delta] openfold3 REAL: mol_type/{conv} -> value bit-identical to native "
              f"{d['bit_identical_value']}, seed bit-identical {d['bit_identical_seed']}")

    rep["delta_moltype_stacks"] = s = arm_delta_moltype_stacks(mapping, weights)
    for stack, d in s.items():
        if stack.startswith("_"):
            continue
        print(f"[delta] {stack:12s} REAL mol_type + synthetic coords: "
              f"{d['n_tokens']} tokens, {d['n_ligand']} ligand -> value "
              f"{d['delta']['value_before']:.6f} to {d['delta']['value_after']:.6f} "
              f"(rel {d['delta']['rel_value']:.4f}), seed rel "
              f"{d['delta']['rel_seed']:.4f}")

    rep["af2"] = a = arm_af2(weights)
    print(f"[delta] af2          SYNTHETIC: no molecule-type feature "
          f"({len(mapping['stacks']['af2']['molecule_type_keys'])} of "
          f"{mapping['stacks']['af2']['n_string_keys_scanned']} string keys across its four "
          f"modules); an all-protein mol_type column is bit-identical to the bare batch "
          f"({a['all_protein_mol_type']['bit_identical_to_bare']}), so the weighting is "
          f"vacuous and without={a['bare']['without']} is the correct report")

    rep["control"] = c = arm_control(mapping, weights)
    p = c["perturbation"]
    print(f"[{'PASS' if c['pass'] else 'FAIL'}] control: ligand mapped to the rna weight "
          f"moves the value {p['measured_value_rel_move']:.4f} and the seed "
          f"{p['seed_rel_move']:.4f}; measured value ratio {p['measured_value_ratio']:.12f} "
          f"vs predicted {p['predicted_value_ratio']:.12f} "
          f"(rel {p['value_prediction_rel_error']:.2e}); per-token seed ratio matches the "
          f"weight ratio: {p['per_token_seed_ratio_matches_weight_ratio']}")

    rep["inference"] = i = arm_inference_reach()
    for model, d in i.items():
        print(f"[gate] {model:12s} inference import -> "
              f"{d.get('tt_bio_train_modules_loaded', d.get('import_failed'))}")

    ok = (f64_ok and o["pass"] and c["pass"] and a["pass"]
          and all(d["delta"]["rel_value"] > 0
                  for k, d in s.items() if not k.startswith("_"))
          and all(d.get("n_loaded") == 0 for d in i.values()))
    rep["pass"] = bool(ok)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/entity_delta.json", "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(f"\n{'PASS' if ok else 'FAIL'} -- wrote {OUT}/entity_delta.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
