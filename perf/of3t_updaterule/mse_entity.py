#!/usr/bin/env python3
"""D12: `objectives._TERMS['mse']` dropped upstream's per-entity loss weighting. No card.

`losses.mse` implements upstream's 5.0 / 5.0 / 10.0 upweighting of DNA, RNA and ligand
tokens and always has. The adapter in `objectives.py` never passed `is_dna`, `is_rna` or
`is_ligand` to it, so every nucleic-acid and ligand token trained at protein weight. The term
FIRES either way, at the wrong weight, which is why a coverage check that reads the
configured weight cannot see it.

Three arms, on `of3t-reference`'s frozen 5nw3 batch (sha256-checked before it is opened):

  1. DELTA, token scope, through the shipped objective. The same call a training step makes,
     with the adapter as it shipped and as it now is. This is the D12 number, re-measured.
  2. REFERENCE, atom scope, float64. Our `losses.mse` against UPSTREAM'S OWN `mse_loss`
     (`core/loss/diffusion.py:106`) run in float64 on the same batch -- not a transcription
     of it -- with upstream's autograd gradient validated by float64 central finite
     differences first (SS3c) and ours then compared against that.
  3. CONTROL (SS3e). The perturbation a value check cannot survive is not "remove the
     weighting", it is "hand it an all-zero flag": a batch with no ligand and a batch whose
     featuriser never produced the flags give the IDENTICAL loss, and only the breakdown's
     `without` field tells them apart. Both are run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

OUT = "perf/of3t_updaterule"
SCRATCH = "/tmp/of3t/of3t-updaterule"
BATCH = f"{SCRATCH}/batch_step003.pt"
MANIFEST = f"{SCRATCH}/MANIFEST.json"
UPSTREAM = "/home/moritz/.coworker/scratch/of3t-reference/upstream050"
# `of3t-gradients`' census seed, so the prediction this is measured on is the one D12 was
# measured on and the two numbers are comparable rather than merely similar.
SEED = 20260919
EPS = 1e-6
# The figure the published reference itself was accepted at: float64 central differences at
# h = 1e-5 over entries with |analytic| >= 1e-6, max relative 1.08e-04. Not a bar invented
# here, and it feeds the SS3d 5.0e-02 per-tensor bar 460x inside it.
FD_BAR = 1.1e-4


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --------------------------------------------------------------------------- arm 1: delta

def shipped_adapter(losses):
    """`_TERMS['mse']` exactly as it shipped: no entity flags reach `losses.mse`."""
    return lambda b, o: losses.mse(o["pred_xyz"], b["true_xyz"], b["coord_mask"],
                                   per_sample_scale=b.get("edm_scale"))


def arm_delta(labels, outputs, weights):
    from tt_bio.train import losses, objectives
    fixed = objectives._TERMS["mse"]
    out = {}
    for tag, adapter in (("fixed", fixed), ("shipped", shipped_adapter(losses))):
        objectives._TERMS["mse"] = adapter
        try:
            total, breakdown, seeds = objectives.af3_loss(labels, outputs, weights)
        finally:
            objectives._TERMS["mse"] = fixed
        out[tag] = {"total": total, "mse_value": breakdown["mse"]["value"],
                    "mse_contribution": breakdown["mse"]["contribution"],
                    "without": breakdown["mse"].get("without"),
                    "seed": np.asarray(seeds["pred_xyz"], np.float64)}
    f, s = out["fixed"], out["shipped"]
    return {"scope": "token, one representative atom per token (the SS6 census's labels)",
            "value_shipped": s["mse_value"], "value_fixed": f["mse_value"],
            "rel_value": abs(f["mse_value"] - s["mse_value"]) / (abs(f["mse_value"]) + 1e-30),
            "rel_seed": rel(s["seed"], f["seed"]),
            "total_shipped": s["total"], "total_fixed": f["total"],
            "rel_total": abs(f["total"] - s["total"]) / (abs(f["total"]) + 1e-30),
            "breakdown_without_shipped": s["without"],
            "breakdown_without_fixed": f["without"]}


# ----------------------------------------------------------------- arm 2: float64 reference

def their_mse(x_pred, batch64, loss_token_mask):
    """Upstream's own `mse_loss`, float64, autograd-live in `x_pred`."""
    from openfold3.core.loss.diffusion import mse_loss
    return mse_loss(x_pred, batch64, loss_token_mask,
                    dna_weight=5.0, rna_weight=5.0, ligand_weight=10.0, eps=EPS)


def arm_reference(batch, rng_seed=SEED, n_fd=12):
    """Ours against theirs at ATOM scope in float64, theirs validated by central differences.

    Atom scope because that is the scope upstream's loss is written at: it broadcasts the
    per-token weight onto atoms and reduces over atoms. The token-scope census in arm 1 is a
    different reduction of the same formula and is not what this arm compares.
    """
    from tt_bio.train import losses
    sys.path.insert(0, UPSTREAM)
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

    gt = batch["ground_truth"]
    t64 = lambda x: x.to(torch.float64)
    b64 = {"is_dna": t64(batch["is_dna"]), "is_rna": t64(batch["is_rna"]),
           "is_ligand": t64(batch["is_ligand"]), "token_mask": t64(batch["token_mask"]),
           "num_atoms_per_token": batch["num_atoms_per_token"],
           "ground_truth": {"atom_positions": t64(gt["atom_positions"]),
                            "atom_resolved_mask": t64(gt["atom_resolved_mask"])}}
    ntok_pad = b64["token_mask"].shape[-1]
    loss_token_mask = b64["token_mask"].clone()

    nat = b64["ground_truth"]["atom_positions"].shape[-2]
    rng = np.random.default_rng(rng_seed)
    true_atoms = b64["ground_truth"]["atom_positions"][0].numpy()
    pred_np = true_atoms + rng.standard_normal(true_atoms.shape)
    x_pred = torch.tensor(pred_np, dtype=torch.float64)[None].requires_grad_(True)

    v_theirs_t = their_mse(x_pred, b64, loss_token_mask)
    v_theirs = float(v_theirs_t.reshape(()).item())
    (g_theirs,) = torch.autograd.grad(v_theirs_t.sum(), x_pred)
    g_theirs = g_theirs[0].numpy()

    # Ours, at the same scope: the per-token entity flags broadcast onto atoms, which is what
    # upstream's `broadcast_token_feat_to_atoms` does to `w` before the reduction.
    to_atoms = lambda feat: broadcast_token_feat_to_atoms(
        token_mask=b64["token_mask"], num_atoms_per_token=b64["num_atoms_per_token"],
        token_feat=feat)[0].numpy()
    v_ours, g_ours = losses.mse(
        pred_np, true_atoms, b64["ground_truth"]["atom_resolved_mask"][0].numpy(),
        is_dna=to_atoms(b64["is_dna"]), is_rna=to_atoms(b64["is_rna"]),
        is_ligand=to_atoms(b64["is_ligand"]), eps=EPS)

    # SS3c: validate THEIR gradient by float64 central differences before ours is compared
    # to it. Method, step size and sampling rule are the published bundle's own
    # (MANIFEST.finite_difference_validation): h = 1e-5, and only entries with
    # |analytic| >= 1e-6, because a relative error on a near-zero entry is not a measurement.
    # Entries are drawn over the ligand atoms as well as the protein ones -- the entity
    # weighting is the thing under test and a sample that misses it proves nothing.
    lig = set(np.nonzero(to_atoms(b64["is_ligand"]) > 0)[0].tolist())
    pick = list(dict.fromkeys(
        [(int(a), c) for a in sorted(lig) for c in range(3)]
        + [(int(a), int(c)) for a, c in zip(rng.integers(0, nat, 4 * n_fd),
                                            rng.integers(0, 3, 4 * n_fd))]))
    pick = [(a, c) for a, c in pick if abs(float(g_theirs[a, c])) >= 1e-6][:n_fd + len(lig) * 3]

    def quotient(a, c, h):
        d = np.zeros_like(pred_np)
        d[a, c] = h
        up = float(their_mse(torch.tensor(pred_np + d, dtype=torch.float64)[None],
                             b64, loss_token_mask).reshape(()).item())
        dn = float(their_mse(torch.tensor(pred_np - d, dtype=torch.float64)[None],
                             b64, loss_token_mask).reshape(()).item())
        return (up - dn) / (2 * h)

    h = 1e-5
    fd = []
    for a, c in pick:
        num, ana = quotient(a, c, h), float(g_theirs[a, c])
        fd.append({"atom": a, "axis": c, "is_ligand_atom": a in lig,
                   "numeric": num, "analytic": ana,
                   "rel": abs(num - ana) / (abs(num) + 1e-30)})
    # The shape of the sweep on the worst entry, so an h-independent disagreement -- the
    # signature SS3c-bis names -- would be visible rather than hidden behind one step size.
    w = max(fd, key=lambda e: e["rel"])
    sweep = {f"{hh:.0e}": abs(quotient(w["atom"], w["axis"], hh) - w["analytic"])
                          / (abs(w["analytic"]) + 1e-30)
             for hh in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)}

    return {"scope": "atom, 422 atoms, float64 both sides",
            "their_source": f"{UPSTREAM}/openfold3/core/loss/diffusion.py",
            "their_source_sha256": sha256(f"{UPSTREAM}/openfold3/core/loss/diffusion.py"),
            "n_atoms": int(nat), "n_tokens_padded": int(ntok_pad),
            "n_ligand_atoms": len(lig),
            "value_ours": v_ours, "value_theirs": v_theirs,
            "rel_value": abs(v_ours - v_theirs) / (abs(v_theirs) + 1e-30),
            "rel_grad": rel(g_ours, g_theirs),
            "max_elementwise_rel_grad": float(np.max(
                np.abs(g_ours - g_theirs) / (np.abs(g_theirs) + 1e-30))),
            "worst_element": (lambda i: {
                "atom": int(i // 3), "axis": int(i % 3),
                "ours": float(g_ours.ravel()[i]), "theirs": float(g_theirs.ravel()[i])})(
                    int(np.argmax(np.abs(g_ours - g_theirs)))),
            "fd": {"h": h, "entries": fd, "n": len(fd),
                   "sampling_rule": "|analytic| >= 1e-6, the published bundle's own rule",
                   "median_rel": float(np.median([e["rel"] for e in fd])),
                   "worst_rel": float(max(e["rel"] for e in fd)),
                   "worst_entry": w, "h_sweep_on_worst_entry": sweep,
                   "bar": FD_BAR,
                   "bar_source": "the published reference bundle's own max_rel_err at the "
                                 "same h and the same sampling rule "
                                 "(MANIFEST.finite_difference_validation)",
                   "ligand_entries": sum(1 for e in fd if e["is_ligand_atom"])}}


# ------------------------------------------------------------------------- arm 3: control

def arm_control(labels, outputs, weights):
    """SS3e. Three perturbations, and what each one must break."""
    from tt_bio.train import objectives
    run = lambda lab: objectives.af3_loss(lab, outputs, weights)

    base_t, base_b, base_s = run(labels)
    base = {"value": base_b["mse"]["value"], "without": base_b["mse"].get("without"),
            "seed": np.asarray(base_s["pred_xyz"], np.float64)}

    dropped = {k: v for k, v in labels.items() if k not in ("is_dna", "is_rna", "is_ligand")}
    d_t, d_b, d_s = run(dropped)

    zeroed = dict(labels)
    for k in ("is_dna", "is_rna", "is_ligand"):
        zeroed[k] = np.zeros_like(np.asarray(labels[k], np.float64))
    z_t, z_b, z_s = run(zeroed)

    # The zeros question the protocol makes every instrument answer in writing.
    zero_out = dict(outputs)
    zero_out["pred_xyz"] = np.zeros_like(np.asarray(outputs["pred_xyz"], np.float64))
    zp_t, zp_b, zp_s = objectives.af3_loss(labels, zero_out, weights)

    return {
        "baseline": {"value": base["value"], "without": base["without"]},
        "flags_dropped": {
            "what": "the three entity keys removed from the batch, which is the featuriser "
                    "that never produced them",
            "value": d_b["mse"]["value"], "without": d_b["mse"].get("without"),
            "rel_value_vs_baseline": abs(d_b["mse"]["value"] - base["value"])
                                     / (abs(base["value"]) + 1e-30),
            "rel_seed_vs_baseline": rel(d_s["pred_xyz"], base["seed"]),
            "breaks": "the weighting no longer reaches losses.mse, and the breakdown says so"},
        "flags_zeroed": {
            "what": "all three flags present and identically zero, which is a batch with no "
                    "DNA, RNA or ligand",
            "value": z_b["mse"]["value"], "without": z_b["mse"].get("without"),
            "value_equals_dropped": z_b["mse"]["value"] == d_b["mse"]["value"],
            "seed_equals_dropped": rel(z_s["pred_xyz"], d_s["pred_xyz"]) == 0.0,
            "breaks": "NOTHING a value or a seed check reads -- zeroed and dropped are "
                      "numerically identical. Only `without` separates them, which is why "
                      "the fix is not the pass-through alone"},
        "prediction_zeroed": {
            "what": "pred_xyz replaced by zeros, the protocol's standing question",
            "value": zp_b["mse"]["value"],
            "rel_value_vs_baseline": abs(zp_b["mse"]["value"] - base["value"])
                                     / (abs(base["value"]) + 1e-30),
            "seed_norm": float(np.linalg.norm(np.asarray(zp_s["pred_xyz"], np.float64))),
            "baseline_seed_norm": float(np.linalg.norm(base["seed"]))},
        "answers": ("which check fails if the entity weighting is dropped? the value and the "
                    "gradient seed, but ONLY on a batch that has a nucleic-acid or ligand "
                    "token -- on a protein-only batch nothing moves and the defect is "
                    "invisible to every numeric check. `without` is what fails there."),
        "pass": bool(
            d_b["mse"]["value"] != base["value"]
            and d_b["mse"].get("without") == ["is_dna", "is_rna", "is_ligand"]
            and base["without"] is None
            and z_b["mse"]["value"] == d_b["mse"]["value"]
            and z_b["mse"].get("without") is None)}


def main() -> int:
    from tt_bio.train.losses import of3_loss_weights
    sys.path.insert(0, UPSTREAM)
    sys.path.insert(0, "perf/of3t_gradients")
    from coverage_census import labels_from

    man = json.load(open(MANIFEST))
    want = {a["file"]: a.get("sha256") for a in man["artifacts"]}
    got = sha256(BATCH)
    if got != want["batch_step003.pt"]:
        print(f"batch_step003.pt sha256 {got} != declared {want['batch_step003.pt']}; "
              f"refusing to measure"); return 1

    batch = torch.load(BATCH, map_location="cpu", weights_only=False)
    labels, outputs, meta = labels_from(batch, np)
    weights = of3_loss_weights("initial_training", "weighted-pdb")

    rep = {"defect": "D12 -- objectives._TERMS['mse'] dropped upstream's per-entity weighting",
           "fix": "tt_bio/train/objectives.py: pass is_dna/is_rna/is_ligand through, and name "
                  "an absent one in the breakdown",
           "upstream_weights": {"dna": 5.0, "rna": 5.0, "ligand": 10.0,
                                "source": "projects/of3_all_atom/config/model_config.py:"
                                          "496-498 into core/loss/diffusion.py:138-141"},
           "batch": {"file": BATCH, "sha256": got, "pdb_id": batch["pdb_id"], **meta},
           "stage_dataset": "initial_training/weighted-pdb", "torch": torch.__version__}

    rep["arm_delta"] = d = arm_delta(labels, outputs, weights)
    print(f"[delta] mse value {d['value_shipped']:.6f} -> {d['value_fixed']:.6f} "
          f"(rel {d['rel_value']:.4f}); gradient seed rel {d['rel_seed']:.4f}; "
          f"objective total rel {d['rel_total']:.4f}")

    rep["arm_reference"] = r = arm_reference(batch)
    f64_ok = r["rel_value"] <= 1e-12 and r["rel_grad"] <= 1e-12 and r["fd"]["worst_rel"] <= FD_BAR
    rep["arm_reference"]["pass"] = bool(f64_ok)
    print(f"[{'PASS' if f64_ok else 'FAIL'}] float64 reference: value rel {r['rel_value']:.3e}, "
          f"gradient rel {r['rel_grad']:.3e}, worst element rel "
          f"{r['max_elementwise_rel_grad']:.3e}; their gradient against central differences "
          f"median {r['fd']['median_rel']:.3e} worst {r['fd']['worst_rel']:.3e} over "
          f"{len(r['fd']['entries'])} entries, {r['fd']['ligand_entries']} on ligand atoms")

    rep["arm_control"] = c = arm_control(labels, outputs, weights)
    print(f"[{'PASS' if c['pass'] else 'FAIL'}] control: dropped flags move the value "
          f"{c['flags_dropped']['rel_value_vs_baseline']:.4f} and the seed "
          f"{c['flags_dropped']['rel_seed_vs_baseline']:.4f} and report "
          f"{c['flags_dropped']['without']}; zeroed flags are numerically IDENTICAL to "
          f"dropped ({c['flags_zeroed']['value_equals_dropped']}) and report "
          f"{c['flags_zeroed']['without']}")

    ok = f64_ok and c["pass"] and d["rel_value"] > 0.0
    rep["pass"] = bool(ok)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/mse_entity.json", "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(f"\n{'PASS' if ok else 'FAIL'} -- wrote {OUT}/mse_entity.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
