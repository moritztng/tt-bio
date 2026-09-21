#!/usr/bin/env python3
"""PROTOCOL SS6: move a parameter gradient with each conditional path, against a control.

`presence_n4.json` shows the featuriser produces the tensors three NOT COVERED paths need:
a non-zero `template_pseudo_beta_mask` on 1kvu, `is_dna` on 4hj5, `is_rna` on 5kla, and a
summed confidence weight of exactly 0.0 on 5oid. SS6 does not accept that. A term or a path
that fires and moves no gradient has been skipped with extra steps, so what is reported here
is the CONTRIBUTION: an arm with the path on, an arm with it off, and the difference of the
two parameter-gradient vectors.

Three paths, three controls, all on `initial_training` / `weighted-pdb`:

  templates    two forwards. One with the template masks the featuriser built, one with
               `template_pseudo_beta_mask` and `template_backbone_frame_mask` zeroed, which
               is how an empty template slot reaches the embedder. The gradient that goes
               away should be the template embedder's.

  nucleotide   one forward, two losses. Upstream's structure losses weight a token by its
               molecule type, so the control hands the LOSS the same prediction with `is_dna`
               and `is_rna` folded into `is_protein`. The forward is byte-identical between
               the arms, so anything that moves is the nucleotide branch of the loss and
               nothing else. On a protein-only target the same edit is a no-op, which is the
               control's own control.

  disabled     one forward, two losses. 5oid arrives at 4.6 Ang, outside
               `[min_resolution, max_resolution] = [0.1, 4.0]`, so `set_loss_weights` zeroes
               all four confidence weights and `runner._get_sample_disabled_param_names`
               returns the confidence head. The arm that matters is not a magnitude: it is
               that those parameters hold `grad is None` rather than a zero gradient, which
               PROTOCOL 3b makes a compared property. The control restores the four weights
               and the same parameters become populated.

Nothing upstream is modified and no feature is synthesised. A zero difference is the finding.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "of3_port"))
sys.path.insert(0, str(HERE.parent / "of3t_reference"))
sys.path.insert(0, str(HERE.parent / "of3t_auxheads"))

import batch_digest as BD  # noqa: E402
import bundle_min as BM  # noqa: E402
from bond_coverage import collate1, snapshot  # noqa: E402

# runner.py:144-151 verbatim. The set is derived from the model's own parameter names below,
# exactly as `_identify_confidence_params` derives it.
CONFIDENCE_PREFIXES = [
    "aux_heads.pairformer_embedding",
    "aux_heads.pde",
    "aux_heads.plddt",
    "aux_heads.experimentally_resolved",
    "aux_heads.pae",
]
CONF_LOSS_NAMES = ["experimentally_resolved", "plddt", "pae", "pde"]
TEMPLATE_MASKS = ["template_pseudo_beta_mask", "template_backbone_frame_mask"]


def confidence_param_names(model) -> set[str]:
    return {n for n, _ in model.named_parameters()
            if any(n.startswith(p + ".") for p in CONFIDENCE_PREFIXES)}


def zero_template_masks(batch: dict) -> dict:
    n = 0
    for k in TEMPLATE_MASKS:
        if k in batch and torch.is_tensor(batch[k]):
            n += int((batch[k] != 0).sum())
            batch[k] = torch.zeros_like(batch[k])
    return {"masks_zeroed": TEMPLATE_MASKS, "entries_cleared": n}


def fold_nucleotide_into_protein(batch: dict) -> dict:
    moved = 0
    for k in ("is_dna", "is_rna"):
        if k in batch and torch.is_tensor(batch[k]):
            m = batch[k]
            moved += int((m > 0).sum())
            if "is_protein" in batch:
                batch["is_protein"] = torch.clamp(batch["is_protein"] + m, max=1.0)
            batch[k] = torch.zeros_like(m)
    gt = batch.get("ground_truth")
    if isinstance(gt, dict):
        for k in ("is_dna", "is_rna"):
            if k in gt and torch.is_tensor(gt[k]):
                m = gt[k]
                if "is_protein" in gt:
                    gt["is_protein"] = torch.clamp(gt["is_protein"] + m, max=1.0)
                gt[k] = torch.zeros_like(m)
    return {"tokens_refiled_as_protein": moved}


def set_confidence_weights(batch: dict, value: float) -> dict:
    was = {}
    for k in CONF_LOSS_NAMES:
        if k in batch["loss_weights"]:
            was[k] = float(batch["loss_weights"][k])
            batch["loss_weights"][k] = torch.full_like(batch["loss_weights"][k], value)
    return {"was": was, "now": value}


def score(gA, gB, conf_names: set[str]) -> dict:
    totA = totD = 0.0
    per_section: dict[str, dict] = {}
    moved = n_params = 0
    for n, a in gA.items():
        if a is None:
            continue
        n_params += 1
        b = gB.get(n)
        a64 = a.double()
        d = a64 if b is None else a64 - b.double()
        sa = float((a64 ** 2).sum())
        sd = float((d ** 2).sum())
        totA += sa
        totD += sd
        if b is None or not torch.equal(a, b):
            moved += 1
        sec = n.split(".")[0]
        e = per_section.setdefault(sec, {"sq": 0.0, "delta_sq": 0.0, "n": 0})
        e["sq"] += sa
        e["delta_sq"] += sd
        e["n"] += 1
    for e in per_section.values():
        e["share_of_own"] = e["delta_sq"] / e["sq"] if e["sq"] else None
    return {
        "squared_norm_arm_on": totA,
        "squared_norm_of_difference": totD,
        "share_of_squared_gradient_norm": totD / totA if totA else None,
        "n_params_moved": moved,
        "n_params_with_gradient_on": n_params,
        "by_section": per_section,
        "confidence_head": {
            "n_params": len(conf_names),
            "n_grad_none_on": sum(1 for n in conf_names if gA.get(n) is None),
            "n_grad_none_off": sum(1 for n in conf_names if gB.get(n) is None),
            "n_grad_zero_on": sum(1 for n in conf_names
                                  if gA.get(n) is not None and float(gA[n].abs().sum()) == 0.0),
            "n_grad_zero_off": sum(1 for n in conf_names
                                   if gB.get(n) is not None and float(gB[n].abs().sum()) == 0.0),
        },
    }


def prefix_scores(gA, gB, prefixes: list[str]) -> dict:
    out = {}
    for p in prefixes:
        sa = sd = 0.0
        k = 0
        for n, a in gA.items():
            if a is None or not n.startswith(p):
                continue
            b = gB.get(n)
            a64 = a.double()
            d = a64 if b is None else a64 - b.double()
            sa += float((a64 ** 2).sum())
            sd += float((d ** 2).sum())
            k += 1
        out[p] = {"n": k, "sq": sa, "delta_sq": sd,
                  "share_of_own": sd / sa if sa else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, choices=("templates", "nucleotide", "disabled"))
    ap.add_argument("--package", default="openfold3")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--cache-file", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--rank-template", type=Path, required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()

    BD.seed_everything(a.seed)
    ds = BD.build_dataset(a.package, a.data_dir, a.n_templates, token_budget=a.crop,
                          split="train", stage=a.stage, cache_file=a.cache_file)
    guard = BD.install_retry_guard(ds)
    dp = ds.datapoint_cache.iloc[a.index]
    sample = ds[a.index]
    if guard["retries"]:
        raise SystemExit(f"{guard['retries']} silent sample substitutions; this is not the "
                         f"target it claims to be")

    tok = sample["token_mask"].bool()
    counts = {k: int((sample[k][tok] > 0).sum())
              for k in ("is_protein", "is_dna", "is_rna", "is_ligand") if k in sample}
    tpl_nnz = int((sample["template_pseudo_beta_mask"] != 0).sum()) \
        if "template_pseudo_beta_mask" in sample else None
    lw = {k: float(v) for k, v in sample["loss_weights"].items()
          if torch.is_tensor(v) and v.numel() == 1}
    conf_sum = sum(lw[n] for n in CONF_LOSS_NAMES if n in lw)
    target = {"pdb_id": str(dp["pdb_id"]),
              "datapoint": str(dp["preferred_chain_or_interface"]),
              "n_tokens_real": int(tok.sum()), "molecule_type_tokens": counts,
              "template_pseudo_beta_mask_nnz": tpl_nnz,
              "loss_weights": lw, "summed_confidence_weight": conf_sum}
    print(f"target {a.index} {target['pdb_id']} {target['datapoint']}: "
          f"tokens {target['n_tokens_real']} {counts} tpl {tpl_nnz} conf_sum {conf_sum}",
          flush=True)

    # A31: refuse the run if the arm the path needs cannot differ on this batch.
    if a.path == "templates" and not tpl_nnz:
        raise SystemExit("template mask is already zero on this batch; the control cannot move")
    if a.path == "nucleotide" and counts.get("is_dna", 0) + counts.get("is_rna", 0) == 0:
        raise SystemExit("no nucleotide token on this batch; the control cannot move")
    if a.path == "disabled" and conf_sum > 0:
        raise SystemExit(f"summed confidence weight is {conf_sum}, so the disable gate does "
                         f"not fire on this batch; pick a target outside [0.1, 4.0] Ang")

    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    BM.pin_deterministic_kernels(True)
    cfg, model, loss_fn, dropout = BM.build(dtype, a.seed, "cpu", 0)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE FAILED: {len(inc.unexpected_keys)} unexpected tensors")
    del ck, sd
    conf_names = confidence_param_names(model)
    print(f"[{time.time()-t0:.0f}s] model built; {len(conf_names)} confidence-head parameters",
          flush=True)

    tmpl = torch.load(a.rank_template, weights_only=False)
    base = BM.move(collate1(sample, tmpl), "cpu", dtype)
    pinned = BM.rng_state(model)
    ctx = BM.no_autocast() if dtype is torch.float64 else BM._null()

    arms: dict[str, dict] = {}
    grads: dict[str, dict] = {}

    def run(tag: str, edit_batch=None, edit_loss=None, shared=None):
        """One arm: forward, loss, backward, gradient snapshot."""
        if shared is None:
            b_in = copy.deepcopy(base)
            if edit_batch is not None:
                note = edit_batch(b_in)
            else:
                note = {}
            BM.set_rng_state(pinned, model)
            with ctx:
                b, out = model(b_in)
            print(f"[{time.time()-t0:.0f}s] {tag}: forward done", flush=True)
        else:
            b, out, note = shared[0], shared[1], {}
            b = copy.deepcopy(b)
        if edit_loss is not None:
            note = {**note, **edit_loss(b)}
        BM.set_rng_state(pinned, model)
        with ctx:
            loss, bd = loss_fn(b, out, _return_breakdown=True)
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=shared is not None and tag.endswith("_on"))
        grads[tag] = snapshot(model)
        arms[tag] = {
            "loss": float(loss), "edit": note,
            "breakdown": {k: float(v) for k, v in bd.items()
                          if torch.is_tensor(v) and v.numel() == 1},
        }
        print(f"[{time.time()-t0:.0f}s] {tag}: loss {float(loss):.9f}", flush=True)

    if a.path == "templates":
        run("templates_on")
        run("templates_off", edit_batch=zero_template_masks)
        on, off = "templates_on", "templates_off"
    elif a.path == "nucleotide":
        b_in = copy.deepcopy(base)
        BM.set_rng_state(pinned, model)
        with ctx:
            shared = model(b_in)
        print(f"[{time.time()-t0:.0f}s] shared forward done", flush=True)
        run("nucleotide_on", shared=shared)
        run("nucleotide_off", edit_loss=fold_nucleotide_into_protein, shared=shared)
        on, off = "nucleotide_on", "nucleotide_off"
    else:
        b_in = copy.deepcopy(base)
        BM.set_rng_state(pinned, model)
        with ctx:
            shared = model(b_in)
        print(f"[{time.time()-t0:.0f}s] shared forward done", flush=True)
        run("disabled_on", shared=shared)
        run("disabled_off", edit_loss=lambda b: set_confidence_weights(b, 1e-4), shared=shared)
        on, off = "disabled_on", "disabled_off"

    s = score(grads[on], grads[off], conf_names)
    report = {
        "instrument": "PROTOCOL SS6: conditional-path gradient contribution against a control",
        "path": a.path, "stage": a.stage, "dataset": "weighted-pdb", "crop": a.crop,
        "index": a.index, "dtype": a.dtype, "seed": a.seed, "n_templates": a.n_templates,
        "package": a.package, "cache_file": str(a.cache_file),
        "checkpoint": str(a.checkpoint),
        "target": target, "arms": arms,
        "contribution": s,
        "confidence_param_names_n": len(conf_names),
        "template_embedder": prefix_scores(grads[on], grads[off], ["template_embedder"])
        if a.path == "templates" else None,
        "dropout": dropout,
        "total_s": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\n{a.path}: share of squared gradient norm "
          f"{s['share_of_squared_gradient_norm']}, {s['n_params_moved']} of "
          f"{s['n_params_with_gradient_on']} tensors moved")
    print(f"  confidence head: grad None on/off "
          f"{s['confidence_head']['n_grad_none_on']}/{s['confidence_head']['n_grad_none_off']} "
          f"of {s['confidence_head']['n_params']}")
    for sec, e in sorted(s["by_section"].items(), key=lambda kv: -kv[1]["delta_sq"])[:8]:
        print(f"  {sec:26s} delta_sq {e['delta_sq']:.6e}  share of its own {e['share_of_own']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
