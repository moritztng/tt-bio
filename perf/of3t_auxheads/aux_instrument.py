#!/usr/bin/env python3
"""Instrument A at `aux_heads` scope, against the 0.4.3 reference, on its own boundary.

PROTOCOL A18 s first clause orders this script: the FORWARD is measured first, per head,
against upstream 0.4.3 s own float64 outputs at the boundary the capture recorded. A
disagreeing forward invalidates the gradient taken at it. An agreeing one clears nothing --
it removes mis-wiring from the list and does not bound the gradient error (A18 addendum,
D9: a 3.2x gradient change once hid under a 12 %% forward change on this very model).

The boundary comes from `capture_boundary.py`, which hooked `model.aux_heads` in a float64
CPU run of upstream 0.4.3 on BUNDLE-MIN-043 s own batch and replayed draws, and whose loss
and parameter gradients reproduce the published reference. Inputs, reference outputs and
the cotangents that seed our backward all come from one run of the reference, so nothing of
ours sits upstream of any of them.

The parameter bijection is `grad_device.tape_parameters`, `of3t-confidence` s, reused.

The crop is 384 tokens carrying 56 real ones. Our `forward_device` passes no mask to the
confidence Pairformer where upstream passes `single_mask` and `pair_mask`, so every pair
figure is reported twice: over the whole padded tensor, and over the real-token block
alone. D28 found the trunk s forward gap living entirely in the crop padding; this is the
same question asked at this boundary, and reporting only one of the two numbers would
answer it by assumption.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "of3t_confidence"))

PAIR_HEADS = ["pae_logits", "pde_logits", "distogram_logits"]
ATOM_HEADS = ["plddt_logits", "experimentally_resolved_logits"]
HEADS = ATOM_HEADS + PAIR_HEADS
PER_TENSOR_BAR = 5.0e-2
MEDIAN_BAR = 2.0e-2
A14_FLOOR_RATIO = 1e-8      # of the compared populations own median reference norm


def rel_l2(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def squeeze_leading(t, rank):
    """Drop leading singleton dims down to `rank`. Fails loudly if one is not singleton."""
    while t.dim() > rank:
        if t.shape[0] != 1:
            raise SystemExit(f"leading dim {t.shape[0]} is not 1; this boundary has a real "
                             f"sample or batch axis and the instrument must loop it")
        t = t[0]
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt")))
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--perturb", default=None,
                    help="NAME:FACTOR -- scale one parameters device gradient. SS3e, with "
                         "A14s sizing rule: the factor is chosen against the MEASURED "
                         "baseline and the bar, not fixed at 1 percent.")
    ap.add_argument("--zero-model", action="store_true",
                    help="replace every device gradient by zeros, so the zero-model answer "
                         "is measured rather than assumed")
    ap.add_argument("--reference-grads", type=Path,
                    help="grads_f64_043.pt, for the aux_heads norm-share denominator")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.perf_counter()

    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    import grad_device as GD
    from openfold3.core.utils.atomize_utils import (
        broadcast_token_feat_to_atoms, get_token_representative_atoms,
        max_atom_per_token_masked_select)

    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    kw, pos = B["inputs"]["kwargs"], B["inputs"]["args"]
    batch = kw["batch"] if "batch" in kw else pos[0]
    si_input = kw["si_input"] if "si_input" in kw else pos[1]
    outd = kw["output"] if "output" in kw else pos[2]
    use_ztrunk = bool(kw.get("use_zij_trunk_embedding", True))
    ref_out, cot, ref_grads = B["outputs"], B["cotangents"], B["param_grads"]

    si_trunk, zij_trunk = outd["si_trunk"], outd["zij_trunk"]
    xpred = outd["atom_positions_predicted"].to(dtype=si_trunk.dtype)
    token_mask = batch["token_mask"]
    repr_x, repr_mask = get_token_representative_atoms(
        batch=batch, x=xpred, atom_mask=batch["atom_mask"])
    max_atom_mask = broadcast_token_feat_to_atoms(
        token_mask=token_mask, num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=token_mask, max_num_atoms_per_token=23)

    n_tok = int(si_trunk.shape[-2])
    n_real = int(token_mask.sum())
    print(f"boundary: {n_tok} tokens ({n_real} real)  use_zij_trunk_embedding={use_ztrunk}",
          flush=True)
    for k, v in sorted(ref_out.items()):
        if torch.is_tensor(v):
            print(f"  ref out {k:34s} {tuple(v.shape)}", flush=True)
    for k, v in sorted(cot.items()):
        print(f"  cotangent {k:32s} {tuple(v.shape)}", flush=True)

    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    reg, restore = GD.record_uploads()
    head = OF3ConfidenceHead(aux, dev, ckc)
    restore()

    si2 = squeeze_leading(si_input, 2)
    st2 = squeeze_leading(si_trunk, 2)
    zt3 = squeeze_leading(zij_trunk, 3)
    rx2 = squeeze_leading(repr_x, 2)
    si_d = ttnn.from_torch(si2.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16)
    st_d = ttnn.from_torch(st2.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.float32)
    zt_d = ttnn.from_torch(zt3.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16)
    oh_d = head.distance_onehot(rx2)

    head.forward_device(si_d, st_d, zt_d, oh_d, use_zij_trunk_embedding=use_ztrunk)
    params, unmapped, inverse_checks = GD.tape_parameters(head, reg)
    dn = lambda t: torch.Tensor(ttnn.to_torch(t)).double()

    report = {
        "instrument": "instrument A at aux_heads scope, 0.4.3 boundary",
        "boundary": {"file": str(a.boundary), "n_tokens": n_tok, "n_tokens_real": n_real,
                     "use_zij_trunk_embedding": use_ztrunk},
        "bars": {"per_tensor": PER_TENSOR_BAR, "median": MEDIAN_BAR},
        "bijection": {"n_taped": len(params), "n_unmapped": len(unmapped),
                      "unmapped": [[p, list(s), why] for p, s, why in unmapped],
                      "inverse_worst": (list(max(inverse_checks, key=lambda x: x[1]))
                                        if inverse_checks else None)},
    }

    # ---- the taped forward, once; the gradient is taken at THIS forward ---------------------
    sti = ag.Tensor(st_d, requires_grad=True)
    zti = ag.Tensor(zt_d, requires_grad=True)
    with ag.tape():
        out = head.forward_device(ag.Tensor(si_d), sti, zti, ag.Tensor(oh_d),
                                  use_zij_trunk_embedding=use_ztrunk)

    tokm = token_mask.reshape(-1)[:n_tok].bool()
    atomm = max_atom_mask.reshape(-1)[:n_tok * 23].bool()
    fwd = []
    for k in HEADS:
        ours = dn(out[k])
        theirs = ref_out[k].double()
        if k in ATOM_HEADS:
            c = theirs.shape[-1]
            th = squeeze_leading(theirs, 2)
            og = ours.reshape(n_tok * 23, c)[atomm]
            row = {"head": k, "layout": "atom-gathered", "shape": list(th.shape),
                   "rel_l2": rel_l2(og, th), "ref_norm": float(th.norm()),
                   "our_norm": float(og.norm())}
        else:
            c = theirs.shape[-1]
            th = squeeze_leading(theirs, 3)
            og = ours.reshape(n_tok, n_tok, c)
            row = {"head": k, "layout": "pair", "shape": list(th.shape),
                   "rel_l2": rel_l2(og, th),
                   "rel_l2_real_block": rel_l2(og[tokm][:, tokm], th[tokm][:, tokm]),
                   "ref_norm": float(th.norm()),
                   "ref_norm_real_block": float(th[tokm][:, tokm].norm()),
                   "our_norm": float(og.norm())}
        fwd.append(row)
        extra = f"  real-block {row[rel_l2_real_block]:.4e}" if "rel_l2_real_block" in row else ""
        print(f"  FWD {k:34s} rel_l2 {row[rel_l2]:.4e}{extra}", flush=True)
    worst = max(fwd, key=lambda r: r["rel_l2"])
    report["forward"] = {
        "rows": fwd, "worst_head": worst["head"], "worst_rel_l2": worst["rel_l2"],
        "n_over_bar": sum(1 for r in fwd if r["rel_l2"] > PER_TENSOR_BAR),
        "n_heads": len(fwd),
        "a18_first_clause": ("PASS -- every head inside the 5.0e-02 bar, so the gradient "
                             "below is taken at an agreeing forward (necessary, not "
                             "sufficient)")
        if worst["rel_l2"] <= PER_TENSOR_BAR else
        ("FAIL -- a head is outside the 5.0e-02 bar, so any gradient taken at this "
         "boundary is invalid until the forward is localised"),
    }
    print(f"[{time.perf_counter()-t0:.0f}s] forward: worst {worst[head]} "
          f"{worst[rel_l2]:.4e}  -> {report[forward][a18_first_clause][:4]}", flush=True)

    if a.forward_only:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        return 0

    # ---- seed the backward with THEIR cotangents -------------------------------------------
    roots, seeds = [], []
    for k in HEADS:
        if k not in cot:
            print(f"  NO COTANGENT for {k}; it contributed nothing to the loss", flush=True)
            continue
        g = cot[k].double()
        o = out[k]
        if k in ATOM_HEADS:
            # adjoint of their masked_select: scatter back into the padded atom layout.
            c = g.shape[-1]
            gs = squeeze_leading(g, 2)
            full = torch.zeros(n_tok * 23, c, dtype=torch.float64)
            full[atomm] = gs
            sv = full.reshape(1, n_tok, 23 * c)
        else:
            sv = squeeze_leading(g, 3).reshape(tuple(int(d) for d in o.shape))
        roots.append(o)
        seeds.append(ttnn.from_torch(sv.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                     dtype=ttnn.bfloat16))
    ag.backward(roots, seeds)
    print(f"[{time.perf_counter()-t0:.0f}s] backward done over {len(roots)} roots", flush=True)

    # ---- per-parameter against the float64 reference ---------------------------------------
    pert_name, pert_factor = (a.perturb.rsplit(":", 1) if a.perturb else (None, None))
    rows = []
    for their, (leaf, inv, _where, _bi, lookup) in sorted(params.items()):
        full = "aux_heads." + lookup
        ref = ref_grads.get(full)
        if leaf.grad is None:
            rows.append({"name": full, "skip": "no gradient reached this leaf"}); continue
        if ref is None:
            rows.append({"name": full, "skip": "no reference gradient under this name"}); continue
        gd = inv(dn(leaf.grad))
        if a.zero_model:
            gd = torch.zeros_like(gd)
        if pert_name and full == pert_name:
            gd = gd * float(pert_factor)
        ref = ref.double()
        if tuple(gd.shape) != tuple(ref.shape):
            rows.append({"name": full,
                         "skip": f"shape {tuple(gd.shape)} vs ref {tuple(ref.shape)}"}); continue
        # SS3a: the checkpoint fuses TriangleMultiplications a/b halves into p_in/g_in, so
        # each half is scored on its own -- a fused number lets one halfs agreement mask
        # the others error.
        if lookup.endswith(("p_in.weight", "g_in.weight")) and gd.shape[0] % 2 == 0:
            h = gd.shape[0] // 2
            for tag, sl in (("[a]", slice(0, h)), ("[b]", slice(h, None))):
                rows.append({"name": full + " " + tag, "rel_l2": rel_l2(gd[sl], ref[sl]),
                             "ref_norm": float(ref[sl].norm()),
                             "our_norm": float(gd[sl].norm()),
                             "ref_sq": float((ref[sl].double() ** 2).sum())})
            continue
        rows.append({"name": full, "rel_l2": rel_l2(gd, ref), "ref_norm": float(ref.norm()),
                     "our_norm": float(gd.norm()), "ref_sq": float((ref ** 2).sum())})

    scored = [r for r in rows if "rel_l2" in r]
    skipped = [r for r in rows if "skip" in r]
    norms = sorted(r["ref_norm"] for r in scored) or [1.0]
    med_ref = norms[len(norms) // 2]
    floor = A14_FLOOR_RATIO * med_ref
    a14 = [r for r in scored if r["ref_norm"] < floor]
    scored = [r for r in scored if r["ref_norm"] >= floor]
    scored.sort(key=lambda r: -r["rel_l2"])
    meds = sorted(r["rel_l2"] for r in scored)
    median = meds[len(meds) // 2] if meds else None

    # ---- reach, in share of the squared gradient norm (A15), never tensor count ------------
    reach = {}
    if a.reference_grads:
        ref_all = torch.load(a.reference_grads, map_location="cpu", weights_only=False)
        aux_sq = sum(float((v.double() ** 2).sum()) for k, v in ref_all.items()
                     if k.startswith("aux_heads.") and v is not None)
        model_sq = sum(float((v.double() ** 2).sum()) for v in ref_all.values() if v is not None)
        cmp_sq = sum(r["ref_sq"] for r in scored)
        reach = {
            "aux_heads_squared_norm": aux_sq,
            "model_squared_norm": model_sq,
            "aux_heads_share_of_model": aux_sq / model_sq,
            "compared_squared_norm": cmp_sq,
            "share_of_aux_heads_own_norm": cmp_sq / aux_sq if aux_sq else None,
            "share_of_model_squared_norm": cmp_sq / model_sq if model_sq else None,
            "n_aux_heads_tensors": sum(1 for k in ref_all if k.startswith("aux_heads.")),
        }

    report["gradient"] = {
        "n_scored": len(scored), "n_skipped": len(skipped),
        "n_a14_excluded": len(a14),
        "a14_floor": floor, "a14_median_ref_norm": med_ref,
        "a14_excluded": [{"name": r["name"], "ref_norm": r["ref_norm"],
                          "our_norm": r["our_norm"]} for r in a14],
        "median_rel_l2": median,
        "worst": scored[0] if scored else None,
        "n_over_bar": sum(1 for r in scored if r["rel_l2"] > PER_TENSOR_BAR),
        "rows": scored, "skipped": skipped,
        "reach": reach,
        "zero_model_arm": bool(a.zero_model),
        "perturbation": a.perturb,
    }
    if scored:
        n_over = report["gradient"]["n_over_bar"]
        print(f"\nPER-PARAMETER vs float64 0.4.3: {len(scored)} scored, {len(a14)} A14-excluded, "
              f"{len(skipped)} skipped")
        print(f"  worst   {scored[0][name]}  {scored[0][rel_l2]:.4e}  (bar 5.0e-02)")
        print(f"  median  {median:.4e}  (bar 2.0e-02)")
        print(f"  over the per-tensor bar: {n_over} of {len(scored)}")
        if reach:
            print(f"  reach: {reach[share_of_aux_heads_own_norm]*100:.3f} %% of aux_heads own "
                  f"squared norm = {reach[share_of_model_squared_norm]*100:.4f} %% of the model")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"[{time.perf_counter()-t0:.0f}s] written {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
