#!/usr/bin/env python3
"""Instrument A at `aux_heads` scope, RE-TAKEN with of3t-auxfind's mask fix on.

of3t-auxheads took this gradient on a `forward_device` that passed no masks to the
confidence Pairformer. of3t-auxfind showed that is a WRONG transform, not an imprecise
one, so the 2.2996e-03 on record is the gradient of a different function. This is the
same instrument with one argument pair added (`--mask`) and three readings the original
did not produce: the A23 mass-weighted headline with its norm ratio and error cosine,
the per-leaf-op medians, and the shipped host-path `forward` run in THIS process so no
forward number is carried in across processes.

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


SQRT2 = 2.0 ** 0.5


# The seven leaf ops of a confidence Pairformer block, in the checkpoint's own naming.
# Medians are reported BY LEAF, because the worst tensor names the tail and not the locus
# (fleet memory `worst-tensor-names-the-tail-not-the-locus`).
_LEAVES = ["tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end",
           "attn_pair_bias", "pair_transition", "single_transition",
           "triangle_multiplication_outgoing", "triangle_multiplication_incoming",
           "triangle_attention_starting_node", "triangle_attention_ending_node",
           "transition_z", "transition_s"]


def _leaf_of(name):
    for leaf in _LEAVES:
        if "." + leaf + "." in name:
            return leaf
    if ".distogram." in name:
        return "distogram (reads zij_trunk, never enters the masked path)"
    if "pairformer_embedding.linear" in name or "linear_distance" in name:
        return "pairformer_embedding input projections"
    return "head output projections"


def _row(name, ours, ref):
    """One scored row, carrying the four sums the A23 mass-weighted headline is made of.

    Sums rather than tensors: the headline is rel_l2 over the CONCATENATED set, and
    ||concat(x)||^2 = sum ||x_i||^2 with <concat(a),concat(b)> = sum <a_i,b_i>, so the
    concatenated norm ratio and error cosine come out exact without holding 176 tensors.
    """
    o, r = ours.flatten().double(), ref.flatten().double()
    d = o - r
    return {"name": name, "leaf": _leaf_of(name),
            "rel_l2": float(d.norm() / (r.norm() + 1e-30)),
            "norm_ratio": float(o.norm() / (r.norm() + 1e-30)),
            "cos": float(torch.dot(o, r) / (o.norm() * r.norm() + 1e-30)),
            "ref_norm": float(r.norm()), "our_norm": float(o.norm()),
            "ref_sq": float((r ** 2).sum()), "our_sq": float((o ** 2).sum()),
            "diff_sq": float((d ** 2).sum()), "dot": float(torch.dot(o, r))}


def _mass_weighted(rows):
    """A23 rule 2: rel_l2 over the concatenation, with the direction readings beside it."""
    if not rows:
        return None
    sd = sum(r["diff_sq"] for r in rows)
    sr = sum(r["ref_sq"] for r in rows)
    so = sum(r["our_sq"] for r in rows)
    dot = sum(r["dot"] for r in rows)
    return {"rel_l2": (sd ** 0.5) / (sr ** 0.5 + 1e-300),
            "norm_ratio": (so ** 0.5) / (sr ** 0.5 + 1e-300),
            "cos": dot / ((so ** 0.5) * (sr ** 0.5) + 1e-300),
            "n_tensors": len(rows), "ref_squared_norm": sr}


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


# grad_device.their_name stops at tt-bios own primitive names for the pair stack, which are
# not the checkpoints. Eight renames close the gap, and every one is checked by the shape
# test at the comparison, so a wrong guess drops the tensor rather than scoring two
# different ones against each other.
_RENAME = [
    ("pair_transition.fc1.", "pair_transition.swiglu.linear_a."),
    ("pair_transition.fc2.", "pair_transition.swiglu.linear_b."),
    ("pair_transition.fc3.", "pair_transition.linear_out."),
    ("pair_transition.norm.", "pair_transition.layer_norm."),
    (".norm_in.", ".layer_norm_in."),
    (".norm_out.", ".layer_norm_out."),
    (".g_out.", ".linear_g."),
    (".p_out.", ".linear_z."),
]


def _upstream_name(n):
    for a, b in _RENAME:
        n = n.replace(a, b)
    return n


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
    ap.add_argument("--scramble-cot", action="store_true", dest="scramble_cot",
                    help="negative control: seed the backward with the SAME cotangent numbers written into the wrong positions (the flattened cotangent reversed). Norm and shape are preserved exactly, and the forward, the weights and the arithmetic are untouched, so a comparison that cannot tell this apart from the real run is not reading their seed at all. A zero-model baseline cannot catch that: it breaks OUR side.")
    ap.add_argument("--dump-grads", default="", dest="dump_grads",
                    help="write the compared device gradient TENSORS to this .pt, keyed by full checkpoint name. Without it this arm publishes only rel_l2 against the one float64 reference it was run against, so its gradient can never be compared to upstream's own bf16 training gradient -- and two distances from a shared reference do not order each other (D72).")
    ap.add_argument("--mask", action="store_true",
                    help="arm M: pass the reference's masks into the confidence "
                         "Pairformer -- pair_mask = token_mask outer token_mask into both "
                         "triangle multiplications, and the additive -1e9 companion into "
                         "both triangle attentions and attention_pair_bias. Default off, "
                         "which is arm N and the shipped default.")
    ap.add_argument("--shipped-forward", action="store_true", dest="shipped_forward",
                    help="also run the shipped HOST-path OF3ConfidenceHead.forward on the "
                         "same boundary in this same process, masked and unmasked, so "
                         "of3t-auxfind's 3.865648e-03 is reproduced here rather than "
                         "quoted from another process.")
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

    # arm M. Built from the boundary's OWN token_mask, not from a contiguous-prefix
    # assumption: token_pad_masks_torch takes a length, and a captured training crop is
    # not obliged to put its real tokens first.
    tm1 = token_mask.reshape(1, n_tok).float()
    repr_mask_1d = repr_mask.reshape(-1).float()
    pair_mask_d = attn_mask_d = None
    if a.mask:
        pm = tm1[:, :, None] * tm1[:, None, :]
        am = (1.0 - tm1).unsqueeze(1).unsqueeze(1) * -1e9
        pair_mask_d = ttnn.from_torch(pm, layout=ttnn.TILE_LAYOUT, device=dev,
                                      dtype=ttnn.bfloat16)
        attn_mask_d = ttnn.from_torch(am, layout=ttnn.TILE_LAYOUT, device=dev,
                                      dtype=ttnn.bfloat16)
    # One attn_mask serves the two triangle attentions AND attention_pair_bias, while
    # upstream passes pair_mask to the first two and single_mask=repr_x_mask to the third.
    # That is only the same function if the two masks agree here, so say whether they do.
    mask_agree = bool(torch.equal(tm1.reshape(-1), repr_mask_1d))
    print(f"arm: mask={a.mask}  token_mask==repr_x_mask: {mask_agree}  "
          f"padded {100.0 * (1.0 - n_real / n_tok):.4f} %", flush=True)


    # The real-token and real-atom scopes, per D95 and A18-as-amended: the pair heads are
    # scored on the real block, the atom heads on the gathered atoms.
    tokm = token_mask.reshape(-1)[:n_tok].bool()
    atomm = max_atom_mask.reshape(-1)[:n_tok * 23].bool()

    if a.shipped_forward:
        # of3t-auxfind measured the shipped `forward` at 3.865648e-03 worst head with the masks
        # on. It is a DIFFERENT function from `forward_device` (host fp32 s-path against device
        # bf16 s-path), so it is run here rather than inferred from the taped arm -- and it runs
        # HERE, before tape_parameters, because after taping the head's weights are
        # autograd.Tensors and ttnn.layer_norm refuses them outside a tape.
        sf = []
        with torch.no_grad():
            for tag, tmarg in (("masked", token_mask), ("unmasked", None)):
                o = head.forward(
                    si_input=squeeze_leading(si_input, 2).float(),
                    si_trunk=st2.float(), zij_trunk=zt3.float(),
                    repr_x_pred=rx2.float(),
                    max_atom_per_token_mask=max_atom_mask.reshape(-1).float(),
                    use_zij_trunk_embedding=use_ztrunk,
                    token_mask=tmarg,
                    single_mask=(repr_mask_1d if tmarg is not None else None))
                for k in HEADS:
                    ours, theirs = o[k].detach().double(), ref_out[k].double()
                    if k in ATOM_HEADS:
                        th = squeeze_leading(theirs, 2)
                        scope = (f"atom-gathered, {int(atomm.sum())} of {int(atomm.numel())} "
                                 f"slots, 0 padding on the atom axis")
                        r = _row(k, ours.reshape(th.shape), th)
                    else:
                        th = squeeze_leading(theirs, 3)
                        og = ours.reshape(th.shape)
                        scope = (f"pair, real {n_real}x{n_real} block of {n_tok}x{n_tok}, "
                                 f"{100.0 * (1.0 - n_real / n_tok):.4f} % padded")
                        r = _row(k, og[tokm][:, tokm], th[tokm][:, tokm])
                        r["rel_l2_padded_scope"] = rel_l2(og, th)
                    sf.append({"arm": tag, "head": k, "scope": scope, **r})
                    print(f"  SHIPPED-FWD {tag:8s} {k:34s} {r['rel_l2']:.6e}  "
                          f"r {r['norm_ratio']:.6f} cos {r['cos']:.6f}", flush=True)
        wm = max((r for r in sf if r["arm"] == "masked"), key=lambda r: r["rel_l2"])
        wu = max((r for r in sf if r["arm"] == "unmasked"), key=lambda r: r["rel_l2"])
        report_shipped = {
            "rows": sf, "worst_masked": wm, "worst_unmasked": wu,
            "n_over_bar_masked": sum(1 for r in sf
                                     if r["arm"] == "masked" and r["rel_l2"] > PER_TENSOR_BAR),
            "n_over_bar_unmasked": sum(1 for r in sf if r["arm"] == "unmasked"
                                       and r["rel_l2"] > PER_TENSOR_BAR),
            "auxfind_published_masked_worst": 3.865648e-03,
            "auxfind_published_unmasked_worst": 5.174368e-01}
        print(f"  SHIPPED-FWD worst masked {wm['head']} {wm['rel_l2']:.6e} "
              f"(auxfind published 3.865648e-03); worst unmasked {wu['head']} "
              f"{wu['rel_l2']:.6e} (published 5.174368e-01)", flush=True)
    else:
        report_shipped = None

    head.forward_device(si_d, st_d, zt_d, oh_d, use_zij_trunk_embedding=use_ztrunk,
                        pair_mask_d=pair_mask_d, attn_mask_d=attn_mask_d)
    params, unmapped, inverse_checks = GD.tape_parameters(head, reg)
    def dn(t):
        # a taped output is an autograd.Tensor wrapping the device handle; a leaf
        # gradient is the raw handle. Take either.
        return torch.Tensor(ttnn.to_torch(getattr(t, "value", t))).double()

    report = {
        "instrument": "instrument A at aux_heads scope, 0.4.3 boundary",
        "boundary": {"file": str(a.boundary), "n_tokens": n_tok, "n_tokens_real": n_real,
                     "use_zij_trunk_embedding": use_ztrunk},
        "bars": {"per_tensor": PER_TENSOR_BAR, "mass_weighted": MEDIAN_BAR,
                 "a26_reachable_per_tensor": SQRT2 * PER_TENSOR_BAR,
                 "a26_reachable_mass_weighted": SQRT2 * MEDIAN_BAR},
        "arm": {"mask": bool(a.mask), "name": "M (masked)" if a.mask else "N (unmasked, the shipped default and of3t-auxheads' arm)",
                "token_mask_equals_repr_x_mask": mask_agree,
                "padding_fraction": 1.0 - n_real / n_tok},
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
                                  use_zij_trunk_embedding=use_ztrunk,
                                  pair_mask_d=pair_mask_d, attn_mask_d=attn_mask_d)

    fwd = []
    for k in HEADS:
        ours = dn(out[k])
        theirs = ref_out[k].double()
        if k in ATOM_HEADS:
            c = theirs.shape[-1]
            th = squeeze_leading(theirs, 2)
            og = ours.reshape(n_tok * 23, c)[atomm]
            row = {"head": k, "layout": "atom-gathered", "shape": list(th.shape),
                   "scope": f"atom-gathered, {int(atomm.sum())} of {int(atomm.numel())} slots, "
                            "0 padding on the atom axis", **_row(k, og, th)}
        else:
            c = theirs.shape[-1]
            th = squeeze_leading(theirs, 3)
            og = ours.reshape(n_tok, n_tok, c)
            # D95 / A18-as-amended: the pair heads are scored on the REAL block. The padded
            # figure is kept beside it because it is a different question, not a worse answer.
            row = {"head": k, "layout": "pair", "shape": list(th.shape),
                   "scope": f"pair, real {n_real}x{n_real} block of {n_tok}x{n_tok}, "
                            f"{100.0 * (1.0 - n_real / n_tok):.4f} % padded",
                   "rel_l2_padded_scope": rel_l2(og, th),
                   **_row(k, og[tokm][:, tokm], th[tokm][:, tokm])}
        fwd.append(row)
        extra = (f"  padded-scope {row['rel_l2_padded_scope']:.4e}"
                 if "rel_l2_padded_scope" in row else "")
        print(f"  FWD {k:34s} rel_l2 {row['rel_l2']:.6e}  r {row['norm_ratio']:.6f} "
              f"cos {row['cos']:.6f}{extra}", flush=True)
    worst = max(fwd, key=lambda r: r["rel_l2"])
    if report_shipped is not None:
        report["shipped_forward"] = report_shipped
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
    print(f"[{time.perf_counter()-t0:.0f}s] forward: worst {worst['head']} "
          f"{worst['rel_l2']:.4e}  -> {report['forward']['a18_first_clause'][:4]}", flush=True)

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
        if a.scramble_cot:
            sv = sv.reshape(-1).flip(0).reshape(sv.shape).contiguous()
        roots.append(o)
        seeds.append(ttnn.from_torch(sv.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                     dtype=ttnn.bfloat16))
    ag.backward(roots, seeds)
    print(f"[{time.perf_counter()-t0:.0f}s] backward done over {len(roots)} roots", flush=True)

    # ---- per-parameter against the float64 reference ---------------------------------------
    pert_name, pert_factor = (a.perturb.rsplit(":", 1) if a.perturb else (None, None))
    rows = []
    dumped = {}
    for their, (leaf, inv, _where, _bi, _lookup) in sorted(params.items()):
        # The dict KEY is their full name; the trailing element of the value is a
        # scope-local lookup that only means anything against grad_device.mains
        # per-block reference dicts. Using it here silently dropped 100 of 180
        # tensors as no reference gradient under this name.
        full = "aux_heads." + _upstream_name(their)
        lookup = their
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
        # The p_in/g_in halves below are two ROWS against one reference tensor, so the
        # tensor dumped under `full` is the whole thing, which is what the bundle keys.
        dumped[full] = gd
        # SS3a: the checkpoint fuses TriangleMultiplications a/b halves into p_in/g_in, so
        # each half is scored on its own -- a fused number lets one halfs agreement mask
        # the others error.
        if lookup.endswith(("p_in.weight", "g_in.weight")) and gd.shape[0] % 2 == 0:
            h = gd.shape[0] // 2
            for tag, sl in (("[a]", slice(0, h)), ("[b]", slice(h, None))):
                rows.append(_row(full + " " + tag, gd[sl], ref[sl]))
            continue
        rows.append(_row(full, gd, ref))

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

    mw = _mass_weighted(scored)
    by_leaf = {}
    for r in scored:
        by_leaf.setdefault(r["leaf"], []).append(r)
    leaf_stats = []
    for leaf, rs in by_leaf.items():
        v = sorted(x["rel_l2"] for x in rs)
        lm = _mass_weighted(rs)
        leaf_stats.append({"leaf": leaf, "n": len(rs), "median_rel_l2": v[len(v) // 2],
                           "worst_rel_l2": v[-1],
                           "mass_weighted_rel_l2": lm["rel_l2"],
                           "share_of_compared_mass": lm["ref_squared_norm"] /
                           (mw["ref_squared_norm"] or 1.0),
                           "n_over_bar": sum(1 for x in rs if x["rel_l2"] > PER_TENSOR_BAR)})
    leaf_stats.sort(key=lambda d: -d["median_rel_l2"])
    mass_inside = (sum(r["ref_sq"] for r in scored if r["rel_l2"] <= PER_TENSOR_BAR)
                   / (mw["ref_squared_norm"] or 1.0)) if mw else None
    report["gradient"] = {
        "mass_weighted": mw,
        "by_leaf": leaf_stats,
        "mass_share_inside_per_tensor_bar": mass_inside,
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
        "cotangent_scrambled": bool(a.scramble_cot),
    }
    if scored:
        n_over = report["gradient"]["n_over_bar"]
        print(f"\nPER-PARAMETER vs float64 0.4.3: {len(scored)} scored, {len(a14)} A14-excluded, "
              f"{len(skipped)} skipped")
        print(f"  MASS-WEIGHTED HEADLINE  {mw['rel_l2']:.6e}   r {mw['norm_ratio']:.6f}  "
              f"cos {mw['cos']:.6f}   (bar 2.0e-02, A26 reachable {SQRT2*MEDIAN_BAR:.6e})")
        print(f"  mass inside the per-tensor bar: {mass_inside*100:.7f} %")
        print(f"  worst   {scored[0]['name']}  {scored[0]['rel_l2']:.4e}  "
              f"r {scored[0]['norm_ratio']:.6f} cos {scored[0]['cos']:.6f}  (bar 5.0e-02)")
        print("  per leaf op (median, then mass-weighted, then mass share):")
        for d in leaf_stats:
            print(f"    {d['leaf']:42s} n {d['n']:3d}  med {d['median_rel_l2']:.4e}  "
                  f"mw {d['mass_weighted_rel_l2']:.4e}  mass {d['share_of_compared_mass']:.3e}  "
                  f"over-bar {d['n_over_bar']}")
        print(f"  median  {median:.4e}  (bar 2.0e-02)")
        print(f"  over the per-tensor bar: {n_over} of {len(scored)}")
        if reach:
            print(f"  reach: {reach['share_of_aux_heads_own_norm']*100:.3f} %% of aux_heads own "
                  f"squared norm = {reach['share_of_model_squared_norm']*100:.4f} %% of the model")
    if a.dump_grads:
        Path(a.dump_grads).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in dumped.items()}, a.dump_grads)
        report["grads_dumped_to"] = a.dump_grads
        print(f"[{time.perf_counter()-t0:.0f}s] wrote {len(dumped)} gradient tensors to "
              f"{a.dump_grads}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"[{time.perf_counter()-t0:.0f}s] written {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
