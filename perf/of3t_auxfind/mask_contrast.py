#!/usr/bin/env python3
"""Does the missing mask explain aux_heads' A18 failure? Arms R and U, float64, CPU.

PREREGISTERED.md, arms R and U. One process, one module, two mask settings:

  R   single_mask=repr_x_mask, pair_mask=token_mask outer -- what upstream 0.4.3 passes.
  U   both forced to ALL ONES -- the function our port computes, in exact arithmetic.

R vs U is the mask's whole effect with precision removed. The arms differ in one VALUE, never
in a code path: the mask tensors are substituted inside PairformerEmbedding.forward and nothing
else is touched, so an accidental second difference cannot hide in the contrast.

Per-leaf localisation, both ways PREREGISTERED.md names:

  created      feed the leaf arm R's OWN captured input and run it with the mask forced to ones.
               Only that leaf's configuration differs, so the number is what that op MAKES.
  propagated   the two full runs compared at the same leaf's output. Carries everything made
               upstream of it.

Median per leaf NAME across the four blocks, never the worst tensor
(`worst-tensor-names-the-tail-not-the-locus`).

Controls, all three fixed in the pre-registration:
  * arm R against the boundary's captured reference outputs must read exactly 0.0 (same-function);
  * the created instrument re-running a leaf with its OWN mask must read exactly 0.0 (vacuous
    direction -- an instrument that always reads non-zero is not reading the mask);
  * the created instrument with a DELIBERATELY corrupted mask must move
    (`negative-control-must-break-what-check-reads`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

BOUND = "/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt"
CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
PAIR_HEADS = ["pae_logits", "pde_logits", "distogram_logits"]
ATOM_HEADS = ["plddt_logits", "experimentally_resolved_logits"]
HEADS = ATOM_HEADS + PAIR_HEADS
BAR = 5.0e-2
LEAVES = ["tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end",
          "pair_transition", "attn_pair_bias", "single_transition"]


def sq(t, rank):
    """Drop leading singleton dims down to `rank`; a non-singleton leading dim is a real axis."""
    while t.dim() > rank:
        assert t.shape[0] == 1, t.shape
        t = t[0]
    return t


def stats(ours, ref):
    """rel_l2 with the norm ratio and the error cosine beside it, never rel alone."""
    a, b = ours.flatten().double(), ref.flatten().double()
    nb = float(b.norm())
    na = float(a.norm())
    return {"rel_l2": float((a - b).norm() / (nb + 1e-300)),
            "r": (na / nb if nb else float("nan")),
            "cos": (float(torch.dot(a, b) / (a.norm() * b.norm())) if na and nb else float("nan")),
            "ref_norm": nb, "our_norm": na}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/home/ttuser/of3t_rebase/of3pkg043")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.set_num_threads(a.threads)

    sys.path.insert(0, a.tree)
    import openfold3
    assert openfold3.__file__.startswith(a.tree), openfold3.__file__
    from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
    import openfold3.core.model.heads.prediction_heads as PH
    from openfold3.projects.of3_all_atom.config.model_config import model_config

    B = torch.load(BOUND, map_location="cpu", weights_only=False)
    kw = {k: v for k, v in B["inputs"]["kwargs"].items()}
    ref_out = {k: v.double().clone() for k, v in B["outputs"].items() if torch.is_tensor(v)}
    token_mask = kw["batch"]["token_mask"].reshape(-1).bool().clone()
    del B
    n_tok = int(token_mask.numel())
    n_real = int(token_mask.sum())

    def move(x):
        if torch.is_tensor(x):
            return x.double() if x.is_floating_point() else x
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(move(v) for v in x)
        return x
    kw = move(kw)

    head = AuxiliaryHeadsAllAtom(config=model_config.architecture.heads).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sub = {k[len("aux_heads."):]: v.double() for k, v in sd.items()
           if k.startswith("aux_heads.") and torch.is_tensor(v)}
    del sd
    inc = head.load_state_dict(sub, strict=False)
    key_gate = {"loaded": len(sub), "missing": len(inc.missing_keys),
                "unexpected": len(inc.unexpected_keys)}
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE: {len(inc.unexpected_keys)} unexpected tensors")
    head = head.double()

    # ---- the one substitution that makes arm U --------------------------------------------
    FORCE = {"mode": "native"}
    _orig_pe_fwd = PH.PairformerEmbedding.forward
    seen = {}

    def _pe(self, *args, **kwargs):
        sm, pm = kwargs["single_mask"], kwargs["pair_mask"]
        if FORCE["mode"] == "ones":
            kwargs["single_mask"] = torch.ones_like(sm)
            kwargs["pair_mask"] = torch.ones_like(pm)
        seen["single_sum"] = float(kwargs["single_mask"].double().sum())
        seen["pair_sum"] = float(kwargs["pair_mask"].double().sum())
        seen["single_sum_native"] = float(sm.double().sum())
        seen["pair_sum_native"] = float(pm.double().sum())
        return _orig_pe_fwd(self, *args, **kwargs)

    PH.PairformerEmbedding.forward = _pe

    # ---- leaf hooks -------------------------------------------------------------------------
    stack = head.pairformer_embedding.pairformer_stack
    blocks = list(stack.blocks)
    CAP = {}          # (arm, block, leaf) -> {"out":..., "args":..., "kwargs":...}
    ARM = {"tag": "R", "capture_io": True}

    def mk(bi, leaf):
        def hook(mod, args, kwargs, output):
            key = (ARM["tag"], bi, leaf)
            rec = {"out": output.detach().clone()}
            if ARM["capture_io"]:
                rec["args"] = tuple(x.detach().clone() if torch.is_tensor(x) else x for x in args)
                rec["kwargs"] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                 for k, v in kwargs.items()}
            CAP[key] = rec
        return hook

    BLK = {}

    def mkb(bi):
        def hook(mod, args, kwargs, output):
            s_, z_ = output
            BLK[(ARM["tag"], bi)] = (s_.detach().clone(), z_.detach().clone())
        return hook

    hooks = []
    for bi, blk in enumerate(blocks):
        hooks.append(blk.register_forward_hook(mkb(bi), with_kwargs=True))
    for bi, blk in enumerate(blocks):
        for leaf in LEAVES:
            mod = getattr(blk.pair_stack, leaf, None) or getattr(blk, leaf)
            hooks.append(mod.register_forward_hook(mk(bi, leaf), with_kwargs=True))

    # ---- arm R ------------------------------------------------------------------------------
    with torch.no_grad():
        outR = {k: v.detach().double().clone() for k, v in head(**kw).items()}
    masks_R = dict(seen)
    print(f"[{time.perf_counter()-t0:.0f}s] arm R done  single_mask sum "
          f"{masks_R['single_sum']:.0f}  pair_mask sum {masks_R['pair_sum']:.0f}", flush=True)

    # ---- arm U ------------------------------------------------------------------------------
    FORCE["mode"] = "ones"
    ARM["tag"], ARM["capture_io"] = "U", False
    with torch.no_grad():
        outU = {k: v.detach().double().clone() for k, v in head(**kw).items()}
    masks_U = dict(seen)
    print(f"[{time.perf_counter()-t0:.0f}s] arm U done  single_mask sum "
          f"{masks_U['single_sum']:.0f}  pair_mask sum {masks_U['pair_sum']:.0f}", flush=True)
    for h in hooks:
        h.remove()

    rep = {"boundary": BOUND, "n_tokens": n_tok, "n_tokens_real": n_real,
           "padding_fraction_tokens": 1.0 - n_real / n_tok,
           "dtype_policy_A27": ("every parameter and every activation float64, no cast on the "
                                "path, of3-p2-155k upcast once at load; autocast('cuda') is inert "
                                "on CPU and torch says so"),
           "key_gate": key_gate, "masks": {"R": masks_R, "U": masks_U}, "bar": BAR}

    # ---- control 1: arm R must reproduce the captured reference exactly ----------------------
    same = {}
    for k in HEADS:
        rank = 2 if k in ATOM_HEADS else 3
        r, o = sq(ref_out[k], rank), sq(outR[k], rank)
        same[k] = {"rel_l2": stats(o, r)["rel_l2"],
                   "max_abs": float((o - r).abs().max())}
    rep["control_same_function"] = same
    print("  control same-function worst rel "
          f"{max(v['rel_l2'] for v in same.values()):.4e}", flush=True)

    # ---- the heads, R vs U, on the scoring scope A18/D95 names -------------------------------
    tokm = token_mask
    heads_rows = []
    for k in HEADS:
        rank = 2 if k in ATOM_HEADS else 3
        r, u = sq(ref_out[k], rank), sq(outU[k], rank)
        row = {"head": k, "scope": "atom-gathered (422 of 422 atoms, no padding)"
               if k in ATOM_HEADS else f"pair, whole {n_tok}x{n_tok}",
               "padding_fraction": 0.0 if k in ATOM_HEADS else 1.0 - n_real / n_tok,
               "U_vs_R": stats(u, r)}
        if k in PAIR_HEADS:
            row["U_vs_R_real_block"] = stats(u[tokm][:, tokm], r[tokm][:, tokm])
            row["scope_real"] = f"pair, real block {n_real}x{n_real}"
        # A16: the zero model on the same scope, measured not assumed
        row["A16_zero_model"] = stats(torch.zeros_like(r), r)["rel_l2"]
        heads_rows.append(row)
        rb = row.get("U_vs_R_real_block", row["U_vs_R"])
        print(f"  HEAD {k:32s} U_vs_R {row['U_vs_R']['rel_l2']:.6e}  "
              f"real {rb['rel_l2']:.6e}  r {rb['r']:.6f}  cos {rb['cos']:.6f}", flush=True)
    rep["heads"] = heads_rows

    # ---- the Pairformer output itself, for the C2 amplification question ---------------------
    # rerun the embedding alone under both masks to get si/zij_conf without the heads' LayerNorm
    rep["note_amplification"] = ("head rel over pairformer-output rel; the pairformer outputs are "
                                 "taken from the last block's captured leaf outputs")
    amp = {}
    for bi in range(len(blocks)):
        sR, zR = BLK[("R", bi)]
        sU, zU = BLK[("U", bi)]
        sR3, sU3 = sq(sR, 2), sq(sU, 2)
        zR3, zU3 = sq(zR, 3), sq(zU, 3)
        amp[f"block{bi}_s"] = stats(sU3[tokm], sR3[tokm])
        amp[f"block{bi}_z"] = stats(zU3[tokm][:, tokm], zR3[tokm][:, tokm])
        amp[f"block{bi}_s_padded"] = stats(sU3, sR3)
        amp[f"block{bi}_z_padded"] = stats(zU3, zR3)
    rep["pairformer_output_contrast_real_block"] = amp
    print("  PAIRFORMER OUT (real block): " + "  ".join(
        f"{k} {v['rel_l2']:.3e}" for k, v in amp.items() if not k.endswith("padded")),
        flush=True)

    # ---- per-leaf: propagated, then created --------------------------------------------------
    prop = []
    for bi in range(len(blocks)):
        for leaf in LEAVES:
            kr, ku = ("R", bi, leaf), ("U", bi, leaf)
            if kr not in CAP or ku not in CAP:
                continue
            s = stats(CAP[ku]["out"], CAP[kr]["out"])
            prop.append({"block": bi, "leaf": leaf, **s})
    rep["leaf_propagated"] = prop

    created, ctl_vacuous, ctl_broken = [], [], []
    for bi, blk in enumerate(blocks):
        for leaf in LEAVES:
            kr = ("R", bi, leaf)
            if kr not in CAP:
                continue
            mod = getattr(blk.pair_stack, leaf, None) or getattr(blk, leaf)
            rec = CAP[kr]
            args = tuple(x.clone() if torch.is_tensor(x) else x for x in rec["args"])
            base_kw = {k: (v.clone() if torch.is_tensor(v) else v)
                       for k, v in rec["kwargs"].items()}
            if "mask" not in base_kw or base_kw["mask"] is None:
                created.append({"block": bi, "leaf": leaf, "rel_l2": None,
                                "note": "leaf received mask=None in arm R; nothing to force"})
                continue
            with torch.no_grad():
                kw_ones = dict(base_kw); kw_ones["mask"] = torch.ones_like(base_kw["mask"])
                o_ones = mod(*args, **kw_ones)
                kw_same = dict(base_kw)
                o_same = mod(*tuple(x.clone() if torch.is_tensor(x) else x for x in args),
                             **kw_same)
                m_brk = base_kw["mask"].clone()
                flat = m_brk.reshape(-1)
                idx = torch.nonzero(flat > 0).flatten()[:4]
                flat[idx] = 0
                kw_brk = dict(base_kw); kw_brk["mask"] = m_brk.reshape(base_kw["mask"].shape)
                o_brk = mod(*tuple(x.clone() if torch.is_tensor(x) else x for x in args),
                            **kw_brk)
            # the residual stream the leaf writes into: positional for the pair leaves,
            # `a` for attn_pair_bias, `x`/`z` for whatever passes it by keyword
            zin = args[0] if args else next(
                (base_kw[k] for k in ("z", "a", "x") if torch.is_tensor(base_kw.get(k))), None)
            if zin is None:
                raise SystemExit(f"{leaf}: no input tensor found in args/kwargs "
                                 f"{list(base_kw)} -- the common denominator cannot be built")
            d = (o_ones - rec["out"]).flatten().double()
            created.append({"block": bi, "leaf": leaf, **stats(o_ones, rec["out"]),
                            "created_abs": float(d.norm()),
                            "input_norm": float(zin.flatten().double().norm()),
                            "created_vs_input": float(d.norm()
                                                      / (zin.flatten().double().norm() + 1e-300))})
            ctl_vacuous.append({"block": bi, "leaf": leaf,
                                "rel_l2": stats(o_same, rec["out"])["rel_l2"],
                                "max_abs": float((o_same - rec["out"]).abs().max())})
            ctl_broken.append({"block": bi, "leaf": leaf, "n_zeroed": int(idx.numel()),
                               "rel_l2": stats(o_brk, rec["out"])["rel_l2"]})
        print(f"[{time.perf_counter()-t0:.0f}s] block {bi} leaves done", flush=True)
    rep["leaf_created"] = created
    rep["control_created_vacuous"] = ctl_vacuous
    rep["control_created_broken"] = ctl_broken

    def med(rows, field="rel_l2"):
        out = {}
        for leaf in LEAVES:
            vs = [r[field] for r in rows if r["leaf"] == leaf and r.get(field) is not None]
            if vs:
                out[leaf] = {"median": statistics.median(vs), "n": len(vs),
                             "min": min(vs), "max": max(vs)}
        return out
    rep["leaf_created_median"] = med(created)
    rep["leaf_propagated_median"] = med(prop)
    # The denominator that makes the seven leaves comparable: each leaf's own output norm is a
    # different quantity (0.4.3's tri_mul leaves return the UPDATED z, the attention and
    # transition leaves return an update), so rel_l2 per leaf cannot be ranked across families.
    # ||created delta|| over the norm of the tensor the leaf was handed can be.
    rep["leaf_created_vs_input_median"] = med(created, "created_vs_input")
    print("  LOCUS created/input medians: " + "  ".join(
        f"{k} {v['median']:.3e}" for k, v in rep["leaf_created_vs_input_median"].items()),
        flush=True)
    print("  LOCUS created medians: " + "  ".join(
        f"{k} {v['median']:.3e}" for k, v in rep["leaf_created_median"].items()), flush=True)
    print("  LOCUS propagated medians: " + "  ".join(
        f"{k} {v['median']:.3e}" for k, v in rep["leaf_propagated_median"].items()), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"[{time.perf_counter()-t0:.0f}s] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
