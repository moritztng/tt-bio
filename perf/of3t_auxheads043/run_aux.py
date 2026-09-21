#!/usr/bin/env python3
"""Run ONE release tree's AuxiliaryHeadsAllAtom on the captured 0.4.3 aux_heads boundary.

Arms 1-3 of PREREGISTERED.md. One tree per process -- the two release trees cannot share an
interpreter -- and the tree is asserted after import rather than trusted from PYTHONPATH.

The inputs are the ones upstream's own float64 run was handed at `model.aux_heads`, replayed
verbatim from `boundary_aux_heads.pt`. Nothing here reconstructs them.

Knobs, one per pre-registered difference:

  --single-mask   force the head_modules difference. `repr` is 0.4.3's convention, `token` is
                  0.5.0's. With it forced the two trees must agree exactly, or something else
                  on the path is live and the arm is void.
  --break-mask N  zero N entries of repr_x_mask so the two conventions genuinely differ. D94
                  measured 0 differing tokens on this batch, which makes the forced-knob control
                  vacuous; this is the control that can actually break.
  --policy        write out an autocast policy by hand. `torch.amp.autocast("cuda", ...)` is
                  inert on CPU, so the revisions' two precision changes cannot be reached by
                  running the trees as-is. D93's pass-176 result applies: a flag that selects a
                  dtype policy can have its policy written out. `p043` runs embed_zij in fp32
                  and the confidence pairformer stack at ambient dtype; `p050` runs embed_zij at
                  ambient dtype and the stack in fp32; `both` and `neither` are the corners.

usage: run_aux.py <TREE> <TAG> <OUT.pt> <f64|f32|bf16> [knobs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BOUND = "/tmp/of3t/auxheads043/boundary_aux_heads.pt"
CKPT = "/home/moritz/.boltz/of3-p2-155k.pt"

STASH: dict = {}


def move(x, dtype):
    if torch.is_tensor(x):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: move(v, dtype) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move(v, dtype) for v in x)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("tag")
    ap.add_argument("out", type=Path)
    ap.add_argument("dtype", choices=["f64", "f32", "bf16"])
    ap.add_argument("--single-mask", choices=["native", "repr", "token"], default="native")
    ap.add_argument("--break-mask", type=int, default=0)
    ap.add_argument("--policy", choices=["native", "p043", "p050", "both", "neither"],
                    default="native")
    ap.add_argument("--transpose-bias", choices=["native", "off", "on"], default="native",
                    help="0.5.0 adds transpose_bias=True to the pairformer block's end-node "
                         "triangle attention. Forcing it matches the one structural difference "
                         "that survives float64; with it forced the trees must agree exactly.")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, a.tree)
    import openfold3
    assert openfold3.__file__.startswith(a.tree), openfold3.__file__

    import openfold3.core.model.layers.triangular_attention as TA
    from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    import openfold3.core.model.heads.head_modules as HM
    import openfold3.core.model.heads.prediction_heads as PH

    if a.transpose_bias != "native":
        want = a.transpose_bias == "on"
        _orig_ta = TA.TriangleAttention.forward

        def _ta(self, *args, **kw):
            kw["transpose_bias"] = want
            try:
                return _orig_ta(self, *args, **kw)
            except TypeError:
                # 0.4.3 has no such parameter at all, so "off" is already its behaviour and
                # "on" is unreachable there. Say so rather than silently running the default.
                kw.pop("transpose_bias")
                if want:
                    raise SystemExit("this tree has no transpose_bias parameter; --transpose-bias "
                                     "on is unreachable here")
                return _orig_ta(self, *args, **kw)

        TA.TriangleAttention.forward = _ta

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    dt = {"f64": torch.float64, "f32": torch.float32, "bf16": torch.bfloat16}[a.dtype]

    # ---- knob 1: the head_modules single_mask difference -----------------------------------
    _orig_get = HM.get_token_representative_atoms

    def _get(batch, x, atom_mask):
        rx, rm = _orig_get(batch=batch, x=x, atom_mask=atom_mask)
        if a.break_mask:
            rm = rm.clone()
            flat = rm.reshape(-1, rm.shape[-1])
            flat[..., : a.break_mask] = 0
            rm = flat.reshape(rm.shape)
        STASH["repr_mask"] = rm.detach().clone()
        STASH["token_mask"] = batch["token_mask"].detach().clone()
        return rx, rm

    HM.get_token_representative_atoms = _get

    _orig_pe = PH.PairformerEmbedding.forward

    def _pe(self, *args, **kw):
        if a.single_mask != "native":
            want = STASH["repr_mask"] if a.single_mask == "repr" else STASH["token_mask"]
            kw["single_mask"] = want.to(kw["single_mask"].dtype)
        STASH["used_mask_sum"] = float(kw["single_mask"].double().sum())
        return _orig_pe(self, *args, **kw)

    PH.PairformerEmbedding.forward = _pe

    # ---- knob 2: the prediction_heads precision policy, written out by hand ------------------
    if a.policy != "native":
        hi = torch.float32
        _orig_embed = PH.PairformerEmbedding.embed_zij
        _orig_stack_call = PH.PairformerEmbedding.pairformer_emb

        embed_hi = a.policy in ("p043", "both")
        stack_hi = a.policy in ("p050", "both")

        def _embed(self, si_input, zij, x_pred):
            if not embed_hi:
                return _orig_embed(self, si_input=si_input, zij=zij, x_pred=x_pred)
            orig = zij.dtype
            saved = {n: p.data for n, p in self.named_parameters()}
            for n, p in self.named_parameters():
                p.data = p.data.to(hi)
            try:
                out = _orig_embed(self, si_input=si_input.to(hi), zij=zij.to(hi),
                                  x_pred=x_pred.to(hi))
            finally:
                for n, p in self.named_parameters():
                    p.data = saved[n]
            return out.to(orig)

        PH.PairformerEmbedding.embed_zij = _embed

        if stack_hi:
            _orig_stack_fwd = None

            def _emb(self, *args, **kw):
                nonlocal _orig_stack_fwd
                stack = self.pairformer_stack
                if _orig_stack_fwd is None:
                    _orig_stack_fwd = type(stack).forward

                def _hi_fwd(slf, *aa, **kk):
                    orig_dt = aa[0].dtype if aa else next(iter(kk.values())).dtype
                    saved = {n: p.data for n, p in slf.named_parameters()}
                    for n, p in slf.named_parameters():
                        p.data = p.data.to(hi)
                    try:
                        aa = tuple(x.to(hi) if torch.is_tensor(x) and x.is_floating_point()
                                   else x for x in aa)
                        kk = {k: (v.to(hi) if torch.is_tensor(v) and v.is_floating_point()
                                  else v) for k, v in kk.items()}
                        o = _orig_stack_fwd(slf, *aa, **kk)
                    finally:
                        for n, p in slf.named_parameters():
                            p.data = saved[n]
                    return tuple(x.to(orig_dt) if torch.is_tensor(x) else x for x in o)

                type(stack).forward = _hi_fwd
                try:
                    return _orig_stack_call(self, *args, **kw)
                finally:
                    type(stack).forward = _orig_stack_fwd

            PH.PairformerEmbedding.pairformer_emb = _emb

    # ---- build, load, run -------------------------------------------------------------------
    b = torch.load(BOUND, map_location="cpu", weights_only=False)
    kw = dict(b["inputs"]["kwargs"])
    del b

    cfg = model_config.architecture.heads
    head = AuxiliaryHeadsAllAtom(config=cfg).eval()

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    pre = "aux_heads."
    sub = {k[len(pre):]: v.to(torch.float64) for k, v in sd.items()
           if k.startswith(pre) and torch.is_tensor(v)}
    del sd
    inc = head.load_state_dict(sub, strict=False)
    key_gate = {"loaded": len(sub), "missing": len(inc.missing_keys),
                "unexpected": len(inc.unexpected_keys),
                "missing_sample": list(inc.missing_keys)[:5],
                "unexpected_sample": list(inc.unexpected_keys)[:5]}
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE: {len(inc.unexpected_keys)} unexpected tensors -- this is "
                         f"not the revision this checkpoint belongs to")

    head = head.to(dt)
    kw = move(kw, dt)

    with torch.no_grad():
        out = head(**kw)

    res = {k: v.detach().to(torch.float64).clone() for k, v in out.items() if torch.is_tensor(v)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, a.out)

    rm, tm = STASH["repr_mask"].double(), STASH["token_mask"].double()
    rep = {
        "tag": a.tag, "tree": a.tree, "dtype": a.dtype,
        "knobs": {"single_mask": a.single_mask, "break_mask": a.break_mask,
                  "policy": a.policy, "transpose_bias": a.transpose_bias},
        "masks": {"repr_mask_sum": float(rm.sum()), "token_mask_sum": float(tm.sum()),
                  "n_differing_tokens": int((rm != tm).sum()),
                  "single_mask_sum_used": STASH["used_mask_sum"]},
        "key_gate": key_gate,
        "norms": {k: float(v.norm()) for k, v in res.items()},
        "shapes": {k: list(v.shape) for k, v in res.items()},
        "out": str(a.out),
    }
    print(json.dumps(rep, indent=1), flush=True)
    Path(str(a.out) + ".json").write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
