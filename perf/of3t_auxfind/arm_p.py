#!/usr/bin/env python3
"""Can the SHIPPED Pairformer express the reference's masking? Arms R, P, U in float64 on CPU.

PREREGISTERED.md addendum (pass 177), branches T1/T2/T3 and controls 5/6/7.

  R   native masks, _mask_trans=True   -- what upstream 0.4.3 computes.
  P   native masks, _mask_trans=False  -- masked triangle ops, UNMASKED transitions, which is
                                          exactly what tenstorrent.py:PairformerLayer computes
                                          when handed `mask` and `attn_mask`.
  U   masks forced to ALL ONES         -- the unmasked port, carried over from mask_contrast.py.

P vs R is what threading the masks into `self.pf` still leaves behind. The three arms differ in
kwarg VALUES inside PairformerEmbedding.forward and in nothing else, so an accidental second
difference cannot hide in the contrast.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402

import torch

BOUND = "/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt"
CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
PAIR_HEADS = ["pae_logits", "pde_logits", "distogram_logits"]
ATOM_HEADS = ["plddt_logits", "experimentally_resolved_logits"]
HEADS = ATOM_HEADS + PAIR_HEADS
BAR = 5.0e-2


def sq(t, rank):
    while t.dim() > rank:
        assert t.shape[0] == 1, t.shape
        t = t[0]
    return t


def stats(ours, ref):
    """rel_l2 with the norm ratio and the error cosine beside it, never rel alone."""
    a, b = ours.flatten().double(), ref.flatten().double()
    nb, na = float(b.norm()), float(a.norm())
    return {"rel_l2": float((a - b).norm() / (nb + 1e-300)),
            "r": (na / nb if nb else float("nan")),
            "cos": (float(torch.dot(a, b) / (a.norm() * b.norm())) if na and nb else float("nan")),
            "ref_norm": nb, "our_norm": na}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=refpath.OF3PKG)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.set_num_threads(a.threads)

    refpath.install(a.tree)
    import openfold3
    print(f"REF_TREE resolved: {refpath.assert_resolved(a.tree)}", flush=True)
    assert openfold3.__file__.startswith(a.tree), openfold3.__file__
    from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
    import openfold3.core.model.heads.prediction_heads as PH
    from openfold3.projects.of3_all_atom.config.model_config import model_config

    B = torch.load(BOUND, map_location="cpu", weights_only=False)
    kw = dict(B["inputs"]["kwargs"])
    ref_out = {k: v.double().clone() for k, v in B["outputs"].items() if torch.is_tensor(v)}
    token_mask = kw["batch"]["token_mask"].reshape(-1).bool().clone()
    del B
    n_tok, n_real = int(token_mask.numel()), int(token_mask.sum())

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

    # ---- the substitution that makes the arms ------------------------------------------------
    # ARM["ones"]        -> force single_mask/pair_mask to all ones
    # ARM["mask_trans"]  -> the _mask_trans kwarg handed to PairformerEmbedding.forward
    ARM = {"ones": False, "mask_trans": True, "tag": "R"}
    SEEN = {}
    _orig = PH.PairformerEmbedding.forward

    def _pe(self, *args, **kwargs):
        sm, pm = kwargs["single_mask"], kwargs["pair_mask"]
        if ARM["ones"]:
            kwargs["single_mask"] = torch.ones_like(sm)
            kwargs["pair_mask"] = torch.ones_like(pm)
        kwargs["_mask_trans"] = ARM["mask_trans"]
        SEEN[ARM["tag"]] = {"single_sum": float(kwargs["single_mask"].double().sum()),
                            "pair_sum": float(kwargs["pair_mask"].double().sum()),
                            "single_sum_native": float(sm.double().sum()),
                            "pair_sum_native": float(pm.double().sum()),
                            "mask_trans": ARM["mask_trans"],
                            "reached": True}
        return _orig(self, *args, **kwargs)

    PH.PairformerEmbedding.forward = _pe

    # Control 5 needs to look where the flag CAN act. The heads are scored on the real block
    # and the gathered atoms by A18/D95, which is exactly the region a pointwise transition
    # cannot reach, so a zero there is not evidence the kwarg went missing. Capture the stack's
    # own s and z, where the padded positions are still visible.
    STACK = {}
    _stack = head.pairformer_embedding.pairformer_stack

    def _stack_hook(mod, args, kwargs, output):
        s_, z_ = output
        STACK[ARM["tag"]] = (s_.detach().double().clone(), z_.detach().double().clone())

    _sh = _stack.register_forward_hook(_stack_hook, with_kwargs=True)

    def run(tag, ones, mask_trans):
        ARM.update(tag=tag, ones=ones, mask_trans=mask_trans)
        with torch.no_grad():
            out = {k: v.detach().double().clone() for k, v in head(**kw).items()}
        s = SEEN[tag]
        print(f"[{time.perf_counter()-t0:.0f}s] arm {tag:9s} single_sum {s['single_sum']:.0f} "
              f"pair_sum {s['pair_sum']:.0f} _mask_trans={mask_trans}", flush=True)
        return out

    outR = run("R", False, True)
    outP = run("P", False, False)
    outU = run("U", True, True)
    # control 6: with the masks forced to ones, the flag must be a no-op
    outU_nt = run("U_nt", True, False)

    rep = {"boundary": BOUND, "n_tokens": n_tok, "n_tokens_real": n_real,
           "padding_fraction_tokens": 1.0 - n_real / n_tok,
           "dtype_policy_A27": ("every parameter and every activation float64, no cast on the "
                                "path, of3-p2-155k upcast once at load; autocast('cuda') is inert "
                                "on CPU and torch says so"),
           "key_gate": key_gate, "masks": SEEN, "bar": BAR,
           "arms": {"R": "native masks, _mask_trans=True (upstream 0.4.3)",
                    "P": "native masks, _mask_trans=False (what the shipped PairformerLayer "
                         "computes when handed mask + attn_mask)",
                    "U": "masks forced to all ones (the unmasked port)"}}

    # ---- control 7: same-function, re-run in THIS process, not carried in --------------------
    same = {}
    for k in HEADS:
        rank = 2 if k in ATOM_HEADS else 3
        r, o = sq(ref_out[k], rank), sq(outR[k], rank)
        same[k] = {"rel_l2": stats(o, r)["rel_l2"], "max_abs": float((o - r).abs().max())}
    rep["control_same_function"] = same
    worst_same = max(v["rel_l2"] for v in same.values())
    print(f"  control 7 same-function worst rel {worst_same:.4e}", flush=True)

    # ---- control 6: ones-masked arm is insensitive to the flag -------------------------------
    c6 = {}
    for k in HEADS:
        rank = 2 if k in ATOM_HEADS else 3
        c6[k] = {"max_abs": float((sq(outU[k], rank) - sq(outU_nt[k], rank)).abs().max())}
    rep["control_ones_mask_flag_is_noop"] = c6
    worst_c6 = max(v["max_abs"] for v in c6.values())
    print(f"  control 6 ones-mask flag no-op worst max_abs {worst_c6:.4e}", flush=True)

    # ---- the heads, P and U against R, on the A18 scope ---------------------------------------
    tokm = token_mask
    rows = []
    for k in HEADS:
        rank = 2 if k in ATOM_HEADS else 3
        r, p, u = sq(outR[k], rank), sq(outP[k], rank), sq(outU[k], rank)
        if k in PAIR_HEADS:
            r, p, u = r[tokm][:, tokm], p[tokm][:, tokm], u[tokm][:, tokm]
            scope = f"pair, real block {n_real}x{n_real}"
            pad = 0.0
        else:
            scope = f"atom-gathered ({r.shape[0]} of {r.shape[0]} atoms, no padding)"
            pad = 0.0
        row = {"head": k, "scope": scope, "padding_fraction_scored": pad,
               "P_vs_R": stats(p, r), "U_vs_R": stats(u, r), "P_vs_U": stats(p, u),
               "A16_zero_model": stats(torch.zeros_like(r), r)["rel_l2"]}
        row["P_over_bar"] = row["P_vs_R"]["rel_l2"] > BAR
        row["P_fraction_of_U"] = (row["P_vs_R"]["rel_l2"] / row["U_vs_R"]["rel_l2"]
                                  if row["U_vs_R"]["rel_l2"] else float("nan"))
        rows.append(row)
        print(f"  HEAD {k:32s} P/R {row['P_vs_R']['rel_l2']:.6e} "
              f"(r {row['P_vs_R']['r']:.6f} cos {row['P_vs_R']['cos']:.6f})  "
              f"U/R {row['U_vs_R']['rel_l2']:.6e}  P/U {row['P_fraction_of_U']:.4f}", flush=True)
    rep["heads"] = rows

    # ---- control 5 + the branch verdict -------------------------------------------------------
    _sh.remove()
    worst_p = max(r["P_vs_R"]["rel_l2"] for r in rows)
    n_over = sum(1 for r in rows if r["P_over_bar"])

    # Control 5, done where the flag can act. _mask_trans zeroes the two transitions' output on
    # PADDED positions. Both transitions are pointwise over the token axis, so that write can
    # only reach the scored real block through an op that mixes tokens -- and every such op
    # (both triangle multiplications, both triangle attentions, attention_pair_bias) is masked
    # in arm P. So the test is: does the flag move the PADDED region? If it does, the kwarg
    # arrived and the exact zero on the real block is a property of the model, not of the
    # instrument.
    sR, zR = STACK["R"]
    sP, zP = STACK["P"]
    sR, sP = sq(sR, 2), sq(sP, 2)
    zR, zP = sq(zR, 3), sq(zP, 3)
    pad = ~tokm
    c5 = {
     "z_real_block": stats(zP[tokm][:, tokm], zR[tokm][:, tokm]),
     "z_full_padded_layout": stats(zP, zR),
     "z_padded_rows_only": stats(zP[pad], zR[pad]),
     "s_real_tokens": stats(sP[tokm], sR[tokm]),
     "s_full_padded_layout": stats(sP, sR),
     "s_padded_rows_only": stats(sP[pad], sR[pad]),
     "z_max_abs_real_block": float((zP[tokm][:, tokm] - zR[tokm][:, tokm]).abs().max()),
     "z_max_abs_padded_rows": float((zP[pad] - zR[pad]).abs().max()),
     "s_max_abs_real_tokens": float((sP[tokm] - sR[tokm]).abs().max()),
     "s_max_abs_padded_rows": float((sP[pad] - sR[pad]).abs().max()),
    }
    flag_moved = c5["z_max_abs_padded_rows"] > 0.0 or c5["s_max_abs_padded_rows"] > 0.0
    c5["moved_where_it_can_act"] = flag_moved
    c5["untouched_on_scored_scope"] = (c5["z_max_abs_real_block"] == 0.0
                                       and c5["s_max_abs_real_tokens"] == 0.0)
    rep["control_flag_does_something"] = c5
    print(f"  control 5 flag moves padded rows: z {c5['z_max_abs_padded_rows']:.6e} "
          f"s {c5['s_max_abs_padded_rows']:.6e}; real block z "
          f"{c5['z_max_abs_real_block']:.6e} s {c5['s_max_abs_real_tokens']:.6e}", flush=True)
    frac = [r["P_fraction_of_U"] for r in rows if r["U_vs_R"]["rel_l2"] > 1e-12]
    within20 = bool(frac) and all(f >= 0.8 for f in frac)
    rep["branch"] = ("VOID-control5" if not flag_moved else ("T2" if n_over else "T1"))
    rep["T3_within_20pct_of_U"] = within20
    rep["verdict"] = {
        "heads_over_bar": n_over, "worst_P_vs_R": worst_p, "bar": BAR,
        "reachable_bar_A26": (2 ** 0.5) * BAR,
        "branch": rep["branch"],
        "T3": within20,
    }
    print(f"\n  BRANCH {rep['branch']}: {n_over} of {len(rows)} heads over {BAR:.1e}, "
          f"worst P/R {worst_p:.6e}; T3 (within 20% of U) = {within20}", flush=True)

    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(f"[{time.perf_counter()-t0:.0f}s] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
