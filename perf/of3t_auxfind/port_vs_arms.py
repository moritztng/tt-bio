#!/usr/bin/env python3
"""The port's own forward against BOTH references: masked (R) and unmasked (U).

PREREGISTERED.md arms P and E, and the FIX_OR_NOT deliverable.

Four readings per head, one process, one card:

  port(no mask) vs R   the published A18 failure, re-measured on the shipped `forward`.
  port(no mask) vs U   the SAME device tensors against the unmasked float64 arm, i.e. the
                       function the unmasked port is actually trying to compute. This is the
                       port's own ARITHMETIC with the mask defect taken out of the comparison.
  port(mask)    vs R   the fix, scored against the reference. This is the A18 re-read.
  port(mask)    vs U   the fix against the wrong reference, which must get WORSE if the mask
                       argument does anything -- the control that breaks the comparison.

Arms R and U are rebuilt here in the same process from upstream 0.4.3 in float64, so nothing
is carried across a file boundary. A16's zero model is measured on every scope.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

BOUND = "/home/ttuser/of3t_auxheads/cap043/boundary_aux_heads.pt"
CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
TREE = "/home/ttuser/of3t_rebase/of3pkg043"
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
    a, b = ours.flatten().double(), ref.flatten().double()
    na, nb = float(a.norm()), float(b.norm())
    return {"rel_l2": float((a - b).norm() / (nb + 1e-300)),
            "r": (na / nb if nb else float("nan")),
            "cos": (float(torch.dot(a, b) / (a.norm() * b.norm())) if na and nb else float("nan"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.set_num_threads(a.threads)

    sys.path.insert(0, TREE)
    import openfold3
    assert openfold3.__file__.startswith(TREE), openfold3.__file__
    from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
    import openfold3.core.model.heads.prediction_heads as PH
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    from openfold3.core.utils.atomize_utils import (
        broadcast_token_feat_to_atoms, get_token_representative_atoms)

    B = torch.load(BOUND, map_location="cpu", weights_only=False)
    kw = dict(B["inputs"]["kwargs"])
    ref_out = {k: v.double().clone() for k, v in B["outputs"].items() if torch.is_tensor(v)}
    del B
    batch = kw["batch"]
    token_mask = batch["token_mask"].reshape(-1).float().clone()
    n_tok, n_real = int(token_mask.numel()), int(token_mask.sum())
    tokm = token_mask.bool()

    def move(x):
        if torch.is_tensor(x):
            return x.double() if x.is_floating_point() else x
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(move(v) for v in x)
        return x
    kw64 = move(kw)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    del sd

    head64 = AuxiliaryHeadsAllAtom(config=model_config.architecture.heads).eval()
    inc = head64.load_state_dict({k: v.double() for k, v in aux.items()}, strict=False)
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE: {len(inc.unexpected_keys)} unexpected")
    key_gate = {"loaded": len(aux), "missing": len(inc.missing_keys), "unexpected": 0}
    head64 = head64.double()

    FORCE = {"mode": "native"}
    _orig = PH.PairformerEmbedding.forward

    def _pe(self, *args, **kwargs):
        if FORCE["mode"] == "ones":
            kwargs["single_mask"] = torch.ones_like(kwargs["single_mask"])
            kwargs["pair_mask"] = torch.ones_like(kwargs["pair_mask"])
        return _orig(self, *args, **kwargs)
    PH.PairformerEmbedding.forward = _pe

    with torch.no_grad():
        R = {k: v.detach().double().clone() for k, v in head64(**kw64).items()}
    FORCE["mode"] = "ones"
    with torch.no_grad():
        U = {k: v.detach().double().clone() for k, v in head64(**kw64).items()}
    del head64
    print(f"[{time.perf_counter()-t0:.0f}s] arms R and U built", flush=True)

    same = {k: stats(sq(R[k], 2 if k in ATOM_HEADS else 3),
                     sq(ref_out[k], 2 if k in ATOM_HEADS else 3))["rel_l2"] for k in HEADS}
    print(f"  control same-function worst {max(same.values()):.4e}", flush=True)

    # ---- the port ---------------------------------------------------------------------------
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_confidence import OF3ConfidenceHead

    outd = kw["output"]
    si_input = sq(kw["si_input"], 2).float()
    si_trunk = sq(outd["si_trunk"], 2).float()
    zij_trunk = sq(outd["zij_trunk"], 3).float()
    xpred = outd["atom_positions_predicted"].to(dtype=outd["si_trunk"].dtype)
    repr_x, repr_mask = get_token_representative_atoms(
        batch=batch, x=xpred, atom_mask=batch["atom_mask"])
    repr_x = sq(repr_x, 2).float()
    repr_mask = repr_mask.reshape(-1).float()
    max_atom_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"], num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["token_mask"], max_num_atoms_per_token=23).reshape(-1).float()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    port = OF3ConfidenceHead(aux, dev, ckc)

    def run(tm):
        with torch.no_grad():
            o = port.forward(si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
                             repr_x_pred=repr_x, max_atom_per_token_mask=max_atom_mask,
                             use_zij_trunk_embedding=bool(kw.get("use_zij_trunk_embedding", True)),
                             token_mask=tm,
                             single_mask=(repr_mask if tm is not None else None))
        return {k: v.detach().double().clone() for k, v in o.items() if torch.is_tensor(v)}

    P0 = run(None)
    print(f"[{time.perf_counter()-t0:.0f}s] port unmasked done", flush=True)
    P1 = run(token_mask)
    print(f"[{time.perf_counter()-t0:.0f}s] port masked done", flush=True)

    rows = []
    for k in HEADS:
        rank = 2 if k in ATOM_HEADS else 3
        r, u = sq(R[k], rank), sq(U[k], rank)
        p0, p1 = sq(P0[k], rank), sq(P1[k], rank)
        if k in PAIR_HEADS:
            cut = lambda t: t[tokm][:, tokm]
            scope = f"pair, real block {n_real}x{n_real} of {n_tok}x{n_tok}"
            pad = 1.0 - n_real / n_tok
        else:
            cut = lambda t: t
            scope = "atom-gathered, 422 of 422 atoms"
            pad = 0.0
        row = {"head": k, "scope": scope, "padding_fraction": pad,
               "port_nomask_vs_R": stats(cut(p0), cut(r)),
               "port_nomask_vs_U": stats(cut(p0), cut(u)),
               "port_mask_vs_R": stats(cut(p1), cut(r)),
               "port_mask_vs_U": stats(cut(p1), cut(u)),
               "U_vs_R": stats(cut(u), cut(r)),
               "A16_zero_model_vs_R": stats(torch.zeros_like(cut(r)), cut(r))["rel_l2"]}
        rows.append(row)
        print(f"  {k:32s} noMask/R {row['port_nomask_vs_R']['rel_l2']:.6e}  "
              f"noMask/U {row['port_nomask_vs_U']['rel_l2']:.6e}  "
              f"MASK/R {row['port_mask_vs_R']['rel_l2']:.6e}  "
              f"MASK/U {row['port_mask_vs_U']['rel_l2']:.6e}", flush=True)

    rep = {"boundary": BOUND, "n_tokens": n_tok, "n_tokens_real": n_real, "bar": BAR,
           "key_gate": key_gate, "control_same_function": same,
           "dtype_policy_A27": {
               "R and U": "upstream 0.4.3, float64 parameters and activations, no cast; "
                          "autocast('cuda') inert on CPU",
               "port": "host fp32 s-path and heads, device bf16 z-path, HiFi4 + fp32 dest acc"},
           "heads": rows,
           "a18_nomask": {"n_over_bar": sum(1 for r in rows
                                            if r["port_nomask_vs_R"]["rel_l2"] > BAR),
                          "worst": max(r["port_nomask_vs_R"]["rel_l2"] for r in rows)},
           "a18_mask": {"n_over_bar": sum(1 for r in rows
                                          if r["port_mask_vs_R"]["rel_l2"] > BAR),
                        "worst": max(r["port_mask_vs_R"]["rel_l2"] for r in rows)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"[{time.perf_counter()-t0:.0f}s] A18 no-mask {rep['a18_nomask']['n_over_bar']}/5 over "
          f"bar worst {rep['a18_nomask']['worst']:.4e}; MASKED "
          f"{rep['a18_mask']['n_over_bar']}/5 over bar worst {rep['a18_mask']['worst']:.4e}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
