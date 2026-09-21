#!/usr/bin/env python3
"""Instrument A on the INPUT EMBEDDER's atom-encoder leg: our device leg against the reference's
float64, seeded with THEIR cotangent at the leg's own two output boundaries.

The nine tensors `READABLE_MASS.json` files as HOST_APPLIED in `input_embedder` are blocked
twice: the leg ran on the host, AND no device arm covers the section on the model's own batch.
`of3t-hostleg` removed the first blocker; this is the second. Two boundaries, because the
shipped leg's pair completion (`linear_l`, `linear_m`, `pair_mlp`) runs on the host between
`cl`/`plm` and `ai`, so a single end-to-end tape does not exist on our side:

  * `(cl, plm)`  -> the eight `ref_atom_feature_embedder` linears.
  * `(ai)` with `ql` as the captured input -> `linear_q.0.weight`, through the shipped
    `AtomEncoderTokenHead`: mask multiply, linear, relu, and the atom-to-token mean, all taped.

The parameter-free block inputs are DERIVED here by the shipped `ref_atom_block_inputs` from the
captured batch features, not read from the capture: freezing a shipped derivation into a fixture
would measure the fixture.

None of the nine is square, so the transpose the loader applies is recovered from the shape
without ambiguity -- which is the case D83 says the shape test is correct for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "of3t_tape"))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
ENC = "input_embedder.atom_attn_enc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", default="/home/ttuser/of3t_hostleg/ie_boundary.pt")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--act", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--break-cot", action="store_true", dest="break_cot",
                    help="roll the cotangent by one block along the atom/token axis. The same "
                         "control --permute-cot is on the diffusion arm: the forward, the "
                         "weights and the arithmetic are untouched, so a reading that does not "
                         "move is measuring something other than the gradient.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import openfold3_host_prep as HP
    from tt_bio.openfold3 import AtomEncoderTokenHead, RefAtomFeatureEmbedder
    from tt_bio.openfold3_weights import _sub
    from tt_bio.tenstorrent import device_dtype_override, get_device

    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    feats = {k: (v[0] if v.dim() > 1 and v.shape[0] == 1 and k != "ref_pos" else v)
             for k, v in B["features"].items()}
    # Their forward unsqueezes a sampling dim into every feature; drop leading singletons so the
    # shapes are the ones the shipped host prep and the device module take.
    def sq(t):
        while t.dim() > 1 and t.shape[0] == 1:
            t = t[0]
        return t
    feats = {k: sq(v) for k, v in B["features"].items()}
    atom_mask = feats["atom_mask"].float()
    n_atom = int(atom_mask.shape[-1])
    cl_ref, plm_ref = sq(B["cl"]), sq(B["plm"])
    ai_ref, ql_ref = sq(B["ai"]), sq(B["ql"])
    cot_cl, cot_plm, cot_ai = sq(B["cot_cl"]), sq(B["cot_plm"]), sq(B["cot_ai"])
    if a.break_cot:
        cot_cl = torch.roll(cot_cl, 1, dims=-2)
        cot_plm = torch.roll(cot_plm, 1, dims=-4)
        cot_ai = torch.roll(cot_ai, 1, dims=-2)
    n_token = int(ai_ref.shape[-2])
    print(f"[{time.perf_counter()-t0:.0f}s] boundary: n_atom={n_atom} n_token={n_token} "
          f"cl {tuple(cl_ref.shape)} plm {tuple(plm_ref.shape)} ai {tuple(ai_ref.shape)} "
          f"ql {tuple(ql_ref.shape)} loss {B['loss']}", flush=True)

    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    enc_sd = _sub(sd, ENC)
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    act = ttnn.float32 if a.act == "fp32" else ttnn.bfloat16

    # the bijection, by identity at the load: every weight is resolved out of the checkpoint
    # sub-dict first, so id(that tensor) -> checkpoint name maps all of them (the diffusion
    # arm's amendment item 1, same construction).
    reg = {}
    orig = ttnn.from_torch
    want = {id(v): f"{ENC}.{k}" for k, v in enc_sd.items() if torch.is_tensor(v)}

    def recording(tensor, *args, **kw):
        v = orig(tensor, *args, **kw)
        base = getattr(tensor, "_base", None)
        nm = want.get(id(tensor)) or (want.get(id(base)) if base is not None else None)
        if nm:
            reg[id(v)] = nm
        return v

    rows, fwd = {}, {}
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)

    def harvest(mod, out_d, cots, label):
        """One tape: forward the module, seed THEIR cotangent, keep every leaf's gradient."""
        walked = {k: v for k, v in _dw(mod).items()}
        for t in walked.values():
            ag.parameter(t)
        with device_dtype_override(act), ag.tape():
            res = out_d()
            outs = res if isinstance(res, (tuple, list)) else [res]
            ag.backward(list(outs), [ft(c) for c in cots])
        got = 0
        for t in walked.values():
            nm = reg.get(id(t))
            leaf = ag._PARAMS.get(id(t))
            g = getattr(leaf, "grad", None)
            if nm is None or g is None:
                continue
            gt = ttnn.to_torch(g.value if hasattr(g, "value") else g).double()
            wantshape = tuple(sd[nm].shape)
            gt = gt.reshape(gt.shape[-2:]) if gt.dim() > 2 else gt
            if tuple(gt.shape) != wantshape and tuple(gt.shape)[::-1] == wantshape:
                gt = gt.t().contiguous()
            if tuple(gt.shape) != wantshape:
                raise AssertionError(f"{nm}: {tuple(gt.shape)} is not {wantshape}")
            rows[nm] = gt
            got += 1
        print(f"[{time.perf_counter()-t0:.0f}s] {label}: {got} gradients", flush=True)
        return outs

    from of3_coverage import _device_weights as _dw

    # ---- boundary 1: the eight, seeded at (cl, plm) --------------------------------------
    ttnn.from_torch = recording
    try:
        with device_dtype_override(act):
            rafe = RefAtomFeatureEmbedder(_sub(enc_sd, "ref_atom_feature_embedder"), cfg)
            rafe_ins = HP.ref_atom_device_inputs(dev, feats, atom_mask)
    finally:
        ttnn.from_torch = orig
    outs = harvest(rafe, lambda: rafe(*rafe_ins), (cot_cl, cot_plm), "ref_atom_feature_embedder")
    for nm, ours, theirs in (("cl", outs[0], cl_ref), ("plm", outs[1], plm_ref)):
        o = ttnn.to_torch(ours.value if hasattr(ours, "value") else ours).double().reshape(-1)
        th = theirs.double().reshape(-1)
        n = min(o.numel(), th.numel())
        fwd[nm] = float(torch.linalg.vector_norm(o[:n] - th[:n])
                        / (torch.linalg.vector_norm(th[:n]) + 1e-300))

    # ---- boundary 2: linear_q.0.weight, seeded at ai -------------------------------------
    a2t = torch.zeros(n_token, n_atom)
    a2t[sq(feats["atom_to_token_index"]).long(), torch.arange(n_atom)] = atom_mask
    a2t = a2t / a2t.sum(-1, keepdim=True).clamp_min(1.0)
    amc = torch.zeros(1, n_atom, 1)
    amc[0, :, 0] = atom_mask
    ttnn.from_torch = recording
    try:
        with device_dtype_override(act):
            head = AtomEncoderTokenHead(_sub(enc_sd, "linear_q"), cfg)
            ql_d = ft(ql_ref[:n_atom].reshape(1, n_atom, -1))
            amc_d, mean_d = ft(amc), ft(a2t.unsqueeze(0))
    finally:
        ttnn.from_torch = orig
    outs = harvest(head, lambda: head(ql_d, amc_d, mean_d), (cot_ai,), "linear_q")
    o = ttnn.to_torch(outs[0].value if hasattr(outs[0], "value") else outs[0]).double()
    th = ai_ref.double()
    o = o.reshape(-1)[: th.numel()]
    fwd["ai"] = float(torch.linalg.vector_norm(o - th.reshape(-1))
                      / (torch.linalg.vector_norm(th) + 1e-300))

    # ---- score against the reference at THIS boundary (the model denominator is score_seventeen's)
    ref = B["grad_f64"]
    cmp_rows = []
    for nm, gt in sorted(rows.items()):
        r = ref.get(nm)
        if r is None or tuple(r.shape) != tuple(gt.shape):
            cmp_rows.append({"param": nm, "rel_l2": None,
                             "why": "absent from the boundary reference"})
            continue
        rd = r.double()
        rn = float(torch.linalg.vector_norm(rd))
        dn = float(torch.linalg.vector_norm(gt))
        cmp_rows.append({
            "param": nm, "ref_norm": rn, "device_norm": dn,
            "rel_l2": float(torch.linalg.vector_norm(gt - rd) / (rn + 1e-300)) if rn else None,
            "norm_ratio": (dn / rn) if rn else None,
            "cos": (float((gt * rd).sum() / (dn * rn)) if dn and rn else None)})
    meas = [r for r in cmp_rows if r.get("rel_l2") is not None]
    worst = max(meas, key=lambda r: r["rel_l2"]) if meas else None
    rep = {
        "instrument": "of3t-hostleg ie_arm.py -- the input-embedder atom-encoder leg on the "
                      "card, seeded with upstream 0.4.3's cotangent at its own two boundaries",
        "host": "qb1 (tt-quietbox) card 1, Blackhole p150a",
        "boundary": a.boundary,
        "boundary_loss": B["loss"],
        "break_cot": bool(a.break_cot),
        "act": a.act,
        "n_atom": n_atom, "n_token": n_token,
        "forward_rel_at_the_boundary": fwd,
        "n_gradients": len(rows),
        "n_measurable": len(meas),
        "n_over_5e-2": sum(1 for r in meas if r["rel_l2"] > 5.0e-2),
        "worst_rel_l2": worst["rel_l2"] if worst else None,
        "worst_tensor": worst["param"] if worst else None,
        "per_tensor": cmp_rows,
    }
    out = a.out or str(Path(__file__).with_name(f"IE_ARM{a.tag}.json"))
    Path(out).write_text(json.dumps(rep, indent=1) + "\n")
    if a.dump:
        torch.save({k: v.cpu() for k, v in rows.items()}, a.dump)
        print(f"wrote {len(rows)} gradient tensors to {a.dump}", flush=True)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_tensor"}, indent=1), flush=True)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
