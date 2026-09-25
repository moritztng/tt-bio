#!/usr/bin/env python3
"""of3t-confpfe bisection step 3: the confidence head's forward, split at its input boundary.

    conf_split.py --dump C.pt --batch B64.pt --batch-sha256 S --checkpoint CK --out OUT.json

C.pt is devstep.py --conf-dump's: the device step's own confidence-head inputs and outputs.
Three things are compared on the real scope only (real tokens, real x real pairs), all in
upstream 0.4.3 at float64, 64 wide (the head is masked, so the real block does not depend on
the width; shown for resolved to every digit by the smoke run):

  inputs   device s_input / s_trunk / z_trunk against upstream's float64 trunk on this batch;
  head     device outputs against upstream's head run on the DEVICE's inputs and structure,
           which is the head's own arithmetic error in this call;
  carried  upstream's head on the device's inputs against it on float64's inputs, at the same
           structure: the input error the head carries forward.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_fullstep64"))
import ref_step  # noqa: E402  (sets up bundle_min and the repo on sys.path)

bm = ref_step.bm
OUT_KEYS = ("resolved_logits", "plddt_logits", "pae_logits", "pde_logits", "si_conf", "zij_conf")
DEV_KEYS = {"resolved_logits": "experimentally_resolved_logits"}


def heads(model, s_input, s, z, repr_x, tok, chunk):
    pe, ah = model.aux_heads.pairformer_embedding, model.aux_heads
    pair = tok[..., None] * tok[..., None, :]
    si_c, zij_c = pe.pairformer_emb(si_input=s_input, si=s, zij=z, x_pred=repr_x[None],
                                    single_mask=tok, pair_mask=pair, chunk_size=chunk,
                                    _mask_trans=True)
    out = {"si_conf": si_c, "zij_conf": zij_c, "pde_logits": ah.pde(zij_c),
           "pae_logits": ah.pae(zij_c)}
    for name, head in (("resolved_logits", ah.experimentally_resolved), ("plddt_logits", ah.plddt)):
        out[name] = head.linear(head.layer_norm(si_c))
    return out


def real(t, n, pair):
    t = t.reshape(-1, *t.shape[-3:]) if pair else t.reshape(-1, *t.shape[-2:])
    return t[0, :n, :n] if pair else t[0, :n]


def rel(a, b):
    a, b = a.double(), b.double()
    return {"rel": float((a - b).norm() / b.norm()),
            "cos": float((a.flatten() @ b.flatten()) / (a.norm() * b.norm())),
            "norm_ratio": float(a.norm() / b.norm())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--batch-sha256", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--build-seed", type=int, default=20260919)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--self-control", action="store_true",
                    help="replace the dump's inputs and outputs with float64's own, cast to "
                         "float32 as the dump is: every figure must read at float32 rounding")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if ref_step.sha256_file(a.batch) != a.batch_sha256:
        raise SystemExit("batch sha256 mismatch")
    bm.pin_deterministic_kernels(True)
    dt = torch.float64
    cfg, model, _dropout, _ck = ref_step.load(dt, a.checkpoint, a.build_seed, None)
    batch = bm.move(torch.load(a.batch, weights_only=False), "cpu", dt)
    tok = batch["token_mask"]                              # [1, W]
    W, n = tok.shape[-1], int(tok.sum())
    d = torch.load(a.dump, weights_only=False)
    if not a.self_control:
        din = {k: v.to(dt).reshape(1, *v.shape[-3:] if k == "z_trunk" else v.shape[-2:])
               for k, v in d["inputs"].items()}
        din = {k: (v[:, :W, :W] if k == "z_trunk" else v[:, :W]) for k, v in din.items()}
    repr_x = d["repr_x"][:W].to(dt)

    rec = {"dump": str(a.dump), "batch": a.batch.name, "width": W, "real_tokens": n,
           "input_dtypes_on_device": d.get("dtypes")}
    with torch.no_grad():
        s_input, s, z = model.run_trunk(batch=batch, num_cycles=1, inplace_safe=False)
        ref_in = {"s_input": s_input, "s_trunk": s, "z_trunk": z}
        at_f64 = heads(model, s_input, s, z, repr_x, tok, None)
        if not a.self_control:
            rec["inputs"] = {k: rel(real(din[k], n, k == "z_trunk"),
                                    real(ref_in[k], n, k == "z_trunk")) for k in ref_in}
            at_dev = heads(model, din["s_input"], din["s_trunk"], din["z_trunk"], repr_x, tok,
                           None)
            dev_out = d["outputs"]
    if a.self_control:
        with torch.no_grad():
            f32 = lambda t: t.float().to(dt)
            din = {k: f32(v) for k, v in ref_in.items()}
            rec["inputs"] = {k: rel(real(din[k], n, k == "z_trunk"),
                                    real(ref_in[k], n, k == "z_trunk")) for k in ref_in}
            at_dev = heads(model, din["s_input"], din["s_trunk"], din["z_trunk"], repr_x, tok, None)
            dev_out = {DEV_KEYS.get(k, k): f32(v) for k, v in at_f64.items()}
        rec["self_control"] = True
    pair_keys = {"pae_logits", "pde_logits", "zij_conf"}
    rec["head"], rec["carried"], rec["total"] = {}, {}, {}
    for k in OUT_KEYS:
        p = k in pair_keys
        dv = real(dev_out[DEV_KEYS.get(k, k)].to(dt), n, p)
        rec["head"][k] = rel(dv, real(at_dev[k], n, p))
        rec["carried"][k] = rel(real(at_dev[k], n, p), real(at_f64[k], n, p))
        rec["total"][k] = rel(dv, real(at_f64[k], n, p))
    # Where in the head: upstream's float64 s-heads applied to the DEVICE's own si_conf. Against
    # the device logits this is the heads' final LayerNorm + Linear; against float64 at the
    # device inputs it is what si_conf's error alone carries into the logits.
    ah = model.aux_heads
    si_dev = dev_out["si_conf"].to(dt).reshape(-1, dev_out["si_conf"].shape[-1])[:W][None]
    rec["post"] = {}
    with torch.no_grad():
        for k, head in (("resolved_logits", ah.experimentally_resolved), ("plddt_logits", ah.plddt)):
            lg = head.linear(head.layer_norm(si_dev))
            dv = real(dev_out[DEV_KEYS.get(k, k)].to(dt), n, False)
            rec["post"][k] = {"final_ln_linear": rel(dv, real(lg, n, False)),
                              "si_conf_carries": rel(real(lg, n, False), real(at_dev[k], n, False))}
        ln_dev = ah.experimentally_resolved.layer_norm(si_dev)
        ln_ref = ah.experimentally_resolved.layer_norm(at_dev["si_conf"])
        rec["post"]["si_conf_after_layernorm"] = rel(real(ln_dev, n, False), real(ln_ref, n, False))
        sr, sd_ = real(at_dev["si_conf"], n, False), real(si_dev, n, False)
        rec["post"]["si_conf_centred"] = rel(sd_ - sd_.mean(-1, keepdim=True),
                                             sr - sr.mean(-1, keepdim=True))
        rec["post"]["si_conf_row_mean_over_std"] = float((sr.mean(-1).abs() / sr.std(-1)).median())
    # The atom-slot heads on the slots that exist. Our layout carries 23 slots per token and the
    # loss weights them by the atom mask; upstream gathers exactly these (max_atom_per_token_mask).
    napt = batch["num_atoms_per_token"].reshape(-1)[:n].long()
    slot = (torch.arange(23)[None, :] < napt[:, None])            # [n, 23]
    rec["existing_atoms"] = {"n_slots": int(slot.sum())}
    for k, c in (("resolved_logits", 2), ("plddt_logits", 50)):
        dv = real(dev_out[DEV_KEYS.get(k, k)].to(dt), n, False).reshape(n, 23, c)[slot]
        rf = real(at_dev[k], n, False).reshape(n, 23, c)[slot]
        f6 = real(at_f64[k], n, False).reshape(n, 23, c)[slot]
        rec["existing_atoms"][k] = {"head": rel(dv, rf), "total": rel(dv, f6)}
    json.dump(rec, open(a.out, "w"), indent=1)
    print("existing_atoms", rec["existing_atoms"])
    for k, v in rec["post"].items():
        print("post", k, v)
    for part in ("inputs", "head", "carried", "total"):
        for k, v in rec[part].items():
            print(f"{part:8s} {k:16s} rel {v['rel']:.4e} cos {v['cos']:.6f} r {v['norm_ratio']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
