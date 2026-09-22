#!/usr/bin/env python3
"""ONE upstream 0.4.3 PairFormerBlock, run on the capture's own operands.

`/home/ttuser/of3t_gradients/cap/block47_boundary.pt` is a capture of block 47's own call in the
model's float64 run at crop 384: `args` is the block's input (s, z), `cot` is the cotangent that
arrived at its output, and `grad` is the block's own 57 parameter gradients, all float64. So a
single block is a complete, self-consistent, frame-matched experiment at the crop the trunk is
worst at, and it costs 1/48 of the stack.

Two things this settles that the 48-block arm cannot afford to:

  1. Running upstream's own block in float64 on `args` with `cot` must reproduce `grad`. That is
     the harness validation, and it is also deliverable 2's all-float64 control at its strongest:
     if float64 here does not reproduce the reference, nothing downstream is readable.
  2. Scoring `grad` against `grads_f64_043.pt`'s block 47 measures the distance between two
     float64 recordings of the same quantity, with no device op and no bf16 anywhere.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import torch

CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
PRE = "pairformer_stack.blocks."
CAP = "/home/ttuser/of3t_gradients/cap/block47_boundary.pt"
PIN = "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4"
DT = {"f64": torch.float64, "f32": torch.float32, "bf16": torch.bfloat16}


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_block(sd, k, dtype, tree):
    sys.path.insert(0, tree)
    import openfold3
    if not openfold3.__file__.startswith(tree):
        raise SystemExit("wrong tree: %s" % openfold3.__file__)
    from openfold3.core.model.latent.pairformer import PairFormerBlock
    p0 = PRE + "0."
    c_s = sd[p0 + "attn_pair_bias.layer_norm_a.weight"].shape[0]
    c_z = sd[p0 + "pair_stack.tri_mul_in.layer_norm_in.weight"].shape[0]
    nhb = sd[p0 + "attn_pair_bias.linear_z.weight"].shape[0]
    nhp = sd[p0 + "pair_stack.tri_att_start.linear_z.weight"].shape[0]
    dims = dict(c_s=c_s, c_z=c_z, c_hidden_pair_bias=c_s // nhb, no_heads_pair_bias=nhb,
                c_hidden_mul=sd[p0 + "pair_stack.tri_mul_in.linear_a_p.weight"].shape[0],
                c_hidden_pair_att=sd[p0 + "pair_stack.tri_att_start.mha.linear_q.weight"].shape[0] // nhp,
                no_heads_pair=nhp, transition_type="swiglu",
                transition_n=sd[p0 + "pair_stack.pair_transition.swiglu.linear_a.weight"].shape[0] // c_z,
                pair_dropout=0.25, fuse_projection_weights=False, inf=1e9)
    pk = "%s%d." % (PRE, k)
    sub = {kk[len(pk):]: v.to(dtype) for kk, v in sd.items() if kk.startswith(pk)}
    m = PairFormerBlock(**dims).to(dtype)
    m.load_state_dict(sub, strict=True)
    return m.eval(), dims, len(sub), openfold3.__file__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/home/ttuser/of3t_trunk043ref/of3pkg043")
    ap.add_argument("--block", type=int, default=47)
    ap.add_argument("--policy", default="f64", choices=tuple(DT))
    ap.add_argument("--cap", default=CAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
    from score import score

    dtype = DT[a.policy]
    cap = torch.load(a.cap, map_location="cpu", weights_only=False)
    s_in, z_in = (t.to(dtype).clone() for t in cap["args"])
    cot_s, cot_z = (t.to(torch.float64) for t in cap["cot"])
    kw = {k: (v.to(dtype) if hasattr(v, "to") and v.is_floating_point() else v)
          for k, v in cap["kwargs"].items()}
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    mod, dims, nload, of3file = build_block(sd, a.block, dtype, a.tree)

    s_in.requires_grad_(True); z_in.requires_grad_(True)
    out = mod(s_in, z_in, **kw)
    s_out, z_out = out if isinstance(out, (tuple, list)) else (out, None)
    torch.autograd.backward([s_out, z_out],
                            [cot_s.to(dtype), cot_z.to(dtype)])
    g = {n: (p.grad.detach().to(torch.float64).clone() if p.grad is not None else None)
         for n, p in mod.named_parameters()}
    torch.save({"grads": g, "block": a.block, "policy": a.policy, "cap": a.cap,
                "s_out": s_out.detach().to(torch.float64), "tree": a.tree}, a.out)

    rep = {"what": __doc__.strip().splitlines()[0], "host": os.uname().nodename,
           "block": a.block, "policy": a.policy, "tree": a.tree, "openfold3_file": of3file,
           "cap": a.cap, "cap_sha256": sha256_file(a.cap), "tensors_loaded": nload,
           "probe": {"s_in_norm": float(s_in.detach().to(torch.float64).norm()),
                     "z_in_norm": float(z_in.detach().to(torch.float64).norm()),
                     "cot_s_norm": float(cot_s.norm()), "cot_z_norm": float(cot_z.norm()),
                     "s_out_norm": float(s_out.detach().to(torch.float64).norm())},
           "seconds": round(time.perf_counter() - t0, 1), "scores": {}}

    CAPG = {k: v for k, v in cap["grad"].items() if v is not None}
    keys = sorted(set(g) & set(CAPG))
    s = score({k: g[k] for k in keys}, {k: CAPG[k] for k in keys}, keys)
    s.pop("_rows")
    rep["scores"]["this_arm_vs_CAPTURE_own_grad"] = {
        "compared": len(keys), **{k: s[k] for k in ("mass_weighted_rel_l2",
        "median_rel_l2_over_tensors", "mass_weighted_norm_ratio", "mass_weighted_cos",
        "reference_squared_norm", "worst_by_rel", "worst_by_error_mass")}}

    d = sha256_file("/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt")
    if d != PIN:
        raise SystemExit("model reference digest %s != pin" % d)
    M = torch.load("/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt",
                   map_location="cpu", weights_only=False)
    M = M["grads"] if isinstance(M, dict) and "grads" in M else M
    pk = "%s%d." % (PRE, a.block)
    M47 = {k[len(pk):]: v for k, v in M.items() if k.startswith(pk)}
    rep["model_reference"] = {"path": "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt",
                              "sha256": d, "matches_pin": True}
    for nm, arm in (("this_arm", g), ("CAPTURE_own_grad", CAPG)):
        kk = sorted(set(arm) & set(M47))
        s = score({k: arm[k] for k in kk}, {k: M47[k] for k in kk}, kk)
        s.pop("_rows")
        rep["scores"]["%s_vs_MODEL_f64" % nm] = {"compared": len(kk), **{k: s[k] for k in
            ("mass_weighted_rel_l2", "median_rel_l2_over_tensors", "mass_weighted_norm_ratio",
             "mass_weighted_cos", "reference_squared_norm", "worst_by_error_mass")}}
    json.dump(rep, open(a.report, "w"), indent=2)
    print(json.dumps({k: v["mass_weighted_rel_l2"] for k, v in rep["scores"].items()}, indent=1))
    print("seconds", rep["seconds"], "s_out_norm", rep["probe"]["s_out_norm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
