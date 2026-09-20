#!/usr/bin/env python3
"""Run ONE upstream release tree's 48-block pairformer stack over the captured trunk boundary.

One tree per process: the two release trees define the same module names and cannot share an
interpreter. The weights are `of3-p2-155k` in every arm and nothing about them is a free choice.

DTYPE POLICY, written out because "float64" names a width and not a policy (PROTOCOL A27):

  f64        every parameter and every activation is float64. The checkpoint is upcast once at
             load. No cast anywhere on the path. Upstream's `_attention` wraps its scores in
             `torch.amp.autocast("cuda", ...)`, which is INERT on CPU tensors and torch says so
             out loud; so `use_high_precision_attention`, the one other thing that differs
             between these two revisions on this path, cannot fire in this arm. That is a limit
             of the arm, stated before the run, and the bf16 arm below is the one without it.
  f32        parameters and activations float32, same structure.
  bf16auto   float32 parameters under `torch.autocast("cpu", bfloat16)` -- upstream's own
             training recipe, the bar a bf16 port is actually measured against.

ATTENTION PRECISION. The one other thing that differs between these revisions on this path is
0.5.0 passing `use_high_precision_attention=True` into the single track's `AttentionPairBias`.
It reaches `_attention`, which wraps its scores in `torch.amp.autocast("cuda", float32)` -- inert
on CPU, so the flag cannot fire here and a float64 arm cannot see it. `--attn hp32` writes that
policy out by hand instead (D93's method: a flag selecting a dtype policy can have its policy
written out), casting q/k, the bias add and the softmax to float32 and the result back. Paired
with `--policy bf16auto` that is the regime where the difference is live. `--attn native32` is
its control: the same machinery selecting the tree's own policy, which must read bit-identical
to an unpatched run or the instrument is the finding.

TRANSPOSE BIAS. 0.5.0's `base_blocks.py` differs from 0.4.3's by exactly one added line,
`transpose_bias=True` on the ending-node triangle attention. `--tb off` forces it back off and
`--tb on` forces it on, by patching `TriangleAttention.forward` rather than by editing a tree.
Forcing 0.5.0 off must reproduce 0.4.3 bit for bit, and forcing 0.4.3 on must reproduce 0.5.0
bit for bit; those two are the convention-matched control, in both directions.

usage: ref_stack.py --tree <dir containing openfold3/> --out <OUT.pt> [--policy f64] [--tb keep]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
PRE = "pairformer_stack.blocks."


def build(sd, n, dtype):
    """Upstream's own PairFormerBlock, dimensions read off the checkpoint, strict load."""
    from openfold3.core.model.latent.pairformer import PairFormerBlock
    p0 = f"{PRE}0."
    c_s = sd[p0 + "attn_pair_bias.layer_norm_a.weight"].shape[0]
    c_z = sd[p0 + "pair_stack.tri_mul_in.layer_norm_in.weight"].shape[0]
    nh_bias = sd[p0 + "attn_pair_bias.linear_z.weight"].shape[0]
    nh_pair = sd[p0 + "pair_stack.tri_att_start.linear_z.weight"].shape[0]
    dims = dict(c_s=c_s, c_z=c_z, c_hidden_pair_bias=c_s // nh_bias, no_heads_pair_bias=nh_bias,
                c_hidden_mul=sd[p0 + "pair_stack.tri_mul_in.linear_a_p.weight"].shape[0],
                c_hidden_pair_att=sd[p0 + "pair_stack.tri_att_start.mha.linear_q.weight"].shape[0]
                // nh_pair,
                no_heads_pair=nh_pair, transition_type="swiglu",
                transition_n=sd[p0 + "pair_stack.pair_transition.swiglu.linear_a.weight"].shape[0]
                // c_z,
                pair_dropout=0.25, fuse_projection_weights=False, inf=1e9)
    mods, missing, unexpected, n_t = [], [], [], 0
    for i in range(n):
        sub = {k[len(f"{PRE}{i}."):]: v.to(dtype) for k, v in sd.items()
               if k.startswith(f"{PRE}{i}.")}
        m = PairFormerBlock(**dims).to(dtype)
        miss, unex = m.load_state_dict(sub, strict=True)
        missing += [f"{i}.{x}" for x in miss]
        unexpected += [f"{i}.{x}" for x in unex]
        n_t += len(sub)
        mods.append(m.eval())          # the capture disabled dropout; so does this
    return mods, dims, {"tensors_loaded": n_t, "missing": missing, "unexpected": unexpected}


def force_tb(value: bool):
    """Pin the ending-node bias orientation, whatever the tree's own call site passes."""
    import openfold3.core.model.layers.triangular_attention as ta
    orig = ta.TriangleAttention.forward

    def patched(self, *a, **kw):
        kw["transpose_bias"] = value
        return orig(self, *a, **kw)

    ta.TriangleAttention.forward = patched


def force_attn(policy: str):
    """Write the attention dtype policy out by hand, because the flag that selects it is a CUDA
    autocast context and CPU disables it."""
    import openfold3.core.model.primitives.attention as att
    soft = att.softmax_no_cast

    def patched(query, key, value, biases, use_high_precision=False):
        in_dtype = query.dtype
        dt = torch.float32 if policy == "hp32" else in_dtype
        scores = torch.einsum("...qc, ...kc->...qk", query.to(dt), key.to(dt))
        for b in biases:
            scores = scores + b.to(dt)
        scores = soft(scores, dim=-1)
        return torch.einsum("...qk, ...kc->...qc",
                            scores.to(value.dtype), value).to(in_dtype)

    att._attention = patched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--policy", default="f64", choices=("f64", "f32", "bf16auto"))
    ap.add_argument("--tb", default="keep", choices=("keep", "off", "on"))
    ap.add_argument("--attn", default="keep", choices=("keep", "native32", "hp32"))
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, a.tree)
    import openfold3
    if not openfold3.__file__.startswith(a.tree):
        raise SystemExit(f"wrong tree on sys.path: {openfold3.__file__} is not under {a.tree}")
    if a.tb != "keep":
        force_tb(a.tb == "on")
    if a.attn != "keep":
        force_attn(a.attn)

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    dt = torch.float32 if a.policy in ("f32", "bf16auto") else torch.float64

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    mods, dims, load = build(sd, a.blocks, dt)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    s = b["s_in"].to(dt).contiguous()
    z = b["z_in"].to(dt).contiguous()
    sm = b["single_mask"].to(dt)
    pm = b["pair_mask"].to(dt)

    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else torch.autocast("cpu", enabled=False))
    t1 = time.perf_counter()
    with ctx, torch.no_grad():
        for m in mods:
            s, z = m(s, z, sm, pm)
    s, z = s.to(torch.float64), z.to(torch.float64)
    torch.save({"s": s, "z": z, "policy": a.policy, "tb": a.tb, "tree": a.tree}, a.out)

    rep = {"what": __doc__.strip().splitlines()[0], "tree": a.tree,
           "openfold3_file": openfold3.__file__, "policy": a.policy, "tb": a.tb,
           "blocks": a.blocks, "dims": dims, "load": load, "attn_policy": a.attn,
           "dtype_policy": {
               "f64": "every parameter and every activation float64; checkpoint upcast once at "
                      "load; no cast on the path; upstream's CUDA-autocast contexts are inert on "
                      "CPU so use_high_precision_attention cannot fire",
               "f32": "parameters and activations float32, no autocast",
               "bf16auto": "float32 parameters under torch.autocast('cpu', bfloat16)",
           }[a.policy],
           "seconds": time.perf_counter() - t1,
           "s_norm": float(s.norm()), "z_norm": float(z.norm()),
           "out": a.out}
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({"tree": a.tree, "policy": a.policy, "tb": a.tb, "attn": a.attn,
                      "loaded": load["tensors_loaded"], "missing": len(load["missing"]),
                      "unexpected": len(load["unexpected"]),
                      "s_norm": rep["s_norm"], "z_norm": rep["z_norm"],
                      "seconds": round(rep["seconds"], 1),
                      "total_seconds": round(time.perf_counter() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
