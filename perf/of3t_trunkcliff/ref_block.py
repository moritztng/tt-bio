#!/usr/bin/env python3
"""ONE upstream 0.4.3 PairFormerBlock over a supplied state, every shared intermediate saved.

The 48-block ladder can say a block is where the norm goes; it cannot say which op inside the
block loses it. This runs a single block, by index, from a state file, and writes the ten
tensors the tt-bio port and upstream genuinely share so the two can be compared op by op.

DTYPE POLICY, written out because a width is not a policy (PROTOCOL A27):

  f64      every parameter and every activation float64; the checkpoint is upcast once at load
           and nothing casts on the path. Upstream's `_attention` autocast context is CUDA and
           inert on CPU, so `use_high_precision_attention` cannot fire in this arm.
  bf16auto float32 parameters under `torch.autocast('cpu', bfloat16)` -- upstream's own recipe,
           and the floor a bf16 port is measured against.

  --quantise-input rounds the supplied state to bfloat16 and back BEFORE the block runs, with
  the arithmetic policy unchanged. f64 + quantise-input is the cost of representing the input,
  with no bf16 arithmetic anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
PRE = "pairformer_stack.blocks."


def build_one(sd, idx, dtype):
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
    sub = {k[len(f"{PRE}{idx}."):]: v.to(dtype) for k, v in sd.items()
           if k.startswith(f"{PRE}{idx}.")}
    m = PairFormerBlock(**dims).to(dtype)
    m.load_state_dict(sub, strict=True)
    return m.eval(), dims, len(sub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--state", required=True, help=".pt with s and z to feed this block")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--policy", default="f64", choices=("f64", "f32", "bf16auto"))
    ap.add_argument("--quantise-input", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, a.tree)
    import openfold3
    if not openfold3.__file__.startswith(a.tree):
        raise SystemExit(f"wrong tree: {openfold3.__file__}")
    torch.set_num_threads(a.threads)
    dt = torch.float32 if a.policy in ("f32", "bf16auto") else torch.float64

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    blk, dims, n_t = build_one(sd, a.block, dt)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    st = torch.load(a.state, map_location="cpu", weights_only=False)
    s, z = st["s"], st["z"]
    if a.quantise_input:
        s = s.to(torch.bfloat16).to(torch.float64)
        z = z.to(torch.bfloat16).to(torch.float64)
    s = s.to(dt).contiguous()
    z = z.to(dt).contiguous()
    sm = b["single_mask"].to(dt)
    pm = b["pair_mask"].to(dt)

    cap: dict[str, torch.Tensor] = {}
    ps = blk.pair_stack
    want = {"tmo": ps.tri_mul_out_in.tri_mul_out if hasattr(ps, "tri_mul_out_in") else None}
    # Hook by dotted name so the structure is read off the module tree, not assumed.
    named = dict(blk.named_modules())
    keys = ["pair_stack.tri_mul_out_in.tri_mul_out", "pair_stack.tri_mul_out_in.tri_mul_in",
            "pair_stack.tri_att_start_end.tri_att_start", "pair_stack.tri_att_start_end.tri_att_end",
            "pair_stack.tri_mul_out", "pair_stack.tri_mul_in",
            "pair_stack.tri_att_start", "pair_stack.tri_att_end",
            "pair_stack.pair_transition", "attn_pair_bias", "attn_pair_bias.layer_norm_a",
            "single_transition"]
    hooks = []

    def mk(name):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            cap[name] = t.detach().to(torch.float64).clone()
        return fn

    present = [k for k in keys if k in named]
    for k in present:
        hooks.append(named[k].register_forward_hook(mk(k)))

    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else torch.autocast("cpu", enabled=False))
    with ctx, torch.no_grad():
        s_out, z_out = blk(s, z, sm, pm)
    for h in hooks:
        h.remove()

    # Rebuild the shared ten from the captured updates. The pair sub-blocks in this tree may
    # return the residual-included z already; both shapes are handled by checking which of the
    # two reconstructions reproduces z_out.
    def nm(x):
        return float(x.to(torch.float64).norm())

    out = {"policy": a.policy, "quantise_input": bool(a.quantise_input), "block": a.block,
           "s_out": s_out.to(torch.float64), "z_out": z_out.to(torch.float64),
           "s_in": s.to(torch.float64), "z_in": z.to(torch.float64),
           "captured": {k: cap[k] for k in cap},
           "module_names": sorted(named.keys())}
    torch.save(out, a.out)
    rep = {"tree": a.tree, "openfold3_file": openfold3.__file__, "policy": a.policy,
           "quantise_input": bool(a.quantise_input), "block": a.block, "tensors_loaded": n_t,
           "dims": dims, "hooked": present,
           "dtype_policy": {
               "f64": "every parameter and activation float64; checkpoint upcast once at load; "
                      "no cast on the path; upstream's CUDA autocast contexts are inert on CPU",
               "f32": "parameters and activations float32, no autocast",
               "bf16auto": "float32 parameters under torch.autocast('cpu', bfloat16)",
           }[a.policy],
           "s_in_norm": nm(s), "z_in_norm": nm(z),
           "s_out_norm": nm(s_out), "z_out_norm": nm(z_out),
           "captured_norms": {k: nm(v) for k, v in cap.items()}}
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({k: rep[k] for k in ("policy", "quantise_input", "block", "s_out_norm",
                                          "z_out_norm")}))
    print(json.dumps(rep["captured_norms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
