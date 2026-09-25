#!/usr/bin/env python3
"""Upstream's own pairformer trunk, at a chosen precision, over the boundary our device arm runs.

WHY THIS EXISTS, and why the pinned arm4 could not be used here. A24 says to score against
`pinned_p175/arm4_bf16_autocast/grads_f64.pt`, upstream's own bf16 autocast training step. That
file is a gradient of the 043 step: manifest loss 1.267624369070698, gradient global norm
3.206188185139011. Every pairformer BOUNDARY this campaign holds -- `of3t_gradients/cap`, which is
the only thing a device trunk arm can be driven by -- was captured from the 0.5.0 step, whose loss
is 1.6591175475821072 and whose gradient global norm is 3.908301894520238. Those are two different
losses, so our gradient and arm4's would not be gradients of the same function and scoring one
against the other would produce a number that looks like agreement and is not.

So the bf16 bar is computed HERE, at the same step and over the same captured boundary the device
arm uses, with upstream's own `PairFormerBlock` and upstream's own autocast recipe (float32
parameters, `torch.autocast(bfloat16)`), which is what `bundle_min.py --dtype float32 --autocast
bf16` runs at model scope. Same construction, restricted to the trunk.

The float64 arm of this same script is the control that makes the restriction legitimate: pair-
former block i's parameters appear once in their graph and at num_recycles 0 the trunk runs once,
so the float64 gradient computed here over the captured boundary must equal the bundle's own
whole-model float64 entries for `pairformer_stack.*`. That is checked over all 48 blocks rather
than asserted -- the capture's own self-check covers only blocks 0, 23 and 47.

Nothing here touches a device. Dropout is in eval, matching the capture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
PRE = "pairformer_stack.blocks."


def build(sd, first, n, dtype):
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
    mods, load = [], []
    for i in range(first, first + n):
        sub = {k[len(f"{PRE}{i}."):]: v.to(dtype) for k, v in sd.items()
               if k.startswith(f"{PRE}{i}.")}
        m = PairFormerBlock(**dims).to(dtype)
        missing, unexpected = m.load_state_dict(sub, strict=True)
        m.eval()                                   # the capture disabled dropout; so does this
        mods.append(m)
        load.append({"block": i, "tensors": len(sub),
                     "missing": list(missing), "unexpected": list(unexpected)})
    return mods, dims, load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default="/home/ttuser/of3t_gradients/cap")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--stack", type=int, default=48)
    ap.add_argument("--crop", type=int, default=0,
                    help="crop the captured boundary to the first N token positions. 56 of the "
                         "384 are real. Every arm scored against each other must use the same "
                         "crop, and the float64 arm at this crop against the bundle's own "
                         "whole-batch float64 entries is what says whether the crop changed the "
                         "function.")
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--autocast", default="none", choices=("none", "bf16"),
                    help="bf16 with --dtype float32 is upstream's own training recipe, the same "
                         "one `bundle_min.py --dtype float32 --autocast bf16` runs at model "
                         "scope to produce arm4.")
    ap.add_argument("--dump-grads", default="", dest="dump_grads", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    last = a.block + a.stack - 1
    dt = {"float64": torch.float64, "float32": torch.float32}[a.dtype]
    rep = {"what": __doc__.strip().splitlines()[0], "block": a.block, "stack": a.stack,
           "last_block": last, "dtype": a.dtype, "autocast": a.autocast, "crop": a.crop,
           "checkpoint": CKPT, "capture_dir": a.cap,
           "boundary": {"inputs_from_block": a.block, "cotangent_from_block": last}}

    cap = torch.load(Path(a.cap) / f"block{a.block}_boundary.pt", map_location="cpu",
                     weights_only=False)
    cap_out = cap if a.stack == 1 else torch.load(
        Path(a.cap) / f"block{last}_boundary.pt", map_location="cpu", weights_only=False)
    s_in, z_in = cap["args"][0], cap["args"][1]
    single_mask = cap["kwargs"]["single_mask"]
    pair_mask = cap["kwargs"]["pair_mask"]
    cot_s, cot_z = cap_out["cot"][0], cap_out["cot"][1]
    s_ref_out, z_ref_out = cap_out["out"][0], cap_out["out"][1]
    if a.crop:
        c = a.crop
        s_in, z_in = s_in[:, :c], z_in[:, :c, :c]
        s_ref_out, z_ref_out = s_ref_out[:, :c], z_ref_out[:, :c, :c]
        cot_s, cot_z = cot_s[:, :c], cot_z[:, :c, :c]
        single_mask, pair_mask = single_mask[:, :c], pair_mask[:, :c, :c]
    N = int(z_in.shape[1])
    rep["probe"] = {"tokens": N, "real_tokens": int(single_mask.sum()),
                    "s_norm": float(s_in.double().norm()), "z_norm": float(z_in.double().norm()),
                    "cot_s_norm": float(cot_s.double().norm()),
                    "cot_z_norm": float(cot_z.double().norm())}
    print(f"[{time.perf_counter()-t0:.0f}s] boundary N={N} real={int(single_mask.sum())}",
          flush=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    mods, dims, load = build(sd, a.block, a.stack, dt)
    rep["their_dims"], rep["their_load"] = dims, load
    n_par = sum(1 for m in mods for _ in m.named_parameters())
    rep["their_tensor_count"] = n_par
    print(f"[{time.perf_counter()-t0:.0f}s] built {a.stack} PairFormerBlock in {a.dtype}, "
          f"{n_par} tensors", flush=True)

    s = s_in.to(dt).contiguous()
    z = z_in.to(dt).contiguous()
    sm, pm = single_mask.to(dt), pair_mask.to(dt)
    cs, cz = cot_s.to(dt), cot_z.to(dt)
    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.autocast == "bf16"
           else torch.autocast("cpu", enabled=False))
    with ctx:
        for m in mods:
            s, z = m(s, z, sm, pm)
        loss = (s.to(dt) * cs).sum() + (z.to(dt) * cz).sum()
    print(f"[{time.perf_counter()-t0:.0f}s] forward done, loss {float(loss):.12e}", flush=True)
    rep["forward"] = {
        "loss": float(loss),
        "s_out_norm": float(s.double().norm()), "z_out_norm": float(z.double().norm()),
        "rel_vs_captured_s": float((s.double() - s_ref_out.double()).norm()
                                   / (s_ref_out.double().norm() + 1e-300)),
        "rel_vs_captured_z": float((z.double() - z_ref_out.double()).norm()
                                   / (z_ref_out.double().norm() + 1e-300))}
    loss.backward()
    print(f"[{time.perf_counter()-t0:.0f}s] backward done", flush=True)

    grads, absent = {}, []
    for j, m in enumerate(mods):
        for n, p in m.named_parameters():
            full = f"{PRE}{a.block + j}.{n}"
            if p.grad is None:
                absent.append(full)
            else:
                grads[full] = p.grad.detach().to(torch.float64).clone()
    rep["gradients"] = {"with_grad": len(grads), "without_grad": len(absent),
                        "absent": absent[:32]}
    d = os.path.dirname(a.dump_grads)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(grads, a.dump_grads)
    rep["dumped_to"] = a.dump_grads
    rep["seconds"] = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[{time.perf_counter()-t0:.0f}s] wrote {len(grads)} tensors to {a.dump_grads} "
          f"and the report to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
