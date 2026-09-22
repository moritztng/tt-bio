#!/usr/bin/env python3
"""One upstream release tree's 48-block pairformer trunk GRADIENT over the captured boundary.

`of3t_trunk043ref/ref_stack.py` runs the forward of one tree at one dtype policy;
`of3t_pairformer/upstream_arm.py` takes the gradient but only from whichever `openfold3` the
interpreter happens to find, which is how a 0.5.0 reference ended up under a checkpoint bound to
0.4.3. This is the two of them joined: the tree is an argument and it is asserted after import,
and the same process writes the parameter gradients AND the forward outputs the A18 check needs.

DTYPE POLICY, written out because "float64" names a width and not a policy (A27):

  f64        every parameter and every activation float64. The checkpoint is upcast once at load
             and nothing casts on the path. Upstream's `_attention` wraps its scores in
             `torch.amp.autocast("cuda", ...)`, inert on CPU, so `use_high_precision_attention`
             cannot fire in this arm.
  f32        parameters and activations float32, no autocast.
  bf16auto   float32 parameters under `torch.autocast("cpu", bfloat16)` -- upstream's own
             training recipe, and the only honest floor for a bf16 port.

THE BOUNDARY. `cap/block0_boundary.pt` in, `cap/block{last}_boundary.pt`'s cotangent back. That
is exact rather than approximate: pairformer block i's parameters appear once in their graph and
at num_recycles 0 the trunk runs once, so

    dL/d(theta_i) = d/d(theta_i) [ <cot_s, s_out> + <cot_z, z_out> ]

with the inputs, masks and cotangents all from one run of their own step. `--boundary-check`
asserts the sliced tensors are bit-identical to `of3t_trunk043ref/boundary_c64.pt`, so the crop
is an artifact with a sha256 and not a convention each script re-derives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
PRE = "pairformer_stack.blocks."


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build(sd, n, dtype, first=0):
    """Upstream's own PairFormerBlock, dimensions read off the checkpoint, strict load.

    `first` is the checkpoint block index the window starts at, so one block in the middle of
    the trunk can be built alone. The window is always contiguous and always keyed by the
    CHECKPOINT index, never by its position in the window.
    """
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
        m.eval()                                # the capture disabled dropout; so does this
        mods.append(m)
        load.append({"block": i, "tensors": len(sub),
                     "missing": list(missing), "unexpected": list(unexpected)})
    return mods, dims, load


def force_tb(value: bool):
    """Pin the ending-node bias orientation whatever the tree's own call site passes."""
    import openfold3.core.model.layers.triangular_attention as ta
    orig = ta.TriangleAttention.forward

    def patched(self, *a, **kw):
        kw["transpose_bias"] = value
        return orig(self, *a, **kw)

    ta.TriangleAttention.forward = patched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="directory containing the openfold3/ package")
    ap.add_argument("--cap", default="/home/ttuser/of3t_gradients/cap")
    ap.add_argument("--crop", type=int, default=64, help="0 means the whole 384-token batch")
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--first", type=int, default=0, metavar="K",
                    help="checkpoint block index the window starts at. The boundary follows it: "
                         "cap/block{K}_boundary.pt in, cap/block{K+blocks-1}_boundary.pt's "
                         "cotangent back. `--first 23 --blocks 1` is block 23's backward ALONE, "
                         "with no chaining at all, which is what separates a wrong leaf-op "
                         "backward from an error the chain accumulates over 48 of them.")
    ap.add_argument("--zero-pad", action="store_true", help="D28, quantitatively. Zero the PAD token positions of the captured input on BOTH sides instead of poisoning them. At this crop-64 boundary 99.76 %% of the pair input's squared mass sits on padded positions at block 0, so a LayerNorm weight gradient -- which sums over every position -- is formed almost entirely out of pad. The NaN discriminator cannot separate the sides because both propagate it; this can, because it changes the function identically on both sides and asks whether the DISAGREEMENT survives.")
    ap.add_argument("--nan-pad", action="store_true",
                    help="D28 discriminator, reference side. Poison the PAD token positions of "
                         "the captured input with NaN and count how many parameter gradients "
                         "come back NaN. A LayerNorm WEIGHT gradient sums over every position "
                         "including the padded ones, so a mask-clean forward does not imply a "
                         "mask-clean backward; this says whether upstream's own backward is "
                         "mask-clean, which is what our side has to be compared against.")
    ap.add_argument("--input-grad-out", default="",
                    help="write dL/ds_in and dL/dz_in to this .pt. Over a single block both "
                         "sides are driven by the SAME captured cotangent, so this is the error "
                         "that block's backward hands to the block before it -- the per-block "
                         "chain error, which is the quantity a 48-deep stack compounds.")
    ap.add_argument("--policy", default="f64", choices=("f64", "f32", "bf16auto", "bf16pure"))
    ap.add_argument("--tb", default="keep", choices=("keep", "off", "on"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--boundary-check", default="",
                    help="assert the sliced boundary is bit-identical to this extract_boundary "
                         "artifact, and record its sha256")
    ap.add_argument("--out", required=True, help="parameter gradients, keyed by checkpoint name")
    ap.add_argument("--forward-out", default="", help="s/z forward outputs, for the A18 check")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, a.tree)
    import openfold3
    if not openfold3.__file__.startswith(a.tree):
        raise SystemExit(f"wrong tree on sys.path: {openfold3.__file__} is not under {a.tree}")
    if a.tb != "keep":
        force_tb(a.tb == "on")
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    last = a.first + a.blocks - 1
    src0 = os.path.join(a.cap, f"block{a.first}_boundary.pt")
    srcL = os.path.join(a.cap, f"block{last}_boundary.pt")
    cap0 = torch.load(src0, map_location="cpu", weights_only=False)
    capL = torch.load(srcL, map_location="cpu", weights_only=False)
    c = a.crop
    cs = (lambda x: x[:, :c].contiguous()) if c else (lambda x: x.contiguous())
    cz = (lambda x: x[:, :c, :c].contiguous()) if c else (lambda x: x.contiguous())
    f64 = lambda x: x.detach().to(torch.float64)

    s_in = f64(cs(cap0["args"][0]))
    z_in = f64(cz(cap0["args"][1]))
    single_mask = f64(cs(cap0["kwargs"]["single_mask"]))
    pair_mask = f64(cz(cap0["kwargs"]["pair_mask"]))
    s_cap = f64(cs(capL["out"][0]))
    z_cap = f64(cz(capL["out"][1]))
    cot_s = f64(cs(capL["cot"][0]))
    cot_z = f64(cz(capL["cot"][1]))

    N = int(z_in.shape[1])
    real = int(single_mask.sum())
    zero_pad_rep = None
    if a.zero_pad:
        pad = (single_mask.reshape(-1) <= 0)
        s_in = s_in.clone(); z_in = z_in.clone()
        s_in[:, pad] = 0.0
        z_in[:, pad, :] = 0.0
        z_in[:, :, pad] = 0.0
        zero_pad_rep = {"pad_rows": int(pad.sum()), "real_rows": real,
                        "s_in_norm": float(s_in.norm()), "z_in_norm": float(z_in.norm())}
        print(f"[{time.perf_counter()-t0:.0f}s] zero-pad: {int(pad.sum())} pad rows cleared, "
              f"|s_in| {float(s_in.norm()):.6e} |z_in| {float(z_in.norm()):.6e}", flush=True)
    if a.nan_pad:
        pad = (single_mask.reshape(-1) <= 0)
        if not bool(pad.any()):
            raise SystemExit("--nan-pad: this boundary has no pad rows, nothing to poison")
        s_in = s_in.clone(); z_in = z_in.clone()
        s_in[:, pad] = float("nan")
        z_in[:, pad, :] = float("nan")
        z_in[:, :, pad] = float("nan")
        print(f"[{time.perf_counter()-t0:.0f}s] D28: poisoned {int(pad.sum())} pad rows of "
              f"{int(pad.numel())} with NaN", flush=True)
    rep = {"what": __doc__.strip().splitlines()[0], "tree": a.tree,
           "openfold3_file": openfold3.__file__, "policy": a.policy, "tb": a.tb,
           "blocks": a.blocks, "crop": c, "checkpoint": CKPT,
           "first_block": a.first,
           "boundary": {"inputs_from_block": a.first, "cotangent_from_block": last,
                        "block_first": src0, "block_last": srcL,
                        "tokens": N, "real_tokens": real,
                        "padding_fraction_single": 1.0 - real / N,
                        "padding_fraction_pair": 1.0 - float(pair_mask.sum()) / (N * N),
                        "s_in_norm": float(s_in.norm()), "z_in_norm": float(z_in.norm()),
                        "cot_s_norm": float(cot_s.norm()), "cot_z_norm": float(cot_z.norm())},
           "dtype_policy": {
               "f64": "every parameter and every activation float64; checkpoint upcast once at "
                      "load; no cast on the path; upstream's CUDA-autocast contexts are inert on "
                      "CPU so use_high_precision_attention cannot fire",
               "f32": "parameters and activations float32, no autocast",
               "bf16auto": "float32 parameters under torch.autocast('cpu', bfloat16) -- "
                           "upstream's own training recipe",
               "bf16pure": "every parameter and every activation bfloat16, NO autocast. "
                           "autocast keeps layer norms, softmaxes and reductions in float32 "
                           "off its own op list; this arm takes that away and is therefore "
                           "the upstream-side control for a stack whose arithmetic really is "
                           "bf16 all the way down. The difference between this and bf16auto "
                           "is exactly what autocast's float32 list is worth on this scope.",
           }[a.policy]}

    rep["zero_pad"] = zero_pad_rep
    if a.boundary_check and (a.first, a.blocks) != (0, 48):
        raise SystemExit("--boundary-check pins the 0..47 crop-64 boundary; it does not "
                         "describe a single-block window")
    if a.boundary_check:
        b = torch.load(a.boundary_check, map_location="cpu", weights_only=False)
        ident = {k: bool(torch.equal(v, b[k])) for k, v in
                 (("s_in", s_in), ("z_in", z_in), ("single_mask", single_mask),
                  ("pair_mask", pair_mask))}
        ident["s_ref_050_captured"] = bool(torch.equal(s_cap, b["s_ref_050_captured"]))
        ident["z_ref_050_captured"] = bool(torch.equal(z_cap, b["z_ref_050_captured"]))
        rep["boundary_identity"] = {"file": a.boundary_check,
                                    "sha256": sha256_file(a.boundary_check),
                                    "bit_identical": ident,
                                    "all": all(ident.values())}
        if not all(ident.values()):
            raise SystemExit(f"boundary slicing diverged from {a.boundary_check}: {ident}")
        print(f"[{time.perf_counter()-t0:.0f}s] boundary bit-identical to {a.boundary_check}",
              flush=True)

    dt = {"f32": torch.float32, "bf16auto": torch.float32,
          "bf16pure": torch.bfloat16}.get(a.policy, torch.float64)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    mods, dims, load = build(sd, a.blocks, dt, first=a.first)
    rep["their_dims"] = dims
    rep["their_load"] = {"blocks": len(load),
                         "tensors": sum(x["tensors"] for x in load),
                         "missing": sum(len(x["missing"]) for x in load),
                         "unexpected": sum(len(x["unexpected"]) for x in load)}
    n_par = sum(1 for m in mods for _ in m.named_parameters())
    print(f"[{time.perf_counter()-t0:.0f}s] built {a.blocks} PairFormerBlock "
          f"[{a.first}..{last}] in {a.policy}, "
          f"{n_par} tensors, N={N} real={real}", flush=True)

    s = s_in.to(dt).contiguous()
    z = z_in.to(dt).contiguous()
    if a.input_grad_out:
        s = s.detach().requires_grad_(True)
        z = z.detach().requires_grad_(True)
    s_leaf, z_leaf = s, z
    sm, pm = single_mask.to(dt), pair_mask.to(dt)
    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else torch.autocast("cpu", enabled=False))
    t1 = time.perf_counter()
    with ctx:
        for m in mods:
            s, z = m(s, z, sm, pm)
        loss = (s.to(torch.float64) * cot_s).sum() + (z.to(torch.float64) * cot_z).sum()
    s_out = s.detach().to(torch.float64)
    z_out = z.detach().to(torch.float64)
    msk = single_mask.reshape(1, N, 1)
    pmk = pair_mask.reshape(1, N, N, 1)
    rel = lambda x, y: float((x - y).norm() / (y.norm() + 1e-300))
    rep["forward"] = {
        "loss": float(loss.detach()),
        "s_out_norm": float(s_out.norm()), "z_out_norm": float(z_out.norm()),
        "seconds": time.perf_counter() - t1,
        "vs_captured_050": {"s": rel(s_out, s_cap), "z": rel(z_out, z_cap),
                            "s_masked": rel(s_out * msk, s_cap * msk),
                            "z_masked": rel(z_out * pmk, z_cap * pmk)}}
    print(f"[{time.perf_counter()-t0:.0f}s] forward done, loss {float(loss.detach()):.12e}, "
          f"vs captured 0.5.0 masked s {rep['forward']['vs_captured_050']['s_masked']:.3e} "
          f"z {rep['forward']['vs_captured_050']['z_masked']:.3e}", flush=True)

    loss.backward()
    grads, absent = {}, []
    for j, m in enumerate(mods, start=a.first):
        for n, p in m.named_parameters():
            full = f"{PRE}{j}.{n}"
            if p.grad is None:
                absent.append(full)
            else:
                grads[full] = p.grad.detach().to(torch.float64).clone()
    if a.nan_pad:
        nan_t = sorted(k for k, v in grads.items() if bool(torch.isnan(v).any()))
        rep["nan_pad"] = {"pad_rows": int((single_mask.reshape(-1) <= 0).sum()),
                          "real_rows": real, "tensors": len(grads), "nan": len(nan_t),
                          "nan_tensors": nan_t[:12],
                          "forward_out_nan": bool(torch.isnan(s_out).any()
                                                  or torch.isnan(z_out).any())}
        print(f"[{time.perf_counter()-t0:.0f}s] D28 reference: {len(nan_t)} of {len(grads)} "
              f"parameter gradients NaN; forward output NaN "
              f"{rep['nan_pad']['forward_out_nan']}", flush=True)
    gsq = sum(float((v ** 2).sum()) for v in grads.values())
    rep["gradient"] = {"with_grad": len(grads), "without_grad": len(absent),
                       "absent": absent[:32],
                       "global_norm": gsq ** 0.5, "squared_norm": gsq}
    print(f"[{time.perf_counter()-t0:.0f}s] backward done, {len(grads)} tensors, "
          f"global norm {gsq ** 0.5:.12e}", flush=True)

    if a.input_grad_out:
        ig = {"ds_in": (s_leaf.grad.detach().to(torch.float64).clone()
                        if s_leaf.grad is not None else None),
              "dz_in": (z_leaf.grad.detach().to(torch.float64).clone()
                        if z_leaf.grad is not None else None),
              "first_block": a.first, "blocks": a.blocks, "crop": c, "policy": a.policy}
        os.makedirs(os.path.dirname(a.input_grad_out) or ".", exist_ok=True)
        torch.save(ig, a.input_grad_out)
        rep["input_grad"] = {
            "file": a.input_grad_out,
            "ds_in_norm": None if ig["ds_in"] is None else float(ig["ds_in"].norm()),
            "dz_in_norm": None if ig["dz_in"] is None else float(ig["dz_in"].norm())}
        print(f"[{time.perf_counter()-t0:.0f}s] input gradients |ds_in| "
              f"{rep['input_grad']['ds_in_norm']:.12e} |dz_in| "
              f"{rep['input_grad']['dz_in_norm']:.12e}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(grads, a.out)
    rep["out"] = a.out
    rep["out_sha256"] = sha256_file(a.out)
    if a.forward_out:
        torch.save({"s": s_out, "z": z_out, "single_mask": single_mask, "pair_mask": pair_mask,
                    "policy": a.policy, "tree": a.tree, "tb": a.tb, "crop": c}, a.forward_out)
        rep["forward_out"] = a.forward_out
        rep["forward_out_sha256"] = sha256_file(a.forward_out)
    rep["seconds"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"[{time.perf_counter()-t0:.0f}s] wrote {a.out} and {a.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
