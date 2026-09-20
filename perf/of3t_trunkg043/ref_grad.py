#!/usr/bin/env python3
"""Upstream's own 48-block pairformer stack, differentiated over the captured trunk boundary.

One tree per process, `of3t-trunk043ref/ref_stack.py`'s rule: the two release trees define the
same module names and cannot share an interpreter. `build()` below is that row's `build()`
unchanged, so this differentiates the same function whose forward it published.

DTYPE POLICY, written out because "float64" names a width and not a policy (PROTOCOL A27):

  f64        every parameter and every activation float64, the checkpoint upcast once at load,
             no cast anywhere on the path. This is the reference every rel_l2 is against.
  bf16auto   float32 parameters under `torch.autocast('cpu', bfloat16)` -- upstream's OWN
             training recipe, and the only honest denominator for "how close can a bf16 port
             get". Its own gradient is compared against the f64 arm to give the floor.
  f32        float32 parameters, no autocast. The instrument floor: whatever this reads is what
             a perfect port of upstream's own arithmetic in a narrower width costs.

THE COTANGENT is the one the capture recorded at block 47's output, not a draw. The stack's
parameters appear once in their graph and at num_recycles 0 the trunk runs once, so

    dL/d(theta)  =  d/d(theta) [ <cot_s, s_out> + <cot_z, z_out> ]

is exact at this boundary rather than a surrogate for the real loss.

VALIDATION. The f64 arm is checked against central finite differences along one random unit
direction through the whole 2,736-tensor parameter set (PROTOCOL SS3c). A reference that
differentiates a plausible neighbour of the function agrees with a wrong gradient.

usage: ref_grad.py --tree <dir containing openfold3/> --boundary B.pt --cap-last C.pt --out O.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def build(sd, n, dtype):
    """`of3t-trunk043ref/ref_stack.py:build`, unchanged. Upstream's own PairFormerBlock,
    dimensions read off the checkpoint, strict load."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--cap-last", required=True,
                    help="the capture the OUTPUT cotangent comes from; block 47's boundary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--policy", default="f64", choices=("f64", "f32", "bf16auto"))
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fd", action="store_true",
                    help="validate against central finite differences along one random unit "
                         "direction through the whole parameter set (SS3c)")
    ap.add_argument("--fd-eps", type=float, default=1e-5)
    ap.add_argument("--permute-cot", type=int, default=0, metavar="SEED",
                    help="BREAK control: permute the cotangent over the REAL token positions "
                         "only. Everything else -- weights, masks, boundary -- is untouched.")
    a = ap.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, a.tree)
    import openfold3
    if not openfold3.__file__.startswith(a.tree):
        raise SystemExit(f"wrong tree on sys.path: {openfold3.__file__} is not under {a.tree}")

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    dt = torch.float32 if a.policy in ("f32", "bf16auto") else torch.float64

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    mods, dims, load = build(sd, a.blocks, dt)
    if load["missing"] or load["unexpected"]:
        raise SystemExit(f"strict load did not hold: {load['missing'][:4]} "
                         f"{load['unexpected'][:4]}")

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    s0 = b["s_in"].to(dt).contiguous()
    z0 = b["z_in"].to(dt).contiguous()
    sm = b["single_mask"].to(dt)
    pm = b["pair_mask"].to(dt)

    cap = torch.load(a.cap_last, map_location="cpu", weights_only=False)
    cot_s, cot_z = cap["cot"][0], cap["cot"][1]
    if cot_s is None or cot_z is None:
        raise SystemExit("captured cotangent missing -- the boundary is unusable")
    c = a.crop
    cot_s = cot_s.to(torch.float64)[:, :c].contiguous() if c else cot_s.to(torch.float64)
    cot_z = (cot_z.to(torch.float64)[:, :c, :c].contiguous() if c
             else cot_z.to(torch.float64))
    perm_rep = None
    if a.permute_cot:
        real = torch.nonzero(sm.reshape(-1) > 0).reshape(-1)
        g = torch.Generator().manual_seed(a.permute_cot)
        order = real[torch.randperm(int(real.numel()), generator=g)]
        idx = torch.arange(int(sm.shape[-1]))
        idx[real] = order
        cot_s = cot_s[:, idx].contiguous()
        cot_z = cot_z[:, idx][:, :, idx].contiguous()
        perm_rep = {"seed": a.permute_cot, "real_positions_permuted": int(real.numel()),
                    "fixed_points": int((order == real).sum()),
                    "what": "the same cotangent, paired with the wrong token positions"}
    cs, cz = cot_s.to(dt), cot_z.to(dt)

    def fwd(s, z):
        for m in mods:
            s, z = m(s, z, sm, pm)
        return s, z

    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else torch.autocast("cpu", enabled=False))

    for m in mods:
        for p in m.parameters():
            p.grad = None
    t1 = time.perf_counter()
    s_in = s0.detach().requires_grad_(True)
    z_in = z0.detach().requires_grad_(True)
    with ctx:
        s_out, z_out = fwd(s_in, z_in)
        loss = (s_out.to(torch.float64) * cot_s).sum() + (z_out.to(torch.float64) * cot_z).sum()
    loss.backward()
    fwd_back_s = time.perf_counter() - t1

    grads, n_none = {}, 0
    for i, m in enumerate(mods):
        for n, p in m.named_parameters():
            k = f"{PRE}{i}.{n}"
            if p.grad is None:
                grads[k] = None
                n_none += 1
            else:
                grads[k] = p.grad.detach().to(torch.float64).clone()
    s_det = s_out.detach().to(torch.float64)
    z_det = z_out.detach().to(torch.float64)
    torch.save({"grads": grads, "s": s_det, "z": z_det,
                "ds_in": (s_in.grad.detach().to(torch.float64)
                          if s_in.grad is not None else None),
                "dz_in": (z_in.grad.detach().to(torch.float64)
                          if z_in.grad is not None else None),
                "policy": a.policy, "tree": a.tree, "permute_cot": a.permute_cot,
                "loss": float(loss)}, a.out)

    sq = {k: float(torch.linalg.vector_norm(v)) ** 2 for k, v in grads.items() if v is not None}
    rep = {"what": __doc__.strip().splitlines()[0], "tree": a.tree,
           "openfold3_file": openfold3.__file__, "policy": a.policy, "blocks": a.blocks,
           "crop": a.crop, "dims": dims, "load": load,
           "dtype_policy": {
               "f64": "every parameter and every activation float64; checkpoint upcast once at "
                      "load; no cast on the path",
               "f32": "parameters and activations float32, no autocast",
               "bf16auto": "float32 parameters under torch.autocast('cpu', bfloat16), which is "
                           "upstream's own training recipe",
           }[a.policy],
           "boundary": a.boundary, "boundary_sha256": sha256_file(a.boundary),
           "cotangent_from": a.cap_last, "cotangent_sha256": sha256_file(a.cap_last),
           "permuted_cotangent": perm_rep,
           "probe": {"s_in_norm": float(s0.norm()), "z_in_norm": float(z0.norm()),
                     "cot_s_norm": float(cot_s.norm()), "cot_z_norm": float(cot_z.norm()),
                     "real_tokens": int(sm.sum()), "tokens": int(sm.shape[-1])},
           "loss": float(loss),
           "s_norm": float(s_det.norm()), "z_norm": float(z_det.norm()),
           "ds_in_norm": (float(s_in.grad.norm()) if s_in.grad is not None else None),
           "dz_in_norm": (float(z_in.grad.norm()) if z_in.grad is not None else None),
           "gradient": {"tensors": len(grads), "with_gradient": len(sq), "none": n_none,
                        "squared_norm_total": sum(sq.values())},
           "seconds_forward_backward": fwd_back_s,
           "out": a.out}

    if a.fd:
        # SS3c. One random unit direction through the whole parameter set, central differences
        # in the arm's own dtype. Per-tensor differences over 2,736 tensors would be 5,472
        # forwards; the joint direction touches every one of them in two.
        params = [(f"{PRE}{i}.{n}", p) for i, m in enumerate(mods)
                  for n, p in m.named_parameters()]
        gen = torch.Generator().manual_seed(5)
        dirs = {n: torch.randn(p.shape, generator=gen, dtype=dt) for n, p in params}
        nrm = torch.sqrt(sum((d.to(torch.float64) ** 2).sum() for d in dirs.values()))
        for d in dirs.values():
            d /= nrm.to(dt)
        base = {n: p.detach().clone() for n, p in params}

        def loss_at(sign):
            with torch.no_grad():
                for n, p in params:
                    p.copy_(base[n] + sign * a.fd_eps * dirs[n])
                with ctx:
                    s, z = fwd(s0, z0)
                    v = float((s.to(torch.float64) * cot_s).sum()
                              + (z.to(torch.float64) * cot_z).sum())
                for n, p in params:
                    p.copy_(base[n])
            return v

        fp, fm = loss_at(+1.0), loss_at(-1.0)
        fd = (fp - fm) / (2 * a.fd_eps)
        an = float(sum((grads[n].to(torch.float64) * dirs[n].to(torch.float64)).sum()
                       for n, _ in params if grads[n] is not None))
        rep["finite_difference"] = {
            "analytic": an, "finite_difference": fd,
            "rel": abs(fd - an) / (abs(an) + 1e-300), "eps": a.fd_eps,
            "direction": "one random unit direction through all "
                         f"{len(params)} parameter tensors",
            "bar": 1e-6}
        print(f"[{time.perf_counter()-t0:.0f}s] joint FD rel "
              f"{rep['finite_difference']['rel']:.3e}", flush=True)

    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({"policy": a.policy, "loss": rep["loss"], "s_norm": rep["s_norm"],
                      "z_norm": rep["z_norm"],
                      "grad_sq_norm": rep["gradient"]["squared_norm_total"],
                      "with_gradient": rep["gradient"]["with_gradient"],
                      "fd_rel": rep.get("finite_difference", {}).get("rel"),
                      "seconds": round(time.perf_counter() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
