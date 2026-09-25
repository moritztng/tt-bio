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

    dL/d(theta)  =  d/d(theta) [ <cot_s, s_out> + <cot_z_ext, z_out> ]

is exact at this boundary rather than a surrogate for the real loss -- PROVIDED the two
cotangents are the EXTERNAL partials. They are not as hooked. D242: the last block's
`attn_pair_bias` reads the `z` its own `pair_stack` just produced, so `z_out` is an ancestor of
`s_out`, `(s_out, z_out)` is not a graph cut, and the hooked `cot_z` is a TOTAL derivative that
already contains the route through `s_out`. Injecting both replays that route twice. The default
here subtracts it,

    cot_z_ext  =  cot_z  -  d<cot_s, s_out>/d(z_out)

which takes block 47 from 0.7849281738435908 against the reference's own full-model backward to
3.0392623414001263e-15. `--legacy-total-cotangent` restores the old, double-counting injection
for reproducing banked artifacts; every artifact stamps which of the two produced it.

VALIDATION. The f64 arm is checked against central finite differences along one random unit
direction through the whole 2,736-tensor parameter set (PROTOCOL SS3c). A reference that
differentiates a plausible neighbour of the function agrees with a wrong gradient.

PAD INVARIANCE. `--pad-scale K` multiplies the boundary's PAD rows (and, in z, the pad rows and
columns) by K and changes nothing else. Under correct masking a real output cannot depend on a
pad position and the captured output cotangent is exactly zero on the pads, so every parameter
gradient must be invariant to K. It is a control on both sides: a reference that moves means the
pads genuinely participate, and an arm that moves while the reference does not has a masking
leak feeding its weight gradients.

usage: ref_grad.py --tree <dir containing openfold3/> --boundary B.pt --cap-last C.pt --out O.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
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


def ancestor_pairs(outs):
    """Which of the injected outputs are reachable from which, by a walk of the grad_fn DAG.

    One cotangent per output is only a valid injection when the outputs form a graph CUT. If
    output B is reachable from output A then A's hooked cotangent is a total derivative that
    already carries the A -> B route, and injecting both counts that route twice (D242,
    `perf/of3t_orchestrator/doublecount/BLAST_RADIUS.json`). Returns the (ancestor, descendant)
    pairs; empty means the outputs are a cut and the two conventions coincide.
    """
    named, alive = {}, [t.grad_fn for _, t in outs]
    for name, t in outs:
        if t.grad_fn is None:
            continue
        if id(t.grad_fn) in named:
            raise SystemExit(
                f"{name} and {named[id(t.grad_fn)]} are outputs of one autograd node, so this "
                "walk cannot see whether one feeds the other. The block that produces the "
                "injected outputs has to run eagerly.")
        named[id(t.grad_fn)] = name
    pairs = []
    for name, t in outs:
        if t.grad_fn is None:
            continue
        seen, stack = set(), [t.grad_fn]
        while stack:
            fn = stack.pop()
            if id(fn) in seen:
                continue
            seen.add(id(fn))
            alive.append(fn)
            other = named.get(id(fn))
            if other is not None and other != name:
                pairs.append((other, name))
            for nxt, _ in fn.next_functions:
                if nxt is not None:
                    stack.append(nxt)
    return sorted(set(pairs))


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
    ap.add_argument("--pad-scale", type=float, default=1.0, metavar="K",
                    help="multiply the boundary's pad rows/columns by K, nothing else")
    ap.add_argument("--checkpoint", action="store_true",
                    help="recompute each block's activations in the backward instead of keeping "
                         "them live (torch.utils.checkpoint, use_reentrant=False). Needed at "
                         "crop 384: the plain float64 arm's saved activations do not fit in any "
                         "host we have. Recompute of an eval()-mode block with no dropout is "
                         "deterministic, so this must be BIT-IDENTICAL to the plain arm, and "
                         "that identity is checked at crop 64 rather than assumed.")
    ap.add_argument("--legacy-total-cotangent", action="store_true",
                    help="D242: inject the captured cot_z as hooked. It is a TOTAL derivative "
                         "and z_out is an ancestor of s_out, so the s_out <- z_out route is "
                         "counted twice. Kept only to reproduce banked artifacts.")
    ap.add_argument("--cot-correction", default=None, metavar="PT",
                    help="take the cot_z correction from a file (a .pt holding "
                         "{'cot_z_correction': T} or a bare tensor) instead of this arm's own "
                         "graph. The external cotangent is a property of the reference's "
                         "downstream, so a narrower-precision arm is driven by the f64 arm's.")
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
    pad_rep = None
    if a.pad_scale != 1.0:
        pad = (sm.reshape(-1) <= 0)
        s0 = s0.clone(); z0 = z0.clone()
        s0[:, pad] *= a.pad_scale
        z0[:, pad, :] *= a.pad_scale
        z0[:, :, pad] *= a.pad_scale
        pad_rep = {"pad_scale": a.pad_scale, "pad_rows": int(pad.sum()),
                   "real_rows": int((~pad).sum()),
                   "s_in_norm_after": float(s0.norm()), "z_in_norm_after": float(z0.norm())}

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
    if a.checkpoint:
        from torch.utils.checkpoint import checkpoint

        def fwd(s, z):
            # The LAST block runs eagerly. The injected outputs are its outputs and the D242
            # correction is a derivative of one against the other, so its graph has to be real.
            # Recompute of an eval()-mode block is deterministic, so this is bit-neutral, and
            # the c64 plain-vs-checkpointed control re-checks that rather than assuming it.
            for m in mods[:-1]:
                s, z = checkpoint(m, s, z, sm, pm, use_reentrant=False)
            return mods[-1](s, z, sm, pm)
    else:
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

    # A41. The graph walk runs on every arm, both conventions, and is recorded. It is what makes
    # D242 unfileable a second time: a boundary that stops being a cut says so here.
    pairs = ancestor_pairs([("s_out", s_out), ("z_out", z_out)])
    if pairs and pairs != [("z_out", "s_out")]:
        raise SystemExit(f"unexpected ancestry among the injected outputs: {pairs}")
    cot_z_hooked, corr = cot_z, None
    if a.legacy_total_cotangent:
        convention = "legacy-total-cotangent"
    else:
        convention = "graph-cut-external"
        if a.cot_correction:
            corr = torch.load(a.cot_correction, map_location="cpu", weights_only=False)
            corr = corr["cot_z_correction"] if isinstance(corr, dict) else corr
            corr = corr.to(torch.float64)
            if tuple(corr.shape) != tuple(cot_z.shape):
                raise SystemExit(f"correction {tuple(corr.shape)} does not fit cot_z "
                                 f"{tuple(cot_z.shape)}")
        elif pairs:
            with ctx:
                corr = torch.autograd.grad(outputs=s_out.to(torch.float64), grad_outputs=cot_s,
                                           inputs=z_out, retain_graph=True)[0]
            corr = corr.detach().to(torch.float64)
        if corr is not None:
            cot_z = cot_z_hooked - corr

    with ctx:
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
                "pad_scale": a.pad_scale, "injection_convention": convention,
                "cot_z_correction": corr,
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
           "permuted_cotangent": perm_rep, "pad_perturbation": pad_rep,
           "injection": {
               "convention": convention,
               "what": ("<cot_s, s_out> + <cot_z - d<cot_s,s_out>/dz_out, z_out>, the graph-cut "
                        "external cotangent (D242 repaired)") if convention ==
                       "graph-cut-external" else
                       ("<cot_s, s_out> + <cot_z, z_out> with cot_z as hooked. z_out is an "
                        "ancestor of s_out, so this DOUBLE COUNTS that route (D242)"),
               "graph_cut": {"is_cut": not pairs,
                             "ancestor_descendant_pairs": [list(x) for x in pairs],
                             "checked_by": "walk of the grad_fn DAG over the injected outputs"},
               "correction_source": (None if corr is None else
                                     (a.cot_correction or "this arm's own graph")),
               "cot_z_norm_hooked": float(cot_z_hooked.norm()),
               "cot_z_norm_injected": float(cot_z.norm()),
               "correction_norm": (None if corr is None else float(corr.norm())),
               "duplicate_share_of_the_hooked_cot_z": (
                   None if corr is None else
                   float(corr.norm()) / float(cot_z_hooked.norm())),
           },
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
           "checkpointed": bool(a.checkpoint),
           "threads": a.threads,
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
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
    rep["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(json.dumps({"policy": a.policy, "injection": convention, "loss": rep["loss"], "s_norm": rep["s_norm"],
                      "z_norm": rep["z_norm"],
                      "grad_sq_norm": rep["gradient"]["squared_norm_total"],
                      "with_gradient": rep["gradient"]["with_gradient"],
                      "fd_rel": rep.get("finite_difference", {}).get("rel"),
                      "peak_rss_gb": round(rep["peak_rss_gb"], 2),
                      "seconds": round(time.perf_counter() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
