#!/usr/bin/env python3
"""Upstream 0.5.0's OWN bf16 gradient for the diffusion module, at the boundary our arm is scored at.

`of3t-lnaffine` established the floor discipline on the trunk: a leaf's share of the error mass
ranks where the error IS, not whether it is ours, so a concentration is not a signature until it is
differenced against what upstream's own bf16 recipe puts on the same leaf. No artifact in the
campaign carries an upstream-bf16 arm at DIFFUSION scope, so this builds it.

A27, and the brief's version string is wrong, so say it plainly. The float64 reference our
diffusion arm is scored against is `S["grad_f64"]` in /home/ttuser/of3t_diffusion_cap/sub_boundary.pt,
produced by `perf/of3t_diffusion/sub_boundary.py` from `/home/ttuser/of3t_gradients/of3pkg`, which is
**openfold3 0.5.0** (its dist-info says so). The 0.4.3 package the trunk floor used is a different
tree and a different boundary. The only floor that can be differenced against our reading is the one
built by the SAME package at the SAME boundary against the SAME float64 reference, which is this.

Policies, exactly as `perf/of3t_trunkg043/ref_grad.py` defines them:
  bf16auto  float32 parameters under `torch.autocast("cpu", bfloat16)` -- upstream's own training
            recipe, and the denominator A26's sqrt(2) applies to.
  f32       float32 parameters, no outer autocast. The instrument floor: whatever this reads is
            what the harness cannot tell apart, and it also proves the module is deterministic
            given these kwargs, without which nothing below is a measurement.

The bf16 arm publishes DTYPE_PROBE, the dtype a Linear inside the first DiT block actually emits.
A probe that does not read bfloat16 in the bf16 arm is a hard failure: it would mean the recipe
never reached the kernels and the "floor" would be an f32 arm wearing a bf16 label.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")

CAP = Path("/home/ttuser/of3t_diffusion_cap")
CKPT = Path("/home/ttuser/of3-weights/of3-p2-155k.pt")


def cast(x, dt):
    if torch.is_tensor(x):
        return x.to(dt) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: cast(v, dt) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cast(v, dt) for v in x)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=("bf16auto", "f32"))
    ap.add_argument("--cap", type=Path, default=CAP)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()

    import bundle_min as BM

    B = torch.load(a.cap / "diffusion_boundary.pt", map_location="cpu", weights_only=False)
    S = torch.load(a.cap / "sub_boundary.pt", map_location="cpu", weights_only=False)
    kwargs, cot, ref_grad = B["kwargs"], B["cot"], S["grad_f64"]
    print(f"[{time.time()-t0:.0f}s] boundary loaded, cot norm {float(cot.double().norm()):.6e}, "
          f"{len(ref_grad)} reference tensors", flush=True)
    del B, S

    dt = torch.float32
    built = BM.build(dt, 20260919, "cpu", num_recycles=0)
    model = built[1]
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    dm = model.diffusion_module
    del ck, sd
    print(f"[{time.time()-t0:.0f}s] model built at {dt}", flush=True)

    probe = {}

    def dtype_hook(mod, inp, out):
        if torch.is_tensor(out):
            probe.setdefault("dit_block0_linear_out_dtype", str(out.dtype))

    hs = []
    for name, mod in dm.diffusion_transformer.blocks[0].named_modules():
        if isinstance(mod, torch.nn.Linear):
            probe["probed_module"] = f"diffusion_transformer.blocks.0.{name}"
            hs.append(mod.register_forward_hook(dtype_hook))
            break

    kw = cast(kwargs, dt)
    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else torch.autocast("cpu", enabled=False))
    t1 = time.time()
    with ctx:
        xl = dm(**kw)
    t_fwd = time.time() - t1
    for h in hs:
        h.remove()
    print(f"[{time.time()-t0:.0f}s] forward {t_fwd:.0f}s, xl {tuple(xl.shape)} {xl.dtype}, "
          f"probe {probe}", flush=True)

    names, params = zip(*[(n, p) for n, p in dm.named_parameters()])
    t1 = time.time()
    g = torch.autograd.grad(xl, tuple(params), grad_outputs=cot.to(xl.dtype),
                            allow_unused=True, retain_graph=False)
    t_bwd = time.time() - t1
    print(f"[{time.time()-t0:.0f}s] backward {t_bwd:.0f}s", flush=True)

    grads = {n: (x.detach().to(torch.float32) if x is not None else None)
             for n, x in zip(names, g)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": a.policy, "grads": grads, "cap": str(a.cap)}, a.out)

    # scope reading against the float64 reference, so the arm is scored before it is trusted
    num = den = 0.0
    worst, worst_n, n = -1.0, None, 0
    for nm, gg in grads.items():
        r = ref_grad.get(nm)
        if gg is None or r is None:
            continue
        gg = gg.reshape(-1).double()
        r = r.reshape(-1).double()
        e = float(torch.linalg.vector_norm(gg - r)) ** 2
        s = float(torch.linalg.vector_norm(r)) ** 2
        num += e
        den += s
        n += 1
        rel = (e / s) ** 0.5 if s else float("inf")
        if rel > worst:
            worst, worst_n = rel, nm
    rep = {"what": __doc__.strip().splitlines()[0],
           "policy": a.policy, "cap": str(a.cap), "ckpt": str(CKPT),
           "upstream_pkg": "/home/ttuser/of3t_gradients/of3pkg (openfold3 0.5.0)",
           "forward_seconds": t_fwd, "backward_seconds": t_bwd,
           "xl_dtype": str(xl.dtype), "DTYPE_PROBE": probe,
           "all_738_mass_weighted_vs_f64": (num / den) ** 0.5 if den else None,
           "compared": n, "worst_rel": worst, "worst_tensor": worst_n,
           "out": str(a.out)}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=1, sort_keys=True), flush=True)
    if a.policy == "bf16auto" and probe.get("dit_block0_linear_out_dtype") != "torch.bfloat16":
        print("HARD FAILURE: the bf16 recipe did not reach the kernels", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
