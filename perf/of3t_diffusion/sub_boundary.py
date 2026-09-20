#!/usr/bin/env python3
"""Re-run THEIR diffusion module alone from the captured boundary, and split it at ours.

`capture_diffusion_boundary.py` captured (kwargs, xl_out, dL/dxl_out) of their r = 0 step and
the 738 float64 parameter gradients under `diffusion_module.*`. Our `OF3DiffusionModule` covers
everything in that module EXCEPT `diffusion_conditioning` -- 712 of 738 tensors, 66.39 % of the
diffusion squared norm and 61.19 % of the whole model's -- because on our side the conditioning
is a separate class that feeds it. So our comparison needs one boundary deeper: the CONDITIONED
(si, zij) their conditioning produces, which is what our module takes as input.

This re-runs `diffusion_module` on its own, from the saved kwargs, seeded with the saved
cotangent. The trunk is never touched, so the 307 s forward is not repeated and D19 stays
upstream of everything here.

It is checked rather than asserted: the parameter gradients this re-run produces must reproduce
the ones the original capture saved. If they do, this sub-boundary is the same function the
bundle comparison was taken at; if they do not, nothing below it is worth running.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_gradients"))

import capture_trunk_boundary as CTB  # noqa: E402

BOUND = Path("/home/ttuser/of3t_diffusion_cap/diffusion_boundary.pt")
OUT = Path("/home/ttuser/of3t_diffusion_cap/sub_boundary.pt")
REPORT = Path("perf/of3t_diffusion/sub_boundary.json")


def main() -> int:
    # D23/R126: a boundary is only as good as the bundle it was captured from, so which one
    # is an argument. Defaults are the published 0.5.0 paths, so nothing already taken moves.
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=Path, default=BOUND)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    a = ap.parse_args()
    t0 = time.time()
    import bundle_min as BM

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    kwargs, cot, ref_grad = b["kwargs"], b["cot"], b["grad_f64"]
    print(f"[{time.time()-t0:.0f}s] boundary loaded, cot norm {float(cot.norm()):.6e}", flush=True)

    dtype = torch.float64
    built = BM.build(dtype, 20260919, "cpu", num_recycles=0)
    model = built[1]
    ck = torch.load(CTB.CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    dm = model.diffusion_module
    print(f"[{time.time()-t0:.0f}s] model built", flush=True)

    cap = {}

    def grab(name, which):
        def pre(mod, args, kw):
            cap[f"{name}.in_args"] = args
            cap[f"{name}.in_kwargs"] = kw
        def post(mod, args, kw, out):
            cap[f"{name}.out"] = out
        return pre if which == "pre" else post

    hs = []
    for name, mod in (("cond", dm.diffusion_conditioning), ("enc", dm.atom_attn_enc),
                      ("dit", dm.diffusion_transformer), ("dec", dm.atom_attn_dec)):
        hs.append(mod.register_forward_pre_hook(grab(name, "pre"), with_kwargs=True))
        hs.append(mod.register_forward_hook(grab(name, "post"), with_kwargs=True))

    t1 = time.time()
    with BM.no_autocast():
        xl = dm(**kwargs)
    t_fwd = time.time() - t1
    for h in hs:
        h.remove()
    print(f"[{time.time()-t0:.0f}s] their diffusion forward alone {t_fwd:.0f}s, "
          f"xl {tuple(xl.shape)}", flush=True)

    names, params = zip(*[(n, p) for n, p in dm.named_parameters()])
    cond_out = cap.get("cond.out")
    extra = [t for t in (cond_out if isinstance(cond_out, (tuple, list)) else [cond_out])
             if torch.is_tensor(t) and t.requires_grad]
    t1 = time.time()
    g = torch.autograd.grad(xl, tuple(params) + tuple(extra), grad_outputs=cot,
                            allow_unused=True, retain_graph=False)
    t_bwd = time.time() - t1
    gp, gextra = g[:len(params)], g[len(params):]
    print(f"[{time.time()-t0:.0f}s] pruned backward {t_bwd:.0f}s", flush=True)

    # ---- the re-run must reproduce the capture, or nothing below it is worth running -------
    worst, worst_n, n = -1.0, None, 0
    for nm, gg in zip(names, gp):
        r = ref_grad.get(nm)
        if gg is None or r is None:
            continue
        d = float(torch.linalg.vector_norm(gg.double() - r.double())
                  / (torch.linalg.vector_norm(r.double()) + 1e-300))
        n += 1
        if d > worst:
            worst, worst_n = d, nm
    print(f"[{time.time()-t0:.0f}s] reproduces capture: {n} tensors, worst {worst:.3e} "
          f"on {worst_n}", flush=True)

    rep = {"reproduces_capture": {"n": n, "worst_rel": worst, "worst_tensor": worst_n,
                                  "what": "the re-run against the gradients the original "
                                          "capture saved; this is the same function or the "
                                          "sub-boundary is not the bundle's"},
           "forward_seconds": t_fwd, "backward_seconds": t_bwd,
           "cond_out_kind": str(type(cond_out)),
           "cond_out_shapes": [tuple(t.shape) for t in
                               (cond_out if isinstance(cond_out, (tuple, list)) else [cond_out])
                               if torch.is_tensor(t)],
           "enc_in_kwargs": sorted(cap.get("enc.in_kwargs", {})),
           "dit_in_kwargs": sorted(cap.get("dit.in_kwargs", {})),
           "dec_in_kwargs": sorted(cap.get("dec.in_kwargs", {}))}

    def det(x):
        if torch.is_tensor(x):
            return x.detach().clone()
        if isinstance(x, dict):
            return {k: det(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(det(v) for v in x)
        return x

    blob = {"cond_out": det(cond_out), "cond_out_cot": [None if x is None else x.detach()
                                                        for x in gextra],
            "enc_in": (det(cap.get("enc.in_args")), det(cap.get("enc.in_kwargs"))),
            "dit_in": (det(cap.get("dit.in_args")), det(cap.get("dit.in_kwargs"))),
            "dit_out": det(cap.get("dit.out")),
            "dec_in": (det(cap.get("dec.in_args")), det(cap.get("dec.in_kwargs"))),
            "xl_out": xl.detach(), "cot": cot,
            "grad_f64": {n_: (x.detach() if x is not None else None)
                         for n_, x in zip(names, gp)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, a.out)
    rep["saved"] = {"file": str(a.out), "bytes": a.out.stat().st_size}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[{time.time()-t0:.0f}s] wrote {a.report} and {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
