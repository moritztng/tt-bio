#!/usr/bin/env python3
"""The conditioning's own boundary: their inputs, their outputs, and the artifacts our port needs.

`sub_boundary.pt` holds the conditioning's OUTPUT `(si, zij)`, the reference's cotangent at it,
and the float64 parameter gradients. What it does not hold is the conditioning's INPUTS --
`sub_boundary.py` captured them in a pre-hook and saved only the downstream stages. This takes
them from the same diffusion boundary by rebuilding `DiffusionConditioning` standalone in
float64 and running it at those kwargs, which the reference's `diffusion_module.forward` passes
straight through from its own (`diffusion_module.py:182-190`).

Standalone rather than through the whole model, and that is checked rather than asserted, twice:

  1. **REPRODUCES_OUTPUT** -- the `(si, zij)` it produces must reproduce `sub_boundary`'s
     `cond_out`, or it is not the function the cotangent was taken at.
  2. **COTANGENT_COMPLETE** -- seeding it with `cond_out_cot` must reproduce all 26
     `diffusion_conditioning.*` gradients in `grad_f64`. This is PROTOCOL A20's precondition
     turned into a measurement: a cotangent at a boundary is a complete gradient only if the
     parameters reach the loss through that boundary alone, and the reference's own float64
     answer is the only thing that can say so. `DiffusionConditioning.forward` returns
     `(si, zij)` and has no other entry point, so the expectation is exact agreement -- and an
     expectation is not a check.

It also captures the two host-computed artifacts our device port takes as inputs: the 139-dim
`relpos_complex` and the post-Fourier noise embedding, via hooks on `layer_norm_z`'s input and
`fourier_emb`'s output. Same discipline as `scripts/of3_diffusion_conditioning_golden.py`, so
the device comparison is gated against the exact reference artifacts rather than against our own
re-derivation of the relpos bins and the Fourier features.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402

import torch

BOUND = Path(refpath.DIFFCAP) / "diffusion_boundary.pt"
SUB = Path(refpath.DIFFCAP) / "sub_boundary.pt"
OUT = Path("/home/ttuser/of3t_cond_cap/cond_boundary.pt")
REPORT = Path("perf/of3t_conditioning/capture_cond_boundary.json")
PREFIX = "diffusion_conditioning."


def _rel(x, y):
    x, y = x.double().reshape(-1), y.double().reshape(-1)
    return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=Path, default=BOUND)
    ap.add_argument("--sub", type=Path, default=SUB)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("~/of3-weights/of3-p2-155k.pt").expanduser())
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--force-conditioning", action="store_true", dest="force",
                    help="run the arm INFERENCE takes. The reference drew "
                         "`use_conditioning=False` at this step (a coin flip at p=0.8), so the "
                         "published gradient is the branch with both trunk inputs zeroed -- and "
                         "that is not the branch a user gets. This re-runs the same float64 "
                         "module on the same inputs with the trunk live, seeded with the same "
                         "cotangent, and saves it as a second reference. The cotangent is the "
                         "real one; the branch is not the one it was taken at, so this arm is a "
                         "SYNTHETIC-BRANCH reference and is labelled as one everywhere it is "
                         "reported. Its value is that it exercises the conditioning path "
                         "inference actually runs, against their code in float64.")
    a = ap.parse_args()
    t0 = time.time()

    from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    dtype = torch.float64
    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    kw = B["kwargs"]
    print(f"[{time.time()-t0:.0f}s] boundary kwargs: {sorted(kw)}", flush=True)
    del B

    S = torch.load(a.sub, map_location="cpu", weights_only=False)
    cond_out, cond_cot = S["cond_out"], S["cond_out_cot"]
    ref_grad = {k[len(PREFIX):]: v for k, v in S["grad_f64"].items() if k.startswith(PREFIX)}
    n_ref_all = len(S["grad_f64"])
    del S
    print(f"[{time.time()-t0:.0f}s] sub boundary: cond_out "
          f"{[tuple(t.shape) for t in cond_out]}, cot "
          f"{[None if t is None else tuple(t.shape) for t in cond_cot]}, "
          f"{len(ref_grad)} conditioning gradients of {n_ref_all}", flush=True)

    # Their config, from the same entry point bundle_min builds the model with, so the
    # standalone module is constructed with the dims the reference ran.
    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
    dc_cfg = dict(cfg.architecture.diffusion_module.diffusion_conditioning)
    dc = DiffusionConditioning(**dc_cfg).to(dtype=dtype)
    dc.train()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    head = "diffusion_module." + PREFIX
    own = {k[len(head):]: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
           for k, v in sd.items() if k.startswith(head)}
    inc = dc.load_state_dict(own, strict=False)
    del ck, sd
    print(f"[{time.time()-t0:.0f}s] conditioning built: {len(own)} checkpoint tensors, "
          f"missing {list(inc.missing_keys)}, unexpected {list(inc.unexpected_keys)}", flush=True)

    cap = {}
    hs = [dc.layer_norm_z.register_forward_pre_hook(
              lambda m, i: cap.__setitem__("cat_z", i[0].detach().clone())),
          dc.fourier_emb.register_forward_hook(
              lambda m, i, o: cap.__setitem__("n_emb", o.detach().clone()))]

    drawn = bool(kw["use_conditioning"])
    call = dict(batch=kw["batch"], t=kw["t"], si_input=kw["si_input"],
                si_trunk=kw["si_trunk"], zij_trunk=kw["zij_trunk"],
                use_conditioning=(True if a.force else drawn),
                chunk_size=kw.get("chunk_size"))
    for k in ("t", "si_input", "si_trunk", "zij_trunk"):
        call[k] = call[k].to(dtype)

    params = dict(dc.named_parameters())
    for p in params.values():
        p.requires_grad_(True)
    t1 = time.time()
    si, zij = dc(**call)
    t_fwd = time.time() - t1
    for h in hs:
        h.remove()
    print(f"[{time.time()-t0:.0f}s] forward {t_fwd:.1f}s, si {tuple(si.shape)} "
          f"zij {tuple(zij.shape)}, use_conditioning={call['use_conditioning']}", flush=True)

    # ---- check 1: is this the function the cotangent was taken at? --------------------------
    # Only on the drawn branch. On the forced arm the answer is deliberately non-zero -- it is
    # a different function -- and is reported as the size of the branch, not as an error.
    repro = {"si": _rel(si, cond_out[0]), "zij": _rel(zij, cond_out[1])}
    label = ("branch separation (forced arm, NOT an error)" if a.force
             else "reproduces cond_out")
    print(f"[{time.time()-t0:.0f}s] {label}: si {repro['si']:.3e} "
          f"zij {repro['zij']:.3e}", flush=True)

    # ---- check 2: is the captured cotangent a COMPLETE gradient (A20)? ----------------------
    names, ps = zip(*sorted(params.items()))
    t1 = time.time()
    g = torch.autograd.grad([si, zij], ps,
                            grad_outputs=[cond_cot[0].to(dtype), cond_cot[1].to(dtype)],
                            allow_unused=True)
    t_bwd = time.time() - t1
    worst, worst_n, n_cmp, absent = -1.0, None, 0, []
    per = {}
    for nm, gg in zip(names, g):
        r = ref_grad.get(nm)
        if r is None:
            absent.append(nm)
            continue
        if gg is None:
            per[nm] = None
            absent.append(nm)
            continue
        d = _rel(gg, r)
        per[nm] = d
        n_cmp += 1
        if d > worst:
            worst, worst_n = d, nm
    print(f"[{time.time()-t0:.0f}s] backward {t_bwd:.1f}s; cotangent completeness: "
          f"{n_cmp} of {len(ref_grad)} tensors, worst {worst:.3e} on {worst_n}", flush=True)

    relpos = cap["cat_z"][..., dc.c_z:].detach().clone()
    n_emb = cap["n_emb"].detach().clone()
    assert relpos.shape[-1] == 139, relpos.shape
    assert n_emb.shape[-1] == dc.c_fourier_emb, n_emb.shape

    token_mask = call["batch"]["token_mask"]
    blob = {"si_trunk": call["si_trunk"], "si_input": call["si_input"],
            "zij_trunk": call["zij_trunk"], "relpos": relpos, "n_emb": n_emb,
            "t": call["t"], "token_mask": token_mask,
            "use_conditioning": call["use_conditioning"], "forced": bool(a.force),
            "sigma_data": float(dc.sigma_data),
            "si_ref": si.detach(), "zij_ref": zij.detach(),
            "si_cot": cond_cot[0], "zij_cot": cond_cot[1],
            # On the drawn branch this is the PUBLISHED reference, which traces to
            # bundle_min's own float64 gradient and which the check above reproduced exactly.
            # On the forced branch there is no published gradient -- the reference never ran
            # that branch at this step -- so it is the float64 one just computed, from their
            # code and their weights at their inputs.
            "grad_f64": ({nm: (gg.detach() if gg is not None else None)
                          for nm, gg in zip(names, g)} if a.force
                         else {nm: r for nm, r in ref_grad.items()}),
            "grad_f64_source": ("float64 recomputed on the forced arm" if a.force
                                else "published reference (bundle_min grads_f64_043.pt)")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, a.out)

    rep = {"arm": ("use_conditioning=True, SYNTHETIC BRANCH (the reference drew False at this "
                   "step; the cotangent is theirs, the branch is not the one it was taken at)"
                   if a.force else "use_conditioning as drawn by the reference"),
           "forced": bool(a.force), "drawn_use_conditioning": drawn,
           "reproduces_cond_out": repro,
           "reproduces_cond_out_what":
               "the standalone float64 conditioning against sub_boundary's captured cond_out; "
               "this is the same function the cotangent was taken at, or nothing below it holds",
           "cotangent_complete": {"compared": n_cmp, "reference_tensors": len(ref_grad),
                                  "worst_rel": worst, "worst_tensor": worst_n,
                                  "absent": absent, "per_tensor": per},
           "cotangent_complete_what_note":
               ("on the forced arm neither check can be zero and neither is claimed as one: "
                "the module is the same object validated on the drawn branch, and these two "
                "numbers say how far the two branches are apart" if a.force else "exact"),
           "cotangent_complete_what":
               "A20: the 26 parameter gradients obtained by seeding cond_out with cond_out_cot, "
               "against the reference's own float64 gradients for the same parameters taken "
               "through the whole diffusion module. Agreement means the conditioning's "
               "parameters reach the loss through this boundary alone",
           "grad_f64_source": blob["grad_f64_source"],
           "shapes": {k: (list(v.shape) if torch.is_tensor(v) else v)
                      for k, v in blob.items()
                      if k not in ("grad_f64", "grad_f64_source")},
           "sigma_data": float(dc.sigma_data), "use_conditioning": bool(call["use_conditioning"]),
           "n_sample": int(si.shape[1]), "n_token": int(si.shape[-2]),
           "token_mask_real": int(token_mask.sum()),
           "dc_config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                         for k, v in dc_cfg.items()},
           "forward_seconds": t_fwd, "backward_seconds": t_bwd,
           "saved": {"file": str(a.out), "bytes": a.out.stat().st_size}}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "cotangent_complete"},
                     indent=1, default=str), flush=True)
    print(f"[{time.time()-t0:.0f}s] wrote {a.report} and {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
