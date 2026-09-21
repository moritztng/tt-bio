#!/usr/bin/env python3
"""Deliverable 1: what is the largest scope that runs a taped forward, a backward and an
optimizer step TWENTY times on this card, and what stops each rung above it.

A rung is refused only with a number. Each entry records what was tried, what the live code
says, and the per-step cost measured here rather than quoted from another row.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/ttuser/of3t_refprec/of3pkg043")

import torch                                                              # noqa: E402

DIFFCAP = "/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
OUT = "perf/of3t_modeltraj/SCOPE_LADDER.json"

res = {"what": "per-step cost of upstream's own float64 side, the binding constraint on how "
                "far up the scope ladder a 20-step model-in-the-loop trajectory can go",
       "rungs": {}}
t0 = time.time()

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

print(f"[{time.time()-t0:.0f}s] loading {DIFFCAP}", flush=True)
D = torch.load(DIFFCAP, map_location="cpu", weights_only=False)
kw = D["kwargs"]
print(f"[{time.time()-t0:.0f}s] kwargs: {sorted(kw)}", flush=True)
res["diffusion_boundary_keys"] = sorted(D)
res["diffusion_boundary_kwargs"] = sorted(kw)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
del ck

from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry   # noqa: E402

cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])


def rung(name, build, call_kwargs, n_sub, sub_key=None):
    """Time ONE forward and ONE backward of upstream's own module in float64, and report what
    20 steps x 4 accumulation samples would cost from it."""
    row = {"scope": name}
    try:
        t1 = time.time()
        mod = build()
        row["build_s"] = time.time() - t1
        params = [p for p in mod.parameters() if p.requires_grad]
        row["n_parameters"] = len(params)
        t1 = time.time()
        out = mod(**call_kwargs)
        row["forward_s"] = time.time() - t1
        first = out[0] if isinstance(out, (tuple, list)) else out
        t1 = time.time()
        torch.autograd.backward([first], [torch.ones_like(first)])
        row["backward_s"] = time.time() - t1
        one = row["forward_s"] + row["backward_s"]
        row["one_fwd_bwd_s"] = one
        # One step is the model's own sample axis once, however it is split across the
        # accumulation cycle: splitting it changes how many times the front end is recomputed,
        # not how much of the model's own work is done.
        row["est_20_step_s"] = one * n_sub * 20
        row["est_20_step_h"] = row["est_20_step_s"] / 3600.0
        row["n_sub_per_step"] = n_sub
    except Exception as e:
        import traceback
        traceback.print_exc()
        row["refused"] = f"{type(e).__name__}: {e}"
    res["rungs"][name] = row
    print(f"[{time.time()-t0:.0f}s] {name}: {json.dumps(row)}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)


# ---- rung 1: diffusion_conditioning, the scope this row ran -----------------------------
def build_cond():
    from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
    m = DiffusionConditioning(
        **dict(cfg.architecture.diffusion_module.diffusion_conditioning)).to(torch.float64)
    m.train()
    head = "diffusion_module.diffusion_conditioning."
    m.load_state_dict({k[len(head):]: (v.to(torch.float64)
                                       if torch.is_tensor(v) and v.is_floating_point() else v)
                       for k, v in sd.items() if k.startswith(head)}, strict=False)
    return m


cond_kw = dict(batch=kw["batch"], t=kw["t"][:, :12].to(torch.float64),
               si_input=kw["si_input"].to(torch.float64),
               si_trunk=kw["si_trunk"].to(torch.float64),
               zij_trunk=kw["zij_trunk"].to(torch.float64),
               use_conditioning=bool(kw["use_conditioning"]), chunk_size=kw.get("chunk_size"))
rung("diffusion_module.diffusion_conditioning", build_cond, cond_kw, n_sub=4)


# ---- rung 2: the whole diffusion_module ---------------------------------------------------
def build_diff():
    from openfold3.core.model.structure.diffusion_module import DiffusionModule
    m = DiffusionModule(**dict(cfg.architecture.diffusion_module)).to(torch.float64)
    m.train()
    head = "diffusion_module."
    m.load_state_dict({k[len(head):]: (v.to(torch.float64)
                                       if torch.is_tensor(v) and v.is_floating_point() else v)
                       for k, v in sd.items() if k.startswith(head)}, strict=False)
    return m


diff_kw = {k: (v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v)
           for k, v in kw.items()}
# One noise level instead of 48: the per-structure cost is what the ladder needs, and 48 of
# them in float64 is the thing being priced.
if torch.is_tensor(diff_kw.get("t")):
    diff_kw["t"] = diff_kw["t"][:, :1]
for key in ("xl_noisy", "x_noisy", "xl", "x"):
    if torch.is_tensor(diff_kw.get(key)) and diff_kw[key].dim() >= 3:
        diff_kw[key] = diff_kw[key][:, :1]
rung("diffusion_module", build_diff, diff_kw, n_sub=48)

json.dump(res, open(OUT, "w"), indent=1, default=str)
print(f"wrote {OUT}", flush=True)
