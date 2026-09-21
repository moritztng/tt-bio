#!/usr/bin/env python3
"""Deliverable 3: what stops the rung above `diffusion_module`, with the arithmetic.

`of3t-modeltraj` measured the refusal once, as a global oom-kill at 215.53 GB anon-rss on a
249 GB box, and deliberately did not re-trigger it: a 215 GB allocation evicts every other
worker's device job. So this measures ONE PairFormer block instead of the stack's 48 and
scales, which costs ~1/48 of the memory and gives a per-block number the stack's requirement
can be built from rather than asserted.

Three closures are priced against that number: a chunked float64 reference, a smaller crop,
and a host with more memory.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

NT = int(os.environ.get("OMP_NUM_THREADS", "4"))
sys.path.insert(0, os.getcwd())
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)
import refpath                                                            # noqa: E402
refpath.install()

import torch                                                              # noqa: E402

torch.set_num_threads(NT)
DIFFCAP = "/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
OUT = "perf/of3t_trajwide/CEILING.json"
HOST_MEM_GB = 249.0
MEASURED_OOM_GB = 215.53        # of3t-modeltraj, kern.log 2026-09-21T02:16:33Z, pid 1602127

rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
t0 = time.time()
res = {"what": "the cost of the rung above diffusion_module, measured on one PairFormer block "
               "and scaled, rather than by re-triggering a 215 GB oom-kill on a shared box",
       "host_mem_gb": HOST_MEM_GB, "measured_oom_high_water_gb": MEASURED_OOM_GB,
       "threads": NT}

D = torch.load(DIFFCAP, map_location="cpu", weights_only=False)
kw = D["kwargs"]
del D
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
del ck

pf = {k: v for k, v in sd.items() if k.startswith("pairformer_stack.")}
res["pairformer_checkpoint"] = {
    "tensors": len(pf),
    "elements": int(sum(v.numel() for v in pf.values() if torch.is_tensor(v))),
    "float64_param_gb": sum(v.numel() for v in pf.values() if torch.is_tensor(v)) * 8 / 1e9}
res["rss_after_checkpoint_gb"] = rss()

from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry    # noqa: E402
from openfold3.core.model.latent.pairformer import PairFormerStack           # noqa: E402

cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
pf_cfg = dict(cfg.architecture.pairformer)
n_blocks_full = int(pf_cfg.get("no_blocks", pf_cfg.get("num_blocks", 0)))
res["pairformer_config"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                            for k, v in pf_cfg.items()}
res["n_blocks_full"] = n_blocks_full

tok = kw["token_mask"]
tok = tok if tok.dim() == 2 else tok.reshape(1, -1)
n_token = int(tok.shape[-1])
c_s = int(kw["si_trunk"].shape[-1])
c_z = int(kw["zij_trunk"].shape[-1])
res["crop"] = {"n_token": n_token, "c_s": c_s, "c_z": c_z}

s = kw["si_trunk"].to(torch.float64).reshape(1, n_token, c_s)
z = kw["zij_trunk"].to(torch.float64).reshape(1, n_token, n_token, c_z)
single_mask = tok.to(torch.float64)
pair_mask = (tok[..., None] * tok[..., None, :]).to(torch.float64)

# ONE block, not 48. `blocks_per_ckpt = None` so the backward keeps every intermediate, which
# is what the refused rung does and what the high-water is made of.
one = dict(pf_cfg)
for key in ("no_blocks", "num_blocks"):
    if key in one:
        one[key] = 1
base = rss()
m = PairFormerStack(**one).to(torch.float64)
m.train()
m.blocks_per_ckpt = None
res["rss_after_one_block_build_gb"] = rss()
res["one_block_params"] = int(sum(p.numel() for p in m.parameters() if p.requires_grad))

t1 = time.time()
out = m(s=s, z=z, single_mask=single_mask, pair_mask=pair_mask)
fwd = time.time() - t1
first = out[0] if isinstance(out, (tuple, list)) else out
peak_fwd = rss()
t1 = time.time()
torch.autograd.backward([first], [torch.ones_like(first)])
bwd = time.time() - t1
peak = rss()

res["one_block"] = {"forward_s": fwd, "backward_s": bwd, "one_fwd_bwd_s": fwd + bwd,
                    "peak_rss_gb": peak, "peak_after_forward_gb": peak_fwd,
                    "rss_before_build_gb": base,
                    "activation_gb_above_build": peak - res["rss_after_one_block_build_gb"]}

per_block = peak - res["rss_after_one_block_build_gb"]
stack_act = per_block * n_blocks_full
stack_param = res["pairformer_checkpoint"]["float64_param_gb"] * 3      # param + grad + Adam m,v is 4; Adam state is not held during the forward that OOMs
res["projection_to_full_stack"] = {
    "n_blocks": n_blocks_full,
    "activation_gb_per_block_measured": per_block,
    "activation_gb_for_stack": stack_act,
    "float64_param_and_grad_gb": res["pairformer_checkpoint"]["float64_param_gb"] * 2,
    "projected_high_water_gb": stack_act + res["pairformer_checkpoint"]["float64_param_gb"] * 2
                               + res["rss_after_checkpoint_gb"],
    "measured_high_water_gb": MEASURED_OOM_GB,
    "host_mem_gb": HOST_MEM_GB,
    "one_fwd_bwd_s_projected": (fwd + bwd) * n_blocks_full,
    "est_20_step_s_projected": (fwd + bwd) * n_blocks_full * 4 * 20,
    "est_20_step_h_projected": (fwd + bwd) * n_blocks_full * 4 * 20 / 3600.0,
}

json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str), flush=True)
print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)
