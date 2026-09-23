#!/usr/bin/env python3
"""Deliverable 3, part 2: price the three closures against the measured per-block number.

`ceiling.py` measured ONE PairFormer block at the 384-token crop: 14.029 GB of activations
above the build and 56.376 s per float64 fwd+bwd, so the 48-block stack projects to 678.57 GB
and 60.13 h for twenty steps. The 215.53 GB the kernel killed it at is where the allocation
died, not what it needed.

Each closure needs a number, not a direction:
  * a chunked float64 reference   -> `blocks_per_ckpt = 1`, measured here, memory and time
  * a smaller crop                -> the same block at 128 and 256 tokens, so the exponent is
                                     measured rather than assumed quadratic or cubic
  * a host with more memory       -> falls out of the projection
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

NT = int(os.environ.get("OMP_NUM_THREADS", "2"))
sys.path.insert(0, os.getcwd())
_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402
refpath.install()

import torch                                                              # noqa: E402

torch.set_num_threads(NT)
DIFFCAP = "/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
OUT = "perf/of3t_trajwide/CEILING_CLOSURES.json"

D = torch.load(DIFFCAP, map_location="cpu", weights_only=False)
kw = D["kwargs"]
del D
tok = kw["token_mask"]
tok = tok if tok.dim() == 2 else tok.reshape(1, -1)
N_FULL = int(tok.shape[-1])
c_s = int(kw["si_trunk"].shape[-1])
c_z = int(kw["zij_trunk"].shape[-1])

from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry    # noqa: E402
from openfold3.core.model.latent.pairformer import PairFormerStack           # noqa: E402

cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
pf_cfg = dict(cfg.architecture.pairformer)
N_BLOCKS = int(pf_cfg.get("no_blocks", pf_cfg.get("num_blocks", 0)))
one = dict(pf_cfg)
for key in ("no_blocks", "num_blocks"):
    if key in one:
        one[key] = 1

rows = []


def point(n_token, ckpt):
    """One block at `n_token`, with or without per-block activation checkpointing. The peak is
    read from a CHILD process so each point starts from a clean high-water: `ru_maxrss` never
    falls, so a sweep inside one process reports the largest point at every later one."""
    import multiprocessing as mp
    q = mp.Queue()

    def body(q):
        torch.set_num_threads(NT)
        s = kw["si_trunk"].to(torch.float64).reshape(1, N_FULL, c_s)[:, :n_token]
        z = kw["zij_trunk"].to(torch.float64).reshape(1, N_FULL, N_FULL, c_z)[:, :n_token,
                                                                             :n_token]
        sm = tok.to(torch.float64)[:, :n_token]
        pm = (sm[..., None] * sm[..., None, :])
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        m = PairFormerStack(**one).to(torch.float64)
        m.train()
        m.blocks_per_ckpt = 1 if ckpt else None
        built = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        t1 = time.time()
        out = m(s=s, z=z, single_mask=sm, pair_mask=pm)
        fwd = time.time() - t1
        first = out[0] if isinstance(out, (tuple, list)) else out
        t1 = time.time()
        torch.autograd.backward([first], [torch.ones_like(first)])
        bwd = time.time() - t1
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        q.put({"n_token": n_token, "blocks_per_ckpt": (1 if ckpt else None),
               "forward_s": fwd, "backward_s": bwd, "one_fwd_bwd_s": fwd + bwd,
               "rss_before_build_gb": base, "rss_after_build_gb": built,
               "peak_rss_gb": peak, "activation_gb_above_build": peak - built})

    p = mp.Process(target=body, args=(q,))
    p.start()
    r = q.get()
    p.join()
    print(json.dumps(r), flush=True)
    rows.append(r)
    return r


for n in (128, 256, N_FULL):
    point(n, False)
point(N_FULL, True)

full = [r for r in rows if r["n_token"] == N_FULL and r["blocks_per_ckpt"] is None][0]
ckpt = [r for r in rows if r["blocks_per_ckpt"] == 1][0]
import math
small = [r for r in rows if r["blocks_per_ckpt"] is None and r["n_token"] < N_FULL]
exps = [(math.log(full["activation_gb_above_build"] / r["activation_gb_above_build"])
         / math.log(N_FULL / r["n_token"]), r["n_token"]) for r in small]

res = {
    "points": rows,
    "n_blocks_full": N_BLOCKS,
    "activation_scaling_exponent_in_n_token": [{"against_n": n, "exponent": e} for e, n in exps],
    "closures": {
        "chunked_float64_reference": {
            "how": "blocks_per_ckpt = 1, upstream's own activation checkpointing, so one "
                   "block's activations live at a time and the backward recomputes each "
                   "block's forward",
            "measured_one_block_peak_gb": ckpt["peak_rss_gb"],
            "measured_one_block_fwd_bwd_s": ckpt["one_fwd_bwd_s"],
            "time_penalty_vs_unchunked": ckpt["one_fwd_bwd_s"] / full["one_fwd_bwd_s"],
            "projected_stack_high_water_gb": ckpt["activation_gb_above_build"]
                                             + full["rss_after_build_gb"],
            "projected_20_step_h": ckpt["one_fwd_bwd_s"] * N_BLOCKS * 4 * 20 / 3600.0},
        "smaller_crop": {
            "how": "the same block at fewer tokens; the exponent above says what a crop buys",
            "points": [{"n_token": r["n_token"], "activation_gb": r["activation_gb_above_build"],
                        "fwd_bwd_s": r["one_fwd_bwd_s"]} for r in rows
                       if r["blocks_per_ckpt"] is None]},
        "bigger_host": {
            "how": "no code change; the projection is the requirement",
            "projected_stack_high_water_gb": full["activation_gb_above_build"] * N_BLOCKS
                                             + full["rss_after_build_gb"],
            "this_host_gb": 249.0},
    },
}
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str), flush=True)
print(f"wrote {OUT}", flush=True)
