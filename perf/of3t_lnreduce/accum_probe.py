#!/usr/bin/env python3
"""of3t-lnreduce step 1: READ the accumulator. Both of them.

`dW = sum_positions (dy * xhat)` lands through `tt_bio/autograd.py::_sum_leading`, which is
`ttnn.sum(flat2d, dim=0, keepdim=True, compute_kernel_config=precise_config())`. A source read
says fp32 destination accumulation and HiFi4, and a source read is not the answer: what matters
is the arithmetic the kernel performs, which is decided by the packer, the output dtype and the
order as much as by the config. So this probes it with inputs whose exact sums are known, and
reads the accumulator off the answers.

The discriminating input is ALL ONES. Summing K ones:

  exact / fp32 accumulator     -> K exactly, for every K below 2^24
  bf16 SEQUENTIAL accumulator  -> stalls near 256, where 1.0 falls under half an ULP
  bf16 TREE / pairwise         -> exact at powers of two, small error otherwise
  fp32 within a 32-row tile then bf16 across tiles -> exact to ~8192, then stalls

and a second probe, every value the bf16 rounding of 1/3, separates "the accumulator is exact on
integers" from "the accumulator is exact".

Every reference here is the float64 sum of the values the CARD actually holds, read back with
`ttnn.to_torch`, so the input rounding is excluded by construction and what is left is the
accumulator alone. Reported per rung: the shipped call, the same call asking for an fp32 output,
the call with no kernel config, a LoFi control that must be worse, and torch's own reductions.
"""
from __future__ import annotations
import json, os, socket, subprocess, sys, time

sys.path.insert(0, os.getcwd())
OUT = sys.argv[1] if len(sys.argv) > 1 else "perf/of3t_lnreduce/ACCUM_PROBE.json"

import torch
import ttnn
from tt_bio import autograd as ag
from tt_bio.tenstorrent import get_device

C = 128                       # OF3 pair channel: the width pair_transition.layer_norm reduces to
KS = [32, 256, 1024, 8192, 32768, 147456]

dev = get_device()


def to_card(t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def shipped_sum(v, fp32_out=False):
    cfg = ag.precise_config()
    if fp32_out:
        return ttnn.sum(v, dim=0, keepdim=True, compute_kernel_config=cfg, dtype=ttnn.float32)
    return ttnn.sum(v, dim=0, keepdim=True, compute_kernel_config=cfg)


def lofi_sum(v):
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                           math_approx_mode=True, fp32_dest_acc_en=False,
                                           packer_l1_acc=False)
    return ttnn.sum(v, dim=0, keepdim=True, compute_kernel_config=cfg)


def noconfig_sum(v):
    return ttnn.sum(v, dim=0, keepdim=True)


def rel(a, b):
    n = torch.linalg.vector_norm(b).item()
    return (torch.linalg.vector_norm(a - b).item() / n) if n > 0 else float("nan")


def probe(name, mk):
    rows = []
    for K in KS:
        t = mk(K)
        v = to_card(t)
        held = ttnn.to_torch(v).to(torch.float64)        # EXACTLY what the card holds
        exact = held.sum(dim=0)                          # float64 reference, no input error
        denom = torch.linalg.vector_norm(exact).item()
        r = {"K": K, "ref_l2": denom,
             "cancellation": (held.abs().sum(dim=0).norm().item() / denom) if denom > 0 else None}
        for tag, fn in (("shipped", lambda v: shipped_sum(v)),
                        ("fp32_out", lambda v: shipped_sum(v, True)),
                        ("no_config", noconfig_sum),
                        ("lofi", lofi_sum)):
            try:
                got = ttnn.to_torch(fn(v)).to(torch.float64).reshape(-1)
                r[tag] = {"rel_l2": rel(got, exact), "first": got[0].item(),
                          "max_abs_err": (got - exact).abs().max().item()}
            except Exception as e:
                r[tag] = {"error": type(e).__name__ + ": " + str(e)[:200]}
        r["host_f64"] = {"rel_l2": 0.0,
                         "note": "the reference IS this reduction, by construction"}
        hb = held.to(torch.bfloat16).sum(dim=0).to(torch.float64)
        r["torch_bf16_sum"] = {"rel_l2": rel(hb, exact), "first": hb[0].item()}
        hb32 = held.to(torch.float32).sum(dim=0).to(torch.float64)
        r["torch_fp32_sum"] = {"rel_l2": rel(hb32, exact), "first": hb32[0].item()}
        rows.append(r)
        print("  %s K=%7d shipped_first=%r rel=%r exact_first=%r"
              % (name, K, r.get("shipped", {}).get("first"),
                 r.get("shipped", {}).get("rel_l2"), exact[0].item()), flush=True)
        ttnn.deallocate(v)
    return rows


torch.manual_seed(20260922)

PROBES = {
  "ones": ("all 1.0: the dtype identifier. An fp32 accumulation returns K exactly; a bf16 "
           "sequential accumulator stalls near 256.",
           lambda K: torch.ones(K, C)),
  "third": ("all bf16(1/3): exact on integers is not exact.",
            lambda K: torch.full((K, C), 1.0 / 3.0)),
  "gauss": ("iid N(0,1): the real regime, where the true sum cancels and any rounding in the "
            "accumulator is multiplied by that cancellation.",
            lambda K: torch.randn(K, C)),
  "spike": ("row 0 = 1.0, the rest 2^-10: the ORDER probe. A big-first sequential accumulator "
            "loses every small term once the running sum dominates it.",
            lambda K: torch.cat([torch.ones(1, C), torch.full((K - 1, C), 2.0 ** -10)])),
}

res = {"what": "What the affine reduction's accumulator actually is, on the shipped call and on "
               "torch's, read from inputs whose exact sums are known.",
       "shipped_call": "ttnn.sum(flat2d, dim=0, keepdim=True, "
                       "compute_kernel_config=precise_config()) -- tt_bio/autograd.py "
                       "_sum_leading, reached by layer_norm's gamma and beta gradients",
       "precise_config": {"math_fidelity": "HiFi4", "math_approx_mode": False,
                          "fp32_dest_acc_en": True, "packer_l1_acc": True},
       "channel": C, "K_rungs": KS,
       "host": socket.gethostname(),
       "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
       "torch": torch.__version__,
       "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "probes": {}}

for nm, (why, mk) in PROBES.items():
    print("== " + nm, flush=True)
    res["probes"][nm] = {"why": why, "rungs": probe(nm, mk)}

res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(res, open(OUT, "w"), indent=1)
print("wrote " + OUT)
