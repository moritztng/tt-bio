#!/usr/bin/env python3
"""The trimul tail's LN input on a real fold, for the centred-weight fold of norm_mix into proj_emit.

    LN(x) @ Wp = rstd * (x @ Wc) + beta @ Wp,   Wc = (I - 11^T / K) diag(gamma) Wp

is exact algebra, so the tail could skip the normalised-x passes. In bf16 its error scales with |x| instead of
|x - mean|, i.e. by sqrt(1 + (mean/std)^2). This hooks every TriangleMultiplicativeBlock.norm_mix in upstream
ESMFold2 (torch trunk on CPU, ttnn ESMC-6B), samples rows of its input and prints, per call, |mean|/std and the
rel RMS error vs float64 of today's form (LN in fp32 -> bf16 -> @ Wp) and the fold (bf16 x @ bf16 Wc, fp32 stats).
Products are exact here (HiFi4); the device's HiFi2 truncation is not modelled.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_trimul/tail_stats.py esmfold2-fast <input.yaml> <out.jsonl>
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model  # noqa: E402
from tt_bio._vendor.esmfold2_hf.modeling_esmfold2_common import TriangleMultiplicativeBlock  # noqa: E402
from tt_bio.esmfold2_runtime import _ESMCAdapter, fold_complex  # noqa: E402
from tt_bio.main import RECYCLING_STEPS, _read_bio_chains  # noqa: E402
from tt_bio.weights import ESMFOLD2_FAST_REPO, ESMFOLD2_REPO, hf_revision  # noqa: E402

model_id, inp, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
repo = ESMFOLD2_FAST_REPO if model_id == "esmfold2-fast" else ESMFOLD2_REPO
torch.set_num_threads(8)
model = ESMFold2Model.from_pretrained(repo, load_esmc=False, revision=hf_revision(repo)).eval()
model._esmc = _ESMCAdapter("biohub/ESMC-6B")
gen = torch.Generator().manual_seed(0)
bf = torch.bfloat16
fh = out.open("w")
calls = [0]


def rel(a, ref):
    return ((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def hook_for(name, blk):
    ln, wp = blk.norm_mix, blk.proj_emit.weight.detach().double().t()     # [K, H]
    g, b, eps = ln.weight.detach().double(), ln.bias.detach().double(), ln.eps
    K = g.numel()
    wc = (wp * g[:, None]) - (wp * g[:, None]).sum(0, keepdim=True) / K      # (I - 11^T/K) diag(g) Wp
    wc_b, wp_b, bias = wc.to(bf).double(), wp.to(bf).double(), (b @ wp)

    def pre(_mod, args):
        x = args[0].detach().reshape(-1, K)
        x = x[torch.randint(0, x.shape[0], (min(4096, x.shape[0]),), generator=gen)].to(bf)   # the device holds bf16
        x64 = x.double()
        mu, var = x64.mean(-1, keepdim=True), x64.var(-1, unbiased=False, keepdim=True)
        ref = ((x64 - mu) / (var + eps).sqrt() * g + b) @ wp
        xf = x.float()
        cur = torch.nn.functional.layer_norm(xf, (K,), ln.weight.float(), ln.bias.float(), eps).to(bf).double() @ wp_b
        m32, s32 = xf.mean(-1, keepdim=True), (xf * xf).mean(-1, keepdim=True)
        rstd = (s32 - m32 * m32 + eps).rsqrt().double()
        fold = (x64 @ wc_b) * rstd + bias
        r = (mu.abs() / var.sqrt().clamp_min(1e-30)).squeeze(-1)
        fh.write(json.dumps({"call": calls[0], "blk": name, "mu_over_sd_med": round(r.median().item(), 3),
                             "mu_over_sd_p99": round(r.quantile(0.99).item(), 3), "mu_over_sd_max": round(r.max().item(), 3),
                             "err_cur": rel(cur, ref), "err_fold": rel(fold, ref)}) + "\n")
        fh.flush()
        calls[0] += 1
    return pre


for name, mod in model.named_modules():
    if isinstance(mod, TriangleMultiplicativeBlock):
        mod.norm_mix.register_forward_pre_hook(hook_for(name, mod))

chains = [(c, s, None, mt, mo) for c, s, _sp, mt, mo in _read_bio_chains(inp, what=model_id)]
with torch.no_grad():
    fold_complex(model, chains, num_loops=RECYCLING_STEPS.get(model_id, 3), num_sampling_steps=2,
                 num_diffusion_samples=1, seed=0)
print("calls", calls[0], flush=True)
