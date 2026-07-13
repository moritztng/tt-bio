#!/usr/bin/env python3
"""Protenix-v2 reference predict (CPU) for the pharma implementation-parity benchmark.

Recreates the qb1 reference leg (lost when qb1 went down 2026-07-12) on qb2 at the
same production settings as the already-measured device leg: official ByteDance
Protenix 2.0.0, model protenix-v2 (464M params), use_msa (server search),
sampling_steps=200, diffusion_samples=5, N_cycle=10, examples/prot.yaml sequence
(117-res, PDB 7ROA), seeds 0/1, dtype bf16. CPU-only: stubs the CUDA FusedLayerNorm
with a torch LayerNorm and forces the torch triangle kernels (no cuequivariance /
deepspeed CUDA extensions), so it runs on a box with no NVIDIA GPU.

One seed per invocation:
  refenv312/bin/python protenix_ref_predict.py <seed> <out_dir>

Dumps the Protenix prediction tree under <out_dir>/raw and writes a
REF_PREDICT_DONE marker to <out_dir>/REF_PREDICT_DONE when the seed finishes.
Repackage to harness format (structures/prot.cif + results.json) with
scripts/protenix_ref_to_harness.py, then run scripts/pharma_parity.py structures.
"""
import sys, types, numbers, os
os.environ.setdefault("PROTENIX_DATA_ROOT_DIR",
                      "/home/ttuser/pharma_protenix_run/protenix-src/release_data/ccd_cache")
# Stub CUDA FusedLayerNorm with a torch equivalent. MUST precede any import that
# pulls protenix.model.protenix (the runner imports it), else the CUDA extension
# import fails on a CPU-only host.
import torch, torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
torch.set_grad_enabled(False)
_stub = types.ModuleType("protenix.model.layer_norm.layer_norm")
class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, create_scale=True, create_offset=True, eps=1e-5, **kw):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.ones(*normalized_shape)) if create_scale else None
        self.bias = Parameter(torch.zeros(*normalized_shape)) if create_offset else None
    def forward(self, x):
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x
_stub.FusedLayerNorm = FusedLayerNorm
sys.modules["protenix.model.layer_norm.layer_norm"] = _stub

from runner.batch_inference import inference_jsons

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/home/ttuser/pharma_protenix_run/ref_seed{SEED}"
RAW = os.path.join(OUT, "raw")
os.makedirs(OUT, exist_ok=True)
print(f"=== protenix-v2 reference predict: seed={SEED} out={OUT} ===", flush=True)
print("settings: use_msa=True(server) N_step=200 N_sample=5 N_cycle=10 "
      "trimul=torch triatt=torch dtype=bf16 target=7ROA(117res)", flush=True)
inference_jsons(
    json_file="/home/ttuser/pharma_protenix_run/prot_7roa.json",
    out_dir=RAW,
    use_msa=True,
    seeds=[SEED],
    n_cycle=10,
    n_step=200,
    n_sample=5,
    dtype="bf16",
    model_name="protenix-v2",
    trimul_kernel="torch",
    triatt_kernel="torch",
    use_template=False,
    use_seeds_in_json=False,
)
with open(os.path.join(OUT, "REF_PREDICT_DONE"), "w") as f:
    f.write(f"seed={SEED} done\n")
print(f"REF_PREDICT_DONE seed={SEED}", flush=True)
