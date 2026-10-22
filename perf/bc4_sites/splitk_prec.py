#!/usr/bin/env python3
"""Where `autograd._split_k` loses precision on [28672,64]^T@[28672,224]: rel L2 against float64 of the
fp32 partials, of each way to sum them, and of ttnn's own plan.

    TT_VISIBLE_DEVICES=<chip> python perf/bc4_sites/splitk_prec.py --out prec.jsonl
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import autograd as ag

dev = T.get_device()
fo = open(A.out, "a")


def emit(**r):
    print(json.dumps(r), flush=True); fo.write(json.dumps(r) + "\n"); fo.flush()


def rel(y, ref):
    return float((ttnn.to_torch(y).double().reshape(ref.shape) - ref).norm() / ref.norm())


cfg = ag.precise_config()
K, M, N, P = 28672, 64, 224, 64
torch.manual_seed(K)
a = ttnn.from_torch(torch.randn(K, M), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
b = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
A64, B64 = ttnn.to_torch(a).double(), ttnn.to_torch(b).double()
ref = A64.T @ B64
pref = torch.einsum("pkm,pkn->pmn", A64.reshape(P, K // P, M), B64.reshape(P, K // P, N))

emit(arm="stock", rel=rel(ttnn.matmul(a, b, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=cfg), ref))
a3, b3 = ttnn.reshape(a, [P, K // P, M]), ttnn.reshape(b, [P, K // P, N])
plan = ag.bmm_program_config(a3, b3, True, dtype=ttnn.float32, compute_kernel_config=cfg)
emit(arm="plan", plan=str(plan)[:400])
for name, pc in (("parts_plan", plan), ("parts_ttnn", None)):
    parts = ttnn.matmul(a3, b3, transpose_a=True, program_config=pc, dtype=ttnn.float32, compute_kernel_config=cfg)
    emit(arm=name, dtype=str(parts.dtype), rel=rel(parts, pref))
    s = ttnn.sum(parts, dim=0, compute_kernel_config=cfg)
    emit(arm=name + "+sum_cfg", dtype=str(s.dtype), rel=rel(s, ref))
    s = ttnn.sum(parts, dim=0)
    emit(arm=name + "+sum_default", dtype=str(s.dtype), rel=rel(s, ref))
    # pairwise fp32 adds over halves
    t, n = parts, P
    while n > 1:
        n //= 2
        lo = ttnn.slice(t, [0, 0, 0], [n, M, N]); hi = ttnn.slice(t, [n, 0, 0], [2 * n, M, N])
        t = ttnn.add(lo, hi, dtype=ttnn.float32)
    emit(arm=name + "+pairwise_add", dtype=str(t.dtype), rel=rel(t, ref))
    # exact fp64 sum of the device partials: the partials' own floor
    emit(arm=name + "+host_sum", rel=float((ttnn.to_torch(parts).double().sum(0) - ref).norm() / ref.norm()))
emit(arm="done")
