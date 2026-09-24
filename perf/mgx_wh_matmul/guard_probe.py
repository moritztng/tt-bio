#!/usr/bin/env python3
"""Every call form the dest-carry guard plans, guard off against guard on, scored against float64.

Forms: plain [M,K]x[K,N]; transpose_b (b stored [N,K]); transpose_a (a stored [K,M]); a 3D batched
left operand; ttnn.linear with bias and a fused sigmoid / gelu. Fault = |err| > 0.25 x the dot
product's own scale (linear_probe.py); `mean_q` is the mean |err| / scale, which must not move
between the arms beyond noise (the plan changes K order, not precision).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=8192)
ap.add_argument("--k", type=int, default=1536)
ap.add_argument("--n", type=int, default=768)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
g = torch.Generator().manual_seed(5000)
M, K, N = a.m, a.k, a.n
A = torch.randn(M, K, generator=g).bfloat16()
B = torch.randn(K, N, generator=g).bfloat16()
bias = (0.5 * torch.randn(1, N, generator=g)).bfloat16()
ref0 = A.double() @ B.double()
scale = ((A.double() ** 2) @ (B.double() ** 2)).sqrt()
dev_t = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
tA, tB, tAt, tBt = dev_t(A), dev_t(B), dev_t(A.t().contiguous()), dev_t(B.t().contiguous())
tA3, tbias = dev_t(A.reshape(4, M // 4, K)), dev_t(bias)
acts = {"sigmoid": torch.sigmoid, "gelu": torch.nn.functional.gelu}
forms = {
    "plain": (lambda: ttnn.matmul(tA, tB, compute_kernel_config=ckc), ref0, 1.0),
    "transpose_b": (lambda: ttnn.matmul(tA, tBt, transpose_b=True, compute_kernel_config=ckc), ref0, 1.0),
    "transpose_a": (lambda: ttnn.matmul(tAt, tB, transpose_a=True, compute_kernel_config=ckc), ref0, 1.0),
    "batched_a": (lambda: ttnn.matmul(tA3, tB, compute_kernel_config=ckc), ref0, 1.0),
    "linear_bias": (lambda: ttnn.linear(tA, tB, bias=tbias, compute_kernel_config=ckc), ref0 + bias.double(), 1.0),
}
for name, f in acts.items():
    forms[f"linear_{name}"] = (lambda name=name: ttnn.linear(tA, tB, bias=tbias, activation=name,
                                                             compute_kernel_config=ckc),
                               f(ref0 + bias.double()), 0.25)  # both activations are 1-Lipschitz-ish
rows = []
for form, (fn, ref, lip) in forms.items():
    for guard in (False, True):
        T._DEST_CARRY_GUARD = guard
        before = dict(T.DEST_CARRY_STATS)
        o = fn()
        took = {k: T.DEST_CARRY_STATS[k] - before[k] for k in before}
        e = ttnn.to_torch(o).double().reshape(ref.shape) - ref
        ttnn.deallocate(o)
        q = e.abs() / (lip * scale).clamp(min=1e-30)
        r = {"form": form, "guard": guard, "took": took, "fault": int((q > 0.25).sum()),
             "max_q": round(float(q.max()), 4), "mean_q": round(float(q.mean()), 6)}
        rows.append(r)
        print(json.dumps(r), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "mkn": [M, K, N], "rows": rows}, indent=1) + "\n")
