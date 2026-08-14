#!/usr/bin/env python3
"""Characterize the fp32 matmul error on p150a (ttnn 0.67.4).

Three probes, K=128, all vs torch fp32 reference on identical inputs:
A. integer-valued inputs (exactly representable in bf16/fp32): exact result expected
   iff multiply+accumulate are true fp32 for representable inputs.
B. randn inputs pre-rounded to bf16, matmul run in fp32: isolates in-kernel rounding
   (input representation error removed by construction).
C. randn fp32 inputs as-is: total error (the screen's number).
"""
import torch
import ttnn

device = ttnn.open_device(device_id=0)
CKC = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

M, K, N = 1024, 128, 128


def run(a_t, w_t, tag):
    ref = a_t @ w_t
    a_d = ttnn.from_torch(a_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    w_d = ttnn.from_torch(w_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    out = ttnn.to_torch(ttnn.linear(a_d, w_d, compute_kernel_config=CKC))
    d = (out - ref).abs()
    rel = (d / ref.abs().clamp_min(1e-6))
    print(f"{tag}: maxabs {d.max():.3e} meanabs {d.mean():.3e} "
          f"maxrel {rel.max():.3e} out_std {ref.std():.3f}", flush=True)


torch.manual_seed(0)
a_int = torch.randint(-3, 4, (M, K)).float()
w_int = torch.randint(-3, 4, (K, N)).float()
run(a_int, w_int, "A integer-valued (bf16-exact inputs)")

a_r = (torch.randn(M, K) / K ** 0.5).bfloat16().float()
w_r = (torch.randn(K, N) / K ** 0.5).bfloat16().float()
run(a_r, w_r, "B bf16-rounded inputs, fp32 matmul")

a_f = torch.randn(M, K) / K ** 0.5
w_f = torch.randn(K, N) / K ** 0.5
run(a_f, w_f, "C full fp32 inputs")

# D: same as C but LoFi, to see if fidelity level moves fp32 at all
CKC_LOFI = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
ref = a_f @ w_f
a_d = ttnn.from_torch(a_f, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
w_d = ttnn.from_torch(w_f, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
out = ttnn.to_torch(ttnn.linear(a_d, w_d, compute_kernel_config=CKC_LOFI))
d = (out - ref).abs()
print(f"D full fp32 inputs, LoFi: maxabs {d.max():.3e} meanabs {d.mean():.3e}", flush=True)

ttnn.close_device(device)
