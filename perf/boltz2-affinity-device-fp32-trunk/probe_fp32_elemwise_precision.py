#!/usr/bin/env python3
"""Probe 2: fp32 layernorm / softmax accuracy vs torch fp32, and signed bias of the
fp32 matmul error (systematic truncation would compound; zero-mean noise random-walks).
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

torch.manual_seed(1)

# layernorm over last dim, z-activations scale (unit-ish) and grown-residual scale
for tag, x_t in (
    ("ln unit-scale [192,192,128]", torch.randn(1, 192, 192, 128)),
    ("ln grown-x30 [192,192,128]", torch.randn(1, 192, 192, 128) * 30),
):
    w_t = torch.randn(128) * 0.5 + 1.0
    b_t = torch.randn(128) * 0.1
    ref = torch.nn.functional.layer_norm(x_t, (128,), w_t, b_t, 1e-5)
    x_d = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    w_d = ttnn.from_torch(w_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    b_d = ttnn.from_torch(b_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    out = ttnn.to_torch(ttnn.layer_norm(x_d, weight=w_d, bias=b_d, epsilon=1e-5,
                                        compute_kernel_config=CKC))
    d = (out - ref).abs()
    print(f"{tag}: maxabs {d.max():.3e} meanabs {d.mean():.3e} out_std {ref.std():.3f}",
          flush=True)

# softmax fp32
x_t = torch.randn(768, 192, 192) * 4
ref = torch.softmax(x_t, dim=-1)
x_d = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
out = ttnn.to_torch(ttnn.softmax(x_d, dim=-1))
d = (out - ref).abs()
print(f"softmax fp32 [768,192,192] x4: maxabs {d.max():.3e} meanabs {d.mean():.3e}", flush=True)

# matmul signed bias: does the fp32 matmul error have a systematic sign?
M, K, N = 1024, 128, 128
a_t = torch.randn(M, K) / K ** 0.5
w_t = torch.randn(K, N) / K ** 0.5
ref = a_t @ w_t
a_d = ttnn.from_torch(a_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
w_d = ttnn.from_torch(w_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
out = ttnn.to_torch(ttnn.linear(a_d, w_d, compute_kernel_config=CKC))
err = out - ref
print(f"matmul fp32 signed error: mean {err.mean():.3e} (out_std {ref.std():.3f}) "
      f"meanabs {err.abs().mean():.3e} -> bias/meanabs {err.mean() / err.abs().mean():.4f}",
      flush=True)

ttnn.close_device(device)
