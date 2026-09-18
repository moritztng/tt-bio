#!/usr/bin/env python3
"""OPM layout: shape legality + float64 correctness of every reformulation.

Untimed. Small size (I=J=64) so a float64 reference fits on host. Every arm is scored against
that reference, not against another approximation.
"""
import sys, json
from pathlib import Path
import torch, ttnn

S, I, J, C, D, CZ = 64, 64, 64, 32, 32, 128
dev = ttnn.open_device(device_id=0)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
g = dev.compute_with_storage_grid_size(); CG = ttnn.CoreGrid(y=g.y, x=g.x)
DRAM = ttnn.DRAM_MEMORY_CONFIG
torch.manual_seed(0)

a_t = (torch.randn(S, I, C) / 8).to(torch.bfloat16)
b_t = (torch.randn(S, J, D) / 8).to(torch.bfloat16)
W_t = (torch.randn(C * D, CZ) / 32).to(torch.bfloat16)
bias_t = (torch.randn(CZ) / 8).to(torch.bfloat16)
scale = 1.0 / S

# ---- float64 reference, from the bf16 inputs (exact arithmetic on the same numbers) ----
a64, b64, W64, bias64 = (x.to(torch.float64) for x in (a_t, b_t, W_t, bias_t))
z64 = torch.einsum('sic,sjd->icdj', a64, b64)                     # [i,c,d,j]
zz = z64.permute(0, 3, 1, 2).reshape(I, J, C * D) * scale          # [i,j,(c,d)]
ref = zz @ W64 + bias64                                            # [i,j,k]

def tt(x, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, layout=layout, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)

def dev_err(out):
    o = ttnn.to_torch(out).to(torch.float64).reshape(I, J, CZ)
    return (float((o - ref).abs().max()), float((o - ref).abs().mean()),
            float(ref.abs().max()))

W = tt(W_t); bias = tt(bias_t.reshape(1, CZ))
Wq_t = W_t.reshape(C, D, CZ).permute(1, 0, 2).reshape(D, C * CZ)   # unused, kept for reference

# =========== ARM 0: the shipped chain ===========
def shipped(a_t, b_t, fold_scale=False):
    a = tt(a_t); b = tt(b_t)
    a = ttnn.permute(a, (1, 2, 0))                    # (I,C,S)
    if fold_scale:
        a = ttnn.multiply(a, scale)
    b = ttnn.permute(b, (2, 1, 0))                    # (D,J,S)
    b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT)
    b = ttnn.reshape(b, (-1, S))
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
    a_flat = ttnn.reshape(a, (I * C, S))
    z = ttnn.matmul(a_flat, b, transpose_b=True, compute_kernel_config=ckc)
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    z = ttnn.reshape(z, (I, C * D, J))
    z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
    z = ttnn.permute(z, (0, 2, 1))                    # (I,J,C*D)
    if not fold_scale:
        z = ttnn.multiply_(z, scale)
    out = ttnn.linear(z, W, bias=bias, compute_kernel_config=ckc, core_grid=CG)
    return out

print("ARM shipped        ", dev_err(shipped(a_t, b_t)))
print("ARM shipped+foldsc ", dev_err(shipped(a_t, b_t, fold_scale=True)))

# =========== ARM 1: flat proj_o (batch merged into M) ===========
def flat_projo(a_t, b_t):
    a = tt(a_t); b = tt(b_t)
    a = ttnn.permute(a, (1, 2, 0))
    a = ttnn.multiply(a, scale)
    b = ttnn.permute(b, (2, 1, 0))
    b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT); b = ttnn.reshape(b, (-1, S))
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
    a_flat = ttnn.reshape(a, (I * C, S))
    z = ttnn.matmul(a_flat, b, transpose_b=True, compute_kernel_config=ckc)
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    z = ttnn.reshape(z, (I, C * D, J))
    z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
    z = ttnn.permute(z, (0, 2, 1))
    z = ttnn.reshape(z, (I * J, C * D))               # free leading merge?
    out = ttnn.linear(z, W, bias=bias, compute_kernel_config=ckc, core_grid=CG)
    return ttnn.reshape(out, (I, J, CZ))
print("ARM flat_projo     ", dev_err(flat_projo(a_t, b_t)))

# =========== ARM 2: (j,d) column order + free tile reinterpret ===========
def jd_order(a_t, b_t, verbose=False):
    a = tt(a_t); b = tt(b_t)
    a = ttnn.permute(a, (1, 2, 0))                    # (I,C,S)
    a = ttnn.multiply(a, scale)
    b = ttnn.permute(b, (1, 2, 0))                    # (J,D,S)
    b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT); b = ttnn.reshape(b, (-1, S))
    b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)           # (J*D, S)
    z = ttnn.matmul(a, b, transpose_b=True, compute_kernel_config=ckc)   # (I,C,J*D)
    if verbose: print("  z", z.shape, z.layout)
    z = ttnn.reshape(z, (I * J, C, D))                # claim: byte-identical reinterpret
    if verbose: print("  z4", z.shape, z.layout)
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    z = ttnn.reshape(z, (I * J, C * D))               # trailing merge, contiguous
    z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
    out = ttnn.linear(z, W, bias=bias, compute_kernel_config=ckc, core_grid=CG)
    return ttnn.reshape(out, (I, J, CZ))
print("ARM jd_order       ", dev_err(jd_order(a_t, b_t, verbose=True)))
ttnn.close_device(dev)
