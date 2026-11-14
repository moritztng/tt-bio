"""Minimal reproducer for the Wormhole matmul erratum (for an upstream tt-metal report).

bf16 x bf16 -> bf16 on Wormhole, HiFi3, fp32_dest_acc_en, packer_l1_acc, ttnn's own program config,
[23552, 9984] x [23552, 9984]^T. Two output elements come out off by exactly 2.0 against an fp32
product of the same bf16 operands, the same elements on every run and every chip (zmm_full.py,
2026-10-09: (6828, 8513) ref 0.555, (23180, 5313) ref -0.736). Everything else is within 0.02.
K = 312 tiles is a multiple of the 8-wide grid, so ttnn's plan takes K blocks wider than one tile;
at one-tile blocks no wrong element was seen in 6 whole matrices.

usage: TT_VISIBLE_DEVICES=<chip> python erratum_repro.py
"""
import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()

N, K, DEPTH = 23552, 9984, 9947
torch.manual_seed(2000)
a = (torch.randn(N, K) / DEPTH ** 0.5).bfloat16(); a[:, DEPTH:] = 0
b = torch.randn(N, K).bfloat16(); b[:, DEPTH:] = 0
dev = ttnn.open_device(device_id=0)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                             math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
z = ttnn.to_torch(ttnn.matmul(ta, tb, transpose_b=True, compute_kernel_config=ckc)).float()
ttnn.close_device(dev)
for i, j in ((6828, 8513), (23180, 5313)):
    ref = (a[i].float() @ b[j].float()).item()
    print(f"z[{i},{j}] = {z[i, j].item():+.4f}  fp32 reference {ref:+.4f}  error {z[i, j].item() - ref:+.4f}")
rows = torch.tensor([6828, 23180])
err = (z[rows] - a[rows].float() @ b.float().T).abs()
print(f"max error over those two rows: {err.max().item():.4f}; elements over 0.25: {(err > 0.25).sum().item()}")
