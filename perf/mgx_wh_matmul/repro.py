#!/usr/bin/env python3
"""Minimal upstream repro: Wormhole HiFi4 + fp32 dest acc returns one element off by -64.

One output tile, one core, plain ttnn: row 145 of A and row 272 of B from the trimul probe's
Kt 16 draw (torch seed 2016, channel 0), every other row zero, K = 512 bf16. The float64 product
of the same bf16 operands is -16.85; with in0_block_w = 8 Wormhole returns -80.8. in0_block_w = 1,
HiFi3 and Blackhole return the right value.

    TT_VISIBLE_DEVICES=<card> python3 perf/mgx_wh_matmul/repro.py

Needs only torch and ttnn (no tt-bio). Prints one line per config.
"""
import torch
import ttnn

dev = ttnn.open_device(device_id=0)
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
g = torch.Generator().manual_seed(2016)
s = 512
A = torch.randn(1, 8, s, s, generator=g).bfloat16()
B = torch.randn(1, 8, s, s, generator=g).bfloat16()
a = torch.zeros(1, 1, 32, s, dtype=torch.bfloat16)
b = torch.zeros(1, 1, 32, s, dtype=torch.bfloat16)
a[0, 0, 145 % 32], b[0, 0, 272 % 32] = A[0, 0, 145], B[0, 0, 272]
ref = float(a[0, 0, 145 % 32].double() @ b[0, 0, 272 % 32].double())
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev)
tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev)
print(f"arch {dev.arch()}  float64 reference {ref:.4f}")
for fid, w in (("HiFi4", 8), ("HiFi4", 1), ("HiFi3", 8)):
    ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False, fp32_dest_acc_en=True,
               packer_l1_acc=True)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(1, 1), in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
        out_block_h=1, out_block_w=1, per_core_M=1, per_core_N=1, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)
    o = ttnn.to_torch(ttnn.matmul(ta, tb, transpose_b=True, compute_kernel_config=ckc, program_config=pc,
                                  dtype=ttnn.float32))
    got = float(o[0, 0, 145 % 32, 272 % 32])
    print(f"{fid} in0_block_w={w}: {got:.4f}  err {got - ref:+.4f}")
ttnn.close_device(dev)
