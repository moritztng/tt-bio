#!/usr/bin/env python3
"""Minimal upstream repro: Wormhole fp32 accumulation returns single elements off by exactly -2^k.

One output tile, one core, plain ttnn, HiFi4 with fp32_dest_acc_en. Two dot products, each placed in
row 0 of in0 and row 0 of in1 (transpose_b), every other row zero:

  dest   K 512, the trimul probe's draw (torch seed 2016, channel 0, rows 145 / 272). The float64
         product of the same bf16 operands is -16.85. With in0_block_w 8 the K tiles accumulate in
         dest and Wormhole returns -80.8 (-64). in0_block_w 1 returns the right value.
  packer K 1024, one wrong element of Protenix-v2's swiglu out-projection at 1536 tokens
         (repro_packer.pt). Float64 -0.4663. With in0_block_w 1, 2 or 4 and packer_l1_acc the tile
         partials accumulate in L1 and Wormhole returns -0.966 (-0.5). in0_block_w 8 returns the
         right value.

So neither K block is safe: a large block faults in the dest accumulation, a small one in the packer's
L1 accumulation. The miss is always negative, always a power of two one to three binades above the
accumulator, and needs negative products: with both operands non-negative, 268M elements gave none
(stress.py). HiFi3 faults too, 150x less often. Blackhole returned the right value on the dest case
and on the triangle-product census; the packer case has not been run there.

    TT_VISIBLE_DEVICES=<card> python3 perf/mgx_wh_matmul/repro.py

Needs only torch and ttnn. Prints one line per config.
"""
from pathlib import Path

import torch
import ttnn

dev = ttnn.open_device(device_id=0)
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)


def dot(av, bv, fid, w, l1):
    k = av.numel()
    a = torch.zeros(1, 1, 32, k, dtype=torch.bfloat16)
    b = torch.zeros(1, 1, 32, k, dtype=torch.bfloat16)
    a[0, 0, 0], b[0, 0, 0] = av, bv
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev)
    ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False, fp32_dest_acc_en=True,
               packer_l1_acc=l1)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(1, 1), in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
        out_block_h=1, out_block_w=1, per_core_M=1, per_core_N=1, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)
    o = ttnn.matmul(ta, tb, transpose_b=True, compute_kernel_config=ckc, program_config=pc, dtype=ttnn.float32)
    return float(ttnn.to_torch(o)[0, 0, 0, 0])


g = torch.Generator().manual_seed(2016)
A = torch.randn(1, 8, 512, 512, generator=g).bfloat16()
B = torch.randn(1, 8, 512, 512, generator=g).bfloat16()
cap = torch.load(Path(__file__).with_name("repro_packer.pt"))
cases = {
    "dest": (A[0, 0, 145], B[0, 0, 272], [("HiFi4", 8, True), ("HiFi4", 1, True), ("HiFi3", 8, True)]),
    "packer": (cap["a"], cap["b"], [("HiFi4", 1, True), ("HiFi4", 4, True), ("HiFi4", 8, True),
                                    ("HiFi4", 1, False)]),
}
print(f"arch {dev.arch()}")
for name, (av, bv, cfgs) in cases.items():
    ref = float(av.double() @ bv.double())
    print(f"{name}: K {av.numel()}, float64 reference {ref:.4f}")
    for fid, w, l1 in cfgs:
        got = dot(av, bv, fid, w, l1)
        print(f"  {fid} in0_block_w={w} packer_l1_acc={l1}: {got:.4f}  err {got - ref:+.4f}")
ttnn.close_device(dev)
