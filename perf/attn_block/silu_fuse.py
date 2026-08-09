#!/usr/bin/env python3
"""W6: move the SwiGLU silu off the linear epilogue and onto the multiply that already reads it.

Transition chunk at the real 298-aa Pairformer shape: x[1,32,320,256] -> fc1/fc2 [256,1024],
silu(x1)*x2, all L1 bf16 HiFi4 fp32_dest_acc.

  A (shipped)  linear(activation="silu") ; multiply_(x1, x2)
  B (proposed) linear(plain)             ; multiply(x1, x2, input_tensor_a_activations=[SILU])

The multiply already reads x1 and writes a same-size result, so the silu rides in its operand
read for free. Prediction if the epilogue cost is real SFPU work and not a scheduling artefact:
B costs A minus roughly the whole 0.17 ms epilogue.
"""
import json
import time

import torch
import ttnn

DEV = ttnn.open_device(device_id=0)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
GRID = DEV.core_grid
L1 = ttnn.L1_MEMORY_CONFIG
B, R, C_Z, HID = 32, 320, 256, 1024

xt = torch.randn(1, B, R, C_Z) * 0.5
x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=L1)
w1 = ttnn.from_torch(torch.randn(C_Z, HID) * 0.05, dtype=ttnn.bfloat16,
                     layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=L1)
w2 = ttnn.from_torch(torch.randn(C_Z, HID) * 0.05, dtype=ttnn.bfloat16,
                     layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=L1)


def lin(w, act=None):
    return ttnn.linear(x, w, activation=act, compute_kernel_config=CKC, memory_config=L1,
                       dtype=ttnn.bfloat16, core_grid=GRID)


def arm_A():
    a = lin(w1, "silu")
    b = lin(w2)
    ttnn.multiply_(a, b)
    ttnn.deallocate(b)
    return a


def arm_B():
    a = lin(w1)
    b = lin(w2)
    o = ttnn.multiply(a, b, memory_config=L1,
                      input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return o


def bench(name, fn, iters=11):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(DEV)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(r)
    ts.sort()
    print(f"{name:44s} {ts[len(ts)//2]:8.4f} ms")
    return ts[len(ts) // 2]


res = {}
res["A_fused_epilogue"] = bench("A  linear(silu) + multiply_", arm_A)
res["B_silu_on_multiply"] = bench("B  linear + multiply(a_act=SILU)", arm_B)
res["speedup"] = res["A_fused_epilogue"] / res["B_silu_on_multiply"]
print(f"speedup A/B = {res['speedup']:.3f}x")

ta = ttnn.to_torch(arm_A()).float()
tb = ttnn.to_torch(arm_B()).float()
eq = torch.equal(ta, tb)
rms = ((ta - tb).pow(2).mean().sqrt() / ta.std()).item()
pcc = torch.corrcoef(torch.stack([ta.flatten(), tb.flatten()]))[0, 1].item()
print(f"A vs B  torch.equal={eq}  max|d|={(ta-tb).abs().max().item():.3e}  "
      f"rmsd/std={rms:.3e}  pcc={pcc:.8f}")
res.update(equal=bool(eq), rmsd_over_std=rms, pcc=pcc)

json.dump(res, open("perf/attn_block/silu_fuse_qb2c1.json", "w"), indent=1)
ttnn.close_device(DEV)
