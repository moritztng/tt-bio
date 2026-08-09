#!/usr/bin/env python3
"""W6: is ttnn.linear(activation="silu") a free epilogue, or does it cost more than the matmul?

Real transition_z chunk shape from the Pairformer block at N=320 (298 aa), c_z=256:
[1,32,320,256] @ [256,1024] -> [1,32,320,1024], all L1, bf16, HiFi4 fp32_dest_acc.
The baseline op table has the silu linear at 0.269 ms and the identical plain linear at
0.098 ms, so the epilogue apparently costs 1.7x the matmul. Falsify or confirm.
"""
import json
import time

import torch
import ttnn

DEV = ttnn.open_device(device_id=0)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
GRID = DEV.core_grid
print("core grid", GRID)

B, R, C_Z, HID = 32, 320, 256, 1024
xt = torch.randn(1, B, R, C_Z, dtype=torch.float32) * 0.5
w1 = torch.randn(C_Z, HID, dtype=torch.float32) * 0.05

L1 = ttnn.L1_MEMORY_CONFIG
x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=L1)
w = ttnn.from_torch(w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=L1)

FLOPS = 2 * B * R * C_Z * HID
# read x + weights, write hidden
BYTES_IN = B * R * C_Z * 2 + C_Z * HID * 2
BYTES_OUT = B * R * HID * 2


def bench(name, fn, iters=9):
    for _ in range(3):
        r = fn()
        ttnn.deallocate(r)
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
    ms = ts[len(ts) // 2]
    print(f"{name:38s} {ms:8.4f} ms  {FLOPS/ms*1e-9:7.1f} TFLOP/s  "
          f"{(BYTES_IN+BYTES_OUT)/ms*1e-6:7.1f} GB/s")
    return ms


def lin(act=None):
    return ttnn.linear(x, w, activation=act, compute_kernel_config=CKC,
                       memory_config=L1, dtype=ttnn.bfloat16, core_grid=GRID)


res = {}
res["linear_silu_fused"] = bench("linear(activation=silu)", lambda: lin("silu"))
res["linear_plain"] = bench("linear(plain)", lambda: lin(None))


def plain_then_silu():
    y = lin(None)
    ttnn.silu(y, output_tensor=y)
    return y


res["linear_plus_inplace_silu"] = bench("linear + ttnn.silu in-place", plain_then_silu)


def plain_then_silu_oop():
    y = lin(None)
    z = ttnn.silu(y, memory_config=L1)
    ttnn.deallocate(y)
    return z


res["linear_plus_oop_silu"] = bench("linear + ttnn.silu out-of-place", plain_then_silu_oop)

# what does a bare silu on the hidden tensor cost on its own?
h = lin(None)
res["silu_alone_inplace"] = bench("ttnn.silu in-place alone",
                                  lambda: (ttnn.silu(h, output_tensor=h), ttnn.clone(h))[1])

# ---- can the silu ride on the multiply that already reads x_1? ----
w2t = torch.randn(C_Z, HID, dtype=torch.float32) * 0.05
w2 = ttnn.from_torch(w2t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=L1)
h2 = ttnn.linear(x, w2, compute_kernel_config=CKC, memory_config=L1,
                 dtype=ttnn.bfloat16, core_grid=GRID)
for kw in ("input_tensor_a_activations", "input_tensor_a_activation", "activations",
           "lhs_activations", "activation"):
    try:
        r = ttnn.multiply(h, h2, memory_config=L1, **{kw: [ttnn.UnaryOpType.SILU]})
        ttnn.deallocate(r)
        print(f"multiply kwarg ACCEPTED: {kw}")
        res["multiply_activation_kwarg"] = kw
        break
    except Exception as e:
        print(f"multiply kwarg rejected: {kw}: {str(e)[:90]}")

# ---- numerics: fused silu vs plain+silu ----
a = lin("silu")
b = plain_then_silu()
ta, tb = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
print("fused vs split  torch.equal:", torch.equal(ta, tb),
      " max|d|:", (ta - tb).abs().max().item(),
      " rmsd/std:", ((ta - tb).pow(2).mean().sqrt() / ta.std()).item())
res["fused_vs_split_equal"] = bool(torch.equal(ta, tb))

json.dump(res, open("/tmp/silu_probe.json", "w"), indent=1)
print(json.dumps(res, indent=1))
ttnn.close_device(DEV)
