#!/usr/bin/env python3
"""W6: why is the tri-attention SDPA at neither roof?

At the real 298-aa shape q[320,8,320,32], bias[1,8,320,320], full-length chunking, SDPA runs
1.84 ms = 18.2 TFLOP/s (18% of the 102.4 TFLOP/s compute roof) and 115 GB/s (29% of the
392 GB/s copy roof). Neither binds. Standing hypothesis: head_dim=32 is exactly one tile, so
QK^T contracts over one tile and PV writes one tile wide, and the matmul cannot block.

Falsifiable prediction: hold batch*heads*head_dim and the sequence length fixed, so total
FLOPs and total bytes are constant, and sweep head_dim. If the 1-tile shape is the mechanism,
time falls sharply as head_dim grows past 32. If time is flat, the mechanism is wrong and the
cost is elsewhere (bias handling, the softmax, dispatch).
"""
import json
import time

import torch
import ttnn

DEV = ttnn.open_device(device_id=0)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
SEQ = 320
BH = 320 * 8 * 32          # batch*heads*head_dim held constant = 81920


def mk(b, h, d):
    t = torch.randn(b, h, SEQ, d) * 0.3
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=DEV, memory_config=DRAM)


def run(b, h, d, with_bias=True, prog=None):
    q, k, v = mk(b, h, d), mk(b, h, d), mk(b, h, d)
    bias = None
    if with_bias:
        bias = ttnn.from_torch(torch.randn(1, h, SEQ, SEQ) * 0.3, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=DRAM)
    kw = dict(attn_mask=bias, is_causal=False, compute_kernel_config=CKC)
    if prog is not None:
        kw["program_config"] = prog

    def call():
        return ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)

    for _ in range(2):
        ttnn.deallocate(call())
    ts = []
    for _ in range(7):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        r = call()
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(r)
    ts.sort()
    for t in (q, k, v):
        ttnn.deallocate(t)
    if bias is not None:
        ttnn.deallocate(bias)
    return ts[len(ts) // 2]


res = {"seq": SEQ, "bh_const": BH, "sweep": []}
print(f"{'b':>5} {'h':>4} {'d':>5} {'bias':>5} {'ms':>9} {'TFLOP/s':>9} {'GB/s':>8}")
for d in (32, 64, 128, 256):
    for h in (8,):
        b = BH // (h * d)
        if b < 1:
            continue
        for wb in (True, False):
            ms = run(b, h, d, with_bias=wb)
            flops = 2 * (2 * b * h * SEQ * SEQ * d)
            byt = 3 * b * h * SEQ * d * 2 + b * h * SEQ * d * 2 + (h * SEQ * SEQ * 2 if wb else 0)
            print(f"{b:>5} {h:>4} {d:>5} {str(wb):>5} {ms:9.4f} "
                  f"{flops/ms*1e-9:9.1f} {byt/ms*1e-6:8.1f}")
            res["sweep"].append(dict(b=b, h=h, d=d, bias=wb, ms=ms,
                                     tflops=flops / ms * 1e-9, gbs=byt / ms * 1e-6))

json.dump(res, open("perf/attn_block/sdpa_headdim_qb2c1.json", "w"), indent=1)
ttnn.close_device(DEV)
