"""Does the encoder's sparse pair-bias pattern pay at the DiT's shape?

p18 step 1 assumed it does (~I/32 off the projection). The DiT differs from the
encoder in two ways that could flip it: z is [B,I,I,128] and must be gathered ON
DEVICE (it is resident since p19 step 3), and n_head=16 not 4, so the scatter
target is 4x taller. Per recycle the real comparison is

    dense : 18 x (linear(z) -> permute -> add mask)
    sparse:  1 x gather  +  18 x (linear(z_g) -> permute -> scatter)

so the gather amortizes over 18 blocks but the scatter does not.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import ttnn

I, C, H, K = 250, 128, 16, 32
NBLOCK = 18
dev = ttnn.open_device(device_id=0)
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def up(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, layout=layout, device=dev, dtype=dtype)


def timed(label, fn, reps=10):
    try:
        for _ in range(3):
            fn()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) / reps * 1e3
        print(f"  {label:<44}{ms:9.3f} ms")
        return ms
    except Exception as exc:
        print(f"  {label:<44}   FAILED {type(exc).__name__}: {str(exc)[:90]}")
        return float("nan")


for B in (1, 8):
    print(f"\n=== B={B} I={I} C={C} n_head={H} K={K} ===")
    z = up(torch.randn(B, I, I, C))
    w = up(torch.randn(C, H))
    mask = up(torch.full((B, H, I, I), -1e4))
    idx = torch.stack([torch.stack([torch.randperm(I)[:K].sort().values
                                    for _ in range(I)]) for _ in range(B)])

    def dense_block():
        pb = ttnn.linear(z, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        pb = ttnn.permute(pb, (0, 3, 1, 2))
        return ttnn.add(pb, mask)

    d1 = timed("dense: linear+permute+add  (per block)", dense_block)

    # device gather of z at the neighbour indices, ttnn.embedding idiom
    flat = (torch.arange(B)[:, None, None] * (I * I)
            + torch.arange(I)[None, :, None] * I + idx).reshape(1, -1).to(torch.int32)
    idx_dev = up(flat, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def gather():
        zr = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        zr = ttnn.reshape(zr, (B * I * I, C))
        g = ttnn.embedding(idx_dev, zr, layout=ttnn.ROW_MAJOR_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.reshape(g, (B, I, K, C))
        return ttnn.to_layout(g, ttnn.TILE_LAYOUT)

    g_ms = timed("sparse: device gather      (per recycle)", gather)

    try:
        zg = gather()
    except Exception as exc:
        print(f"  gather unusable: {exc}"); continue
    scat_idx = up(idx.unsqueeze(1).expand(B, H, I, K).contiguous().to(torch.int32),
                  dtype=ttnn.uint32)

    def sparse_block():
        pb = ttnn.linear(zg, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        pb = ttnn.permute(pb, (0, 3, 1, 2))
        return ttnn.scatter(mask, 3, scat_idx, pb)

    s1 = timed("sparse: linear+permute+scatter (per block)", sparse_block)

    print(f"  --- per recycle ({NBLOCK} blocks) ---")
    print(f"  dense   {d1 * NBLOCK:9.2f} ms")
    print(f"  sparse  {g_ms + s1 * NBLOCK:9.2f} ms   "
          f"(gather {g_ms:.2f} + {NBLOCK}x{s1:.3f})")
    if s1 == s1:
        print(f"  VERDICT: {'WIN' if g_ms + s1 * NBLOCK < d1 * NBLOCK else 'LOSS'} "
              f"{d1 * NBLOCK / (g_ms + s1 * NBLOCK):.2f}x")

ttnn.close_device(dev)
