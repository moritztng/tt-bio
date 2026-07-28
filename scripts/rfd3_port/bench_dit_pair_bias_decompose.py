"""Decompose the DiT sparse pair-bias block to name the limiter."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import ttnn

I, C, H, K = 250, 128, 16, 32
dev = ttnn.open_device(device_id=0)
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def up(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, layout=layout, device=dev, dtype=dtype)


def timed(label, fn, reps=15):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) / reps * 1e3
    print(f"  {label:<46}{ms:9.3f} ms")
    return ms


for B in (1, 8):
    print(f"\n=== B={B} I={I} n_head={H} K={K} ===")
    z = up(torch.randn(B, I, I, C))
    zg = up(torch.randn(B, I, K, C))
    w = up(torch.randn(C, H))
    mask = up(torch.full((B, H, I, I), -1e4))
    idx = torch.stack([torch.stack([torch.randperm(I)[:K].sort().values
                                    for _ in range(I)]) for _ in range(B)])
    scat_idx = up(idx.unsqueeze(1).expand(B, H, I, K).contiguous().to(torch.int32),
                  dtype=ttnn.uint32)

    dl = timed("DENSE  linear  z[B,I,I,128]->[B,I,I,16]",
               lambda: ttnn.linear(z, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16))
    pbd = ttnn.linear(z, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    dp = timed("DENSE  permute -> [B,16,I,I]",
               lambda: ttnn.permute(pbd, (0, 3, 1, 2)))
    pbdp = ttnn.permute(pbd, (0, 3, 1, 2))
    da = timed("DENSE  add mask", lambda: ttnn.add(pbdp, mask))

    sl = timed("SPARSE linear  z[B,I,32,128]->[B,I,32,16]",
               lambda: ttnn.linear(zg, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16))
    pbs = ttnn.linear(zg, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    sp = timed("SPARSE permute -> [B,16,I,32]",
               lambda: ttnn.permute(pbs, (0, 3, 1, 2)))
    pbsp = ttnn.permute(pbs, (0, 3, 1, 2))
    ss = timed("SPARSE scatter into [B,16,I,I] template",
               lambda: ttnn.scatter(mask, 3, scat_idx, pbsp))

    print(f"  projection shrinks {dl / sl:5.2f}x  ({dl:.3f} -> {sl:.3f} ms)")
    print(f"  but scatter {ss:.3f} ms replaces add {da:.3f} ms  (+{ss - da:.3f})")
    print(f"  net per block: dense {dl + dp + da:.3f}  sparse {sl + sp + ss:.3f}")

ttnn.close_device(dev)
