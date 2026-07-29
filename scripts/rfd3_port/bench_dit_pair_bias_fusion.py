"""The fused pair-bias matmul is 11x cheaper than 18 separate ones; the per-block
slice+permute is what eats the win. Try to get rid of it."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import ttnn

I, C, H, NB = 250, 128, 16, 18
dev = ttnn.open_device(device_id=0)
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def up(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, layout=layout, device=dev, dtype=dtype)


def timed(label, fn, reps=8):
    try:
        for _ in range(2):
            fn()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) / reps * 1e3
        print(f"  {label:<52}{ms:9.3f} ms")
        return ms
    except Exception as exc:
        print(f"  {label:<52}   FAILED {type(exc).__name__}: {str(exc)[:70]}")
        return float("nan")


for B in (1, 8):
    print(f"\n=== B={B} I={I} blocks={NB} ===")
    z = up(torch.randn(B, I, I, C))
    ws = [up(torch.randn(C, H)) for _ in range(NB)]
    wfused = up(torch.cat([torch.randn(C, H) for _ in range(NB)], dim=1))

    base = timed(f"A {NB}x separate linear + permute (shipped)",
                 lambda: [ttnn.permute(ttnn.linear(z, w, compute_kernel_config=ckc,
                                                   dtype=ttnn.bfloat16), (0, 3, 1, 2))
                          for w in ws])

    def slice_then_permute():
        big = ttnn.linear(z, wfused, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        return [ttnn.permute(ttnn.slice(big, [0, 0, 0, b * H], [B, I, I, (b + 1) * H]),
                             (0, 3, 1, 2)) for b in range(NB)]

    v1 = timed("B fused -> slice -> permute (per block)", slice_then_permute)

    def permute_then_slice():
        big = ttnn.linear(z, wfused, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        big = ttnn.permute(big, (0, 3, 1, 2))          # [B, 288, I, I], one permute
        return [ttnn.slice(big, [0, b * H, 0, 0], [B, (b + 1) * H, I, I])
                for b in range(NB)]

    v2 = timed("C fused -> ONE permute -> slice dim1 (per block)", permute_then_slice)

    for nm, v in (("B", v1), ("C", v2)):
        if v == v:
            print(f"  {nm}: {base / v:5.2f}x vs shipped   ({base:.2f} -> {v:.2f} ms/recycle)")

ttnn.close_device(dev)
