"""Steady-state cost of the head-split layout ops in the RFD3 token DiT.

The token DiT runs at c_a=768 / n_head=16, so head_dim=48 is not a multiple of
the 32-wide ttnn tile. This measures what that costs against the tile-aligned
alternatives (head_dim 64 by zero-padding, or a matmul that scatters straight
into the padded layout), plus a reference linear so the numbers are comparable
to real model work.

Run:
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... \
    PYTHONPATH=$PWD python3 -m scripts.rfd3_port.bench_layout_ops
"""

from __future__ import annotations

import time

import torch


def main() -> None:
    import ttnn

    from tt_bio.rfd3 import BATCH_INVARIANT_GRID

    device = ttnn.open_device(device_id=0)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False,
        packer_l1_acc=True, math_approx_mode=False,
    )
    I, C, H = 250, 768, 16
    dt = ttnn.bfloat16

    def up(t):
        return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=dt)

    x3 = up(torch.randn(1, I, C))
    x3p = up(torch.randn(1, I, H * 64))
    w = up(torch.randn(C, C))
    wp = up(torch.randn(C, H * 64))
    scatter = up(torch.zeros(C, H * 64))
    heads48 = up(torch.randn(1, I, H, 48))
    heads64 = up(torch.randn(1, I, H, 64))
    hmajor48 = up(torch.randn(1, H, I, 48))
    hmajor64 = up(torch.randn(1, H, I, 64))

    cases = [
        ("linear (1,250,768)@(768,768)  [reference]",
         lambda: ttnn.linear(x3, w, compute_kernel_config=ckc, dtype=dt,
                             core_grid=BATCH_INVARIANT_GRID)),
        ("linear (1,250,768)@(768,1024) [padded ref]",
         lambda: ttnn.linear(x3, wp, compute_kernel_config=ckc, dtype=dt,
                             core_grid=BATCH_INVARIANT_GRID)),
        ("reshape (1,250,768)->(1,250,16,48)   BASELINE",
         lambda: ttnn.reshape(x3, (1, I, H, 48))),
        ("reshape (1,250,1024)->(1,250,16,64)  ALIGNED",
         lambda: ttnn.reshape(x3p, (1, I, H, 64))),
        ("permute (1,250,16,48)->(1,16,250,48) BASELINE",
         lambda: ttnn.permute(heads48, (0, 2, 1, 3))),
        ("permute (1,250,16,64)->(1,16,250,64) ALIGNED",
         lambda: ttnn.permute(heads64, (0, 2, 1, 3))),
        ("permute (1,16,250,48)->(1,16,48,250) BASELINE",
         lambda: ttnn.permute(hmajor48, (0, 1, 3, 2))),
        ("permute (1,16,250,64)->(1,16,64,250) ALIGNED",
         lambda: ttnn.permute(hmajor64, (0, 1, 3, 2))),
        ("permute (1,16,250,48)->(1,250,16,48) BASELINE",
         lambda: ttnn.permute(hmajor48, (0, 2, 1, 3))),
        ("permute (1,16,250,64)->(1,250,16,64) ALIGNED",
         lambda: ttnn.permute(hmajor64, (0, 2, 1, 3))),
        ("reshape (1,250,16,48)->(1,250,768)   BASELINE",
         lambda: ttnn.reshape(heads48, (1, I, C))),
        ("reshape (1,250,16,64)->(1,250,1024)  ALIGNED",
         lambda: ttnn.reshape(heads64, (1, I, H * 64))),
        ("matmul  scatter 768->1024 (fold pad into a matmul)",
         lambda: ttnn.linear(x3, scatter, compute_kernel_config=ckc, dtype=dt,
                             core_grid=BATCH_INVARIANT_GRID)),
        ("pad     (1,250,768)->(1,250,1024)",
         lambda: ttnn.pad(x3, [(0, 0), (0, 0), (0, H * 64 - C)], 0.0)),
        ("rms_norm (1,250,768)",
         lambda: ttnn.rms_norm(x3, epsilon=1e-6, compute_kernel_config=ckc)),
        ("rms_norm (1,250,1024)",
         lambda: ttnn.rms_norm(x3p, epsilon=1e-6, compute_kernel_config=ckc)),
    ]

    N = 60
    print(f"{'case':52s} {'us/call':>9s}")
    for label, fn in cases:
        for _ in range(4):  # compile + warm
            out = fn()
        ttnn.synchronize_device(device)
        start = time.perf_counter_ns()
        for _ in range(N):
            out = fn()
        ttnn.synchronize_device(device)
        us = (time.perf_counter_ns() - start) / N / 1000
        print(f"{label:52s} {us:9.1f}", flush=True)
        ttnn.deallocate(out)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
