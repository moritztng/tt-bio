"""S2b -- why the token encoder's Transition matmuls run at a twentieth of the card.

S2 (perf/p42/r4_s2_tokenenc.json) put `Transition` at 517.5 ms/step, 67.5 % of the token-encoder
region and 29 % of the whole step, and every row of it is far under both roofs: fc1 writes 5.95 GB
in 128.3 ms (54 GB/s against a measured 385 GB/s clone roof) while doing 738 GFLOP (5.8 TFLOP/s).
Neither bandwidth nor arithmetic binds. The remaining candidate is the SHAPE: the pair tensor is
[B, I, I, c_z], so `ttnn.linear` sees B*I = 1370 batch entries of a M=685, K=128 matmul.

This screens that, and nothing else. Same FLOPs, four shapes:

  4d      [2,685,685,128] @ [128,N]   the shipped call
  3d      [1370,685,128]  @ [128,N]   leading dims merged; layout-identical, so a free reshape
  2d      [1,1,938450,128] @ [128,N]  the ideal single matmul, built fresh
  2dpad   [1,1,990208,128] @ [128,N]  I padded 685->704, which is what makes the flatten free

plus the measured cost of actually getting from 4d to 2d, because a rewrite has to pay it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

B, I, C = 2, 685, 128
IP = 704  # 685 rounded up to a tile multiple


def timeit(dev, fn, n=5):
    fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roof", type=float, default=385.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                           math_approx_mode=False, fp32_dest_acc_en=False,
                                           packer_l1_acc=True)
    grid = dev.compute_with_storage_grid_size()
    full = ttnn.CoreGrid(y=grid.y, x=grid.x)
    print(f"[grid] {grid.x} x {grid.y} = {grid.x * grid.y} cores", flush=True)

    res = {}
    for N, tag in ((512, "z_transition n=4 (128->512)"), (256, "transition_2 n=2 (128->256)")):
        w = ttnn.from_torch(torch.randn(C, N) * 0.05, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev)
        flops = 2.0 * B * I * I * C * N
        rows = {}

        x4 = ttnn.from_torch(torch.randn(B, I, I, C) * 0.05, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        rows["4d grid=None"] = timeit(dev, lambda: ttnn.linear(
            x4, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16))
        rows["4d grid=full"] = timeit(dev, lambda: ttnn.linear(
            x4, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=full))

        # leading dims merged. [B,I,I,C] and [B*I,I,C] have the SAME tile padding (only the last
        # two dims are padded), so this reshape should be metadata only -- measured below.
        t0 = time.perf_counter()
        x3 = ttnn.reshape(x4, (B * I, I, C))
        ttnn.synchronize_device(dev)
        rows["_reshape 4d->3d"] = time.perf_counter() - t0
        rows["3d grid=None"] = timeit(dev, lambda: ttnn.linear(
            x3, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16))

        x2 = ttnn.from_torch(torch.randn(1, 1, B * I * I, C) * 0.05, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        rows["2d grid=None"] = timeit(dev, lambda: ttnn.linear(
            x2, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16))
        rows["2d grid=full"] = timeit(dev, lambda: ttnn.linear(
            x2, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=full))
        ttnn.deallocate(x2)

        xp4 = ttnn.from_torch(torch.randn(B, IP, IP, C) * 0.05, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=dev)
        t0 = time.perf_counter()
        xp2 = ttnn.reshape(xp4, (1, 1, B * IP * IP, C))
        ttnn.synchronize_device(dev)
        rows["_reshape 4dpad->2d"] = time.perf_counter() - t0
        rows["2dpad grid=None"] = timeit(dev, lambda: ttnn.linear(
            xp2, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16))
        for t in (xp4, x3, w):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass

        print(f"\n=== {tag}  |  {flops / 1e9:.1f} GFLOP per call ===", flush=True)
        print(f"{'arm':24s} {'ms':>9s} {'TFLOP/s':>9s} {'out GB/s':>9s} {'vs 4d':>7s}")
        base = rows["4d grid=None"]
        for k, v in rows.items():
            if k.startswith("_"):
                print(f"{k:24s} {v * 1e3:9.3f}", flush=True)
                continue
            outgb = B * I * I * N * 2 / 1e9 if "pad" not in k else B * IP * IP * N * 2 / 1e9
            print(f"{k:24s} {v * 1e3:9.3f} {flops / v / 1e12:9.2f} {outgb / v:9.1f} "
                  f"{base / v:6.2f}x", flush=True)
        res[tag] = {k: v * 1e3 for k, v in rows.items()}
        res[tag]["_gflop"] = flops / 1e9

    ttnn.close_device(dev)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
        print(f"\n[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
