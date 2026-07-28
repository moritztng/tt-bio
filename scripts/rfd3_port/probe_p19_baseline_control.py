"""p19's "fused DiT pair bias is 4.9x" was measured against a baseline that is not the
shipped call. This is the control that shows it.

`bench_dit_pair_bias_fusion.py` (p19) timed `ttnn.linear(z, w, ...)` with no `core_grid=`.
The shipped projection is `_tuned_linear(p, b_w, core_grid=CORE_GRID_MAIN)` -- the grid hint
alone is most of the claimed win. Against the real call the fusion is worth 1.5-1.8x on the
pair-bias chain, which is a fraction of a percent of a diffusion step, and the per-shape
exactness calibration it needs costs about as much as it saves over a whole design.

Run on the real DiT weights so the numbers are the ones that ship, not synthetic ones.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

import ttnn  # noqa: E402
from tt_bio.rfd3 import CORE_GRID_MAIN, _tuned_linear, build_diffusion_module  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

SHAPES = [(40, 1), (40, 8), (150, 1), (150, 8), (250, 1), (250, 8)]


def timed(fn, reps=5):
    dev = get_device()
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                     map_location="cpu", weights_only=True)
    dit = build_diffusion_module(dmw).diffusion_transformer
    ckc, dt, h = dit.compute_kernel_config, dit.dtype, dit.N_HEAD
    ws = [b.b_w for b in dit.blocks]
    # the fused weight, exactly as a shipped fusion would build it
    wf = ttnn.from_torch(
        torch.cat([dmw[f"diffusion_transformer.blocks.{i}.attention_pair_bias.to_b.weight"].t()
                   for i in range(len(ws))], dim=1),
        layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=dt)

    print(f"per recycle, ms  |  {'p19 baseline':>13s} {'SHIPPED':>9s} {'fused':>8s} | "
          f"{'p19 x':>6s} {'real x':>7s}")
    for I, D in SHAPES:
        torch.manual_seed(0)
        z = ttnn.from_torch(torch.randn(D, I, I, dit.C_PAIR), layout=ttnn.TILE_LAYOUT,
                            device=get_device(), dtype=dt)

        def nogrid():  # what p19 called "shipped"
            return [ttnn.permute(ttnn.linear(z, w, compute_kernel_config=ckc, dtype=dt),
                                 (0, 3, 1, 2)) for w in ws]

        def shipped():
            return [ttnn.permute(_tuned_linear(z, w, ckc=ckc, dtype=dt,
                                               core_grid=CORE_GRID_MAIN), (0, 3, 1, 2))
                    for w in ws]

        def fused():  # p19's winning variant C: one matmul, ONE permute, slice dim 1
            big = ttnn.linear(z, wf, compute_kernel_config=ckc, dtype=dt,
                              core_grid=CORE_GRID_MAIN)
            stack = ttnn.permute(big, (0, 3, 1, 2))
            ttnn.deallocate(big)
            out = [ttnn.slice(stack, [0, i * h, 0, 0], [D, (i + 1) * h, I, I])
                   for i in range(len(ws))]
            ttnn.deallocate(stack)
            return out

        tn, ts, tf = timed(nogrid), timed(shipped), timed(fused)
        print(f"I={I:<4d} D={D:<2d}       |  {tn:13.3f} {ts:9.3f} {tf:8.3f} | "
              f"{tn / tf:6.2f} {ts / tf:7.2f}", flush=True)
        ttnn.deallocate(z)


if __name__ == "__main__":
    main()
