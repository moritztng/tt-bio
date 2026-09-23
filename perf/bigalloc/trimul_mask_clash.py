"""Reproduce OpenDDE's 1280-residue refiner trimul clash in isolation.

mgx-speed's cdk2x2_1280 fold (single sequence) died after the trunk with "Statically allocated
circular buffers ... clash with L1 buffers", L1 buffer at 1325056, CB region ending at 1344800, on
the trimul channel matmul. The L1 buffer is 174080 B per core, which is the moved pair mask at a
78-tile structural axis on the 8x9 grid. This runs one masked trimul at that shape with random
weights and prints where the mask went, whether the matmul compiled, and whether the default
route equals a run with the mask in DRAM from the start.

    python3 perf/bigalloc/trimul_mask_clash.py 2496 [2016 3008 ...]
"""
import sys

import torch
import ttnn

from tt_bio import tenstorrent as tt


def _ck(dev):
    return ttnn.types.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def weights(c_z, hidden, g):
    r = lambda *s: torch.randn(*s, generator=g) * s[-1] ** -0.5  # noqa: E731
    return {"norm_in.weight": torch.ones(c_z), "norm_in.bias": torch.zeros(c_z),
            "norm_out.weight": torch.ones(hidden), "norm_out.bias": torch.zeros(hidden),
            "g_in.weight": r(2 * hidden, c_z), "p_in.weight": r(2 * hidden, c_z),
            "g_out.weight": r(c_z, c_z), "p_out.weight": r(c_z, hidden)}


def main():
    dev = tt.get_device()
    print("l1 unreserved/core", ttnn.get_max_worker_l1_unreserved_size(),
          "grid", tt.COMPUTE_GRID_MAIN, flush=True)
    g = torch.Generator().manual_seed(0)
    c_z, hidden = 384, 128
    for ending in (False, True):
        tm = tt.TriangleMultiplication(ending, weights(c_z, hidden, g), _ck(dev), gated_move=True)
        for s in map(int, sys.argv[1:]):
            x = ttnn.from_torch(torch.randn(1, s, s, c_z, generator=g), layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16)
            mask = ttnn.from_torch(torch.ones(1, s, s), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
            outs = []
            for arm in ("default", "mask-dram"):
                tt._TRIMUL_MASK_L1 = arm == "default"
                tt._TRIMUL_MASK_DRAM_SHAPES.clear()
                before = list(tt.TRIMUL_MASK_L1_STATS)
                try:
                    out = tm(x, mask=mask)
                    outs.append(ttnn.to_torch(out))
                    ttnn.deallocate(out)
                    verdict = "OK"
                except RuntimeError as e:
                    verdict = "THROW " + " ".join(
                        ln.strip() for ln in str(e).splitlines() if "clash" in ln)[:300]
                l1, dram = (a - b for a, b in zip(tt.TRIMUL_MASK_L1_STATS, before))
                print(f"S={s} ending={ending} {arm}: mask reads l1={l1} dram={dram}, "
                      f"dram-shapes={len(tt._TRIMUL_MASK_DRAM_SHAPES)}: {verdict}", flush=True)
            if len(outs) == 2:
                print(f"S={s} ending={ending} default == mask-dram bit for bit: "
                      f"{torch.equal(outs[0], outs[1])}", flush=True)
            tt._TRIMUL_MASK_L1 = True
            for t in (x, mask):
                if t.is_allocated():
                    ttnn.deallocate(t)


if __name__ == "__main__":
    main()
