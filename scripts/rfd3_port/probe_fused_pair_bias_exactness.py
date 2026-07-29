"""Can the fused DiT pair-bias projection be made BIT-EXACT against the shipped 18 separate
`ttnn.linear(z, [128,16], core_grid=CORE_GRID_MAIN)` calls?

p19 measured the fusion at 4.9x on the whole chain but reverted it: through the shipped
`core_grid=CORE_GRID_MAIN` the fused `[128,288]` matmul is exact at I=150 and at I=40/D=8 and
NOT exact at I=40/D=1 (maxabs 2.236e-1). Exactness is a property of the whole (M,K,N,grid)
tuple -- the K-blocking `in0_block_w` is the only field that regroups the fp32 accumulation
(see the `_calibrate_linear` note in tt_bio/rfd3.py), and N differs (Nt 1 vs 9) between the
two forms, so the heuristic can pick a different one.

This sweeps I x D x {grid variants, explicit in0_block_w} and reports, per cell, whether the
fused output is BITWISE equal to the unfused reference. If one variant is exact everywhere it
is the answer; if not, the shape gate is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] if "scripts" in str(Path(__file__)) else Path("/home/ttuser/.coworker/wt/tt-bio-rfdiffusion3-largedesign-gap-p20")
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

C, H, NB = 128, 16, 18
TOKENS = [40, 80, 96, 128, 150, 180, 227, 250, 256, 300]
BATCHES = [1, 8]


def ckc():
    dev = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def up(t):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=ttnn.bfloat16)


def pc_1d(B, I, N, bw):
    grid = ttnn.CoreCoord(CORE_GRID_MAIN.x, CORE_GRID_MAIN.y)
    mt = (B * I * I) // 32
    nt = N // 32
    per_core_m = -(-mt // (CORE_GRID_MAIN.x * CORE_GRID_MAIN.y))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=bw,
        out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=1,
        per_core_M=per_core_m, per_core_N=max(nt, 1), fuse_batch=True, mcast_in0=False)


def main():
    kw = dict(compute_kernel_config=ckc(), dtype=ttnn.bfloat16)
    variants = ["grid=MAIN", "grid=default"] + [f"pc bw={bw}" for bw in (1, 2, 4)]
    print(f"fused [{C},{NB*H}] vs {NB}x [{C},{H}] @ core_grid=MAIN -- BITWISE equal?\n")
    print(f"{'I':>5s} {'D':>3s} | " + " ".join(f"{v:<13s}" for v in variants))
    print("-" * (11 + 14 * len(variants)))
    rows = []
    for I in TOKENS:
        for D in BATCHES:
            torch.manual_seed(0)
            zt = torch.randn(D, I, I, C)
            wt = [torch.randn(C, H) for _ in range(NB)]
            z = up(zt)
            ws = [up(w) for w in wt]
            wf = up(torch.cat(wt, dim=1))
            # reference: the shipped call, per block, permuted
            ref = [ttnn.to_torch(ttnn.permute(
                ttnn.linear(z, w, core_grid=CORE_GRID_MAIN, **kw), (0, 3, 1, 2)))
                for w in ws]
            cells = []
            for v in variants:
                try:
                    if v == "grid=MAIN":
                        big = ttnn.linear(z, wf, core_grid=CORE_GRID_MAIN, **kw)
                    elif v == "grid=default":
                        big = ttnn.linear(z, wf, **kw)
                    else:
                        bw = int(v.split("=")[1])
                        big = ttnn.linear(z, wf, program_config=pc_1d(D, I, NB * H, bw), **kw)
                    big = ttnn.permute(big, (0, 3, 1, 2))
                    m = 0.0
                    for b in range(NB):
                        got = ttnn.to_torch(ttnn.slice(
                            big, [0, b * H, 0, 0], [D, (b + 1) * H, I, I]))
                        m = max(m, (got.float() - ref[b].float()).abs().max().item())
                    ttnn.deallocate(big)
                    cells.append("EXACT" if m == 0.0 else f"{m:.2e}")
                except Exception as exc:
                    cells.append(f"ERR:{type(exc).__name__[:8]}")
            print(f"{I:5d} {D:3d} | " + " ".join(f"{c:<13s}" for c in cells), flush=True)
            rows.append((I, D, cells))
            ttnn.deallocate(z)
            for w in ws:
                ttnn.deallocate(w)
            ttnn.deallocate(wf)
    print()
    for i, v in enumerate(variants):
        n_ex = sum(1 for _, _, c in rows if c[i] == "EXACT")
        print(f"  {v:<14s} exact in {n_ex}/{len(rows)} cells")


if __name__ == "__main__":
    main()
