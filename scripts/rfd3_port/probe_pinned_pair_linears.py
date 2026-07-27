"""p15 part 2: the same pinned-in0_block_w lever on the [D,I,I,C] @ [C,N] PAIR linears.

probe_pinned_in0_block_w.py established the rule on the DiT token linears
([1,D,I,C] @ [C,N]): an explicit program config with in0_block_w pinned to the default's
value is bit-exact while moving the work from 24 to the full grid (1.66-2.83x). On the pair
linears every candidate FAILED there, because folding the batch into M gives M = D*I*ceil(I/32)
tiles (16000 at D=8/I=250) and the auto-derived out_block_h/out_block_w then do not fit L1.

This probe sweeps out_block_h/out_block_w (which cannot change the K accumulation, hence
cannot change the numbers) and the core grid, over the two pair shapes that are 122 of 1738 ms
of D=8 per-step device time, and prints the full error text for anything that still fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

REPEAT = 3

CASES = [
    ("pair z_transition fc1/fc2", 128, 512, 67.19),
    ("pair z_transition fc3", 512, 128, 37.35),
    ("pair transition_2 fc1/fc2", 128, 256, 55.24),
    ("pair pair_bias to_b", 128, 32, 29.00),
]


def ckc():
    dev = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def tt(x):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=ttnn.bfloat16)


def bench(fn):
    dev = get_device()
    out = fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        out = fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / REPEAT * 1e3, out


def subblock(h_tiles, w_tiles, budget=4):
    for h, w in ((2, 2), (1, 4), (4, 1), (1, 2), (2, 1), (1, 1)):
        if h_tiles % h == 0 and w_tiles % w == 0 and h * w <= budget:
            return h, w
    return 1, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[250])
    ap.add_argument("--batches", type=int, nargs="+", default=[8])
    args = ap.parse_args()

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    print(f"device grid {g.x}x{g.y} = {g.x * g.y} cores")
    kw = dict(compute_kernel_config=ckc(), dtype=ttnn.bfloat16)
    verdict = {}

    for label, K, N, obs_ms in CASES:
        Kt, Nt = K // 32, N // 32
        for I in args.tokens:
            for D in args.batches:
                shape = (D, I, I, K)
                It = (I + 31) // 32
                Mt_fused = D * I * It
                torch.manual_seed(0)
                a = tt(torch.randn(*shape))
                b = tt(torch.randn(K, N))
                t0, o0 = bench(lambda: ttnn.matmul(a, b, **kw))
                ref = ttnn.to_torch(o0).float()
                print(f"\n=== {label} {shape} @ [{K},{N}] Mt_fused={Mt_fused} Nt={Nt} "
                      f"(profile {obs_ms} ms at I=250/D=8)")
                print(f"    {'default':<44s} {t0:8.2f} ms   1.00x")
                cands = []
                for bw in [d for d in (1, 2, 4) if Kt % d == 0]:
                    for ncores in (g.x * g.y, 110):
                        pcM = -(-Mt_fused // ncores)
                        for obh in (1, 2, 4, 8):
                            if pcM % obh:
                                continue
                            for obw in sorted({1, 2, Nt}):
                                if Nt % obw:
                                    continue
                                h, w = subblock(obh, obw)
                                grid = (ttnn.CoreCoord(g.x, g.y) if ncores == g.x * g.y
                                        else ttnn.CoreCoord(11, 10))
                                cands.append((
                                    f"1D fb=1 mc0=0 bw={bw} n={ncores} ob={obh}x{obw}",
                                    ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                                        compute_with_storage_grid_size=grid, in0_block_w=bw,
                                        out_subblock_h=h, out_subblock_w=w,
                                        out_block_h=obh, out_block_w=obw,
                                        per_core_M=pcM, per_core_N=Nt,
                                        fuse_batch=True, mcast_in0=False)))
                            break  # one out_block_h that divides pcM is enough per (bw,ncores)
                for name, pc in cands:
                    try:
                        t, o = bench(lambda: ttnn.matmul(a, b, program_config=pc, **kw))
                        d = (ttnn.to_torch(o).float() - ref).abs().max().item()
                        tag = "EXACT" if d == 0.0 else "BREAKS"
                        print(f"    {name:<44s} {t:8.2f} ms  {t0/t:5.2f}x  maxabs {d:.3e}  {tag}")
                        verdict.setdefault((label, name), []).append((I, D, tag, t0 / t))
                    except Exception as e:
                        print(f"    {name:<44s} FAILED {type(e).__name__}")
                        print(f"        {str(e)[:400]}")
                        verdict.setdefault((label, name), []).append((I, D, "FAIL", 0.0))
                ttnn.deallocate(a)
                ttnn.deallocate(b)

    print("\n=== EXACT everywhere ===")
    for (label, name), rows in sorted(verdict.items()):
        if {r[2] for r in rows} == {"EXACT"}:
            sp = [r[3] for r in rows]
            print(f"  {label:<28s} {name:<44s} {min(sp):.2f}x-{max(sp):.2f}x")


if __name__ == "__main__":
    main()
