"""p15: is a matmul bit-exact under a core-distribution change when in0_block_w is PINNED?

p14 closed `core_grid=` for K>1-tile matmuls because exactness there is a property of the
whole (M, K, N, D) tuple, not of the call site. Reading tt-metal's config heuristic
(`ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp`) says why: the
K-blocking `in0_block_w` is picked by whichever *branch* of the heuristic fires, and the
branch predicate depends on M/batch/grid --

    create_matmul_1d_systolic_array_program_config:  in0_block_w = (Kt % 2 == 0) ? 2 : 1
    create_matmul_program_config (2D interleaved):   in0_block_w = largest d <= 4 with Kt % d == 0
    create_simple_matmul_program_config:             in0_block_w = 2, demoted to 1 if Kt % 2

so for Kt == 1 every branch collapses to in0_block_w = 1 (which is exactly p14's
single-K-tile rule, now explained at the source rather than measured) and for Kt > 1 the
branches disagree, so any change that flips the branch also regroups the fp32 accumulation.

The hypothesis this probe tests is the constructive consequence: *only* in0_block_w moves the
numbers, so an EXPLICIT program_config that pins in0_block_w to the default's value is
bit-exact under any per_core_M / per_core_N / grid / fuse_batch change. If that holds, the
whole "how many cores does this matmul get" knob becomes safe at any K -- which is the lever
p13 asked for (matmul on 52.6 of 130 cores) and p14 could only take on the K<=1-tile subset.

Shapes are the real fuse_batch=0 / few-core matmuls from p13's D=8 per-step profile
(scripts/rfd3_port/p13_fpu_b8_ops_perf.csv), which together are ~208 ms of 1738 ms device time.

Run (card 0 on pc):
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:<slug> \
      python3 scripts/rfd3_port/probe_pinned_in0_block_w.py [--tokens 40 250] [--batches 1 8]
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
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

REPEAT = 3

# (label, a-shape builder, K, N, observed default core count at D=8/I=250, ms at D=8)
CASES = [
    ("pair z_transition fc1/fc2", lambda D, I: (D, I, I, 128), 128, 512, 16, 67.19),
    ("pair transition_2 fc1/fc2", lambda D, I: (D, I, I, 128), 128, 256, 64, 55.24),
    ("DiT q/k/v/g/o", lambda D, I: (1, D, I, 768), 768, 768, 24, 33.86),
    ("DiT adaLN gain/bias", lambda D, I: (1, D, I, 384), 384, 768, 24, 23.96),
    ("DiT transition fc1/fc2", lambda D, I: (1, D, I, 768), 768, 1536, 48, 15.13),
    ("DiT transition fc3", lambda D, I: (1, D, I, 1536), 1536, 768, 24, 12.66),
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


def divisors_upto(n, cap=8):
    return [d for d in range(1, cap + 1) if n % d == 0]


def subblock(h_tiles, w_tiles, budget=4):
    """out_subblock_h/w with h*w <= budget (fp32_dest_acc_en halves the dest budget)."""
    for h, w in ((2, 2), (1, 4), (4, 1), (1, 2), (2, 1), (1, 1)):
        if h <= h_tiles and w <= w_tiles and h * w <= budget:
            return h, w
    return 1, 1


def candidates(a_shape, K, N, grid):
    """Explicit configs that redistribute cores, parameterised by pinned in0_block_w."""
    Kt, Nt = K // 32, N // 32
    Mt_batched = ((a_shape[-2] + 31) // 32)                       # M tiles per batch entry
    batch = 1
    for d in a_shape[:-2]:
        batch *= d
    Mt_fused = batch * Mt_batched                                 # M tiles if the batch folds in
    ncores = grid.x * grid.y
    out = []
    for bw in divisors_upto(Kt):
        # 1D, batch folded into M, M split across the whole grid (mcast_in0=False)
        pcM = -(-Mt_fused // ncores)
        h, w = subblock(pcM, Nt)
        out.append((f"1D fb=1 mcast_in0=0 bw={bw}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=bw, out_subblock_h=h, out_subblock_w=w,
            per_core_M=pcM, per_core_N=Nt, fuse_batch=True, mcast_in0=False)))
        # 2D, batch folded into M, M over grid rows and N over grid cols
        pcM2 = -(-Mt_fused // grid.y)
        pcN2 = -(-Nt // grid.x)
        h2, w2 = subblock(pcM2, pcN2)
        out.append((f"2D fb=1 bw={bw}", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=bw, out_subblock_h=h2, out_subblock_w=w2,
            per_core_M=pcM2, per_core_N=pcN2, transpose_mcast=False, fuse_batch=True)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[40, 250])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    args = ap.parse_args()

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    print(f"device grid {g.x}x{g.y} = {g.x * g.y} cores; CORE_GRID_MAIN={CORE_GRID_MAIN}")
    kw = dict(compute_kernel_config=ckc(), dtype=ttnn.bfloat16)
    verdict = {}

    for label, ashape, K, N, obs_cores, obs_ms in CASES:
        for I in args.tokens:
            for D in args.batches:
                shape = ashape(D, I)
                torch.manual_seed(0)
                a = tt(torch.randn(*shape))
                b = tt(torch.randn(K, N))
                t0, o0 = bench(lambda: ttnn.matmul(a, b, **kw))
                ref = ttnn.to_torch(o0).float()
                print(f"\n=== {label}  {shape} @ [{K},{N}]  I={I} D={D} "
                      f"(profile: {obs_cores} cores, {obs_ms} ms at I=250/D=8)")
                print(f"    {'default':<34s} {t0:8.2f} ms   1.00x")
                try:
                    tg, og = bench(lambda: ttnn.matmul(a, b, core_grid=CORE_GRID_MAIN, **kw))
                    dg = (ttnn.to_torch(og).float() - ref).abs().max().item()
                    print(f"    {'core_grid=MAIN (p14, closed)':<34s} {tg:8.2f} ms  {t0/tg:5.2f}x  "
                          f"maxabs {dg:.3e}{'  EXACT' if dg == 0.0 else '  BREAKS'}")
                except Exception as e:
                    print(f"    {'core_grid=MAIN (p14, closed)':<34s} FAILED {type(e).__name__}: {str(e)[:70]}")
                for name, pc in candidates(shape, K, N, g):
                    try:
                        t, o = bench(lambda: ttnn.matmul(a, b, program_config=pc, **kw))
                        d = (ttnn.to_torch(o).float() - ref).abs().max().item()
                        tag = "EXACT" if d == 0.0 else "BREAKS"
                        print(f"    {name:<34s} {t:8.2f} ms  {t0/t:5.2f}x  maxabs {d:.3e}  {tag}")
                        verdict.setdefault((label, name), []).append((I, D, tag, t0 / t))
                    except Exception as e:
                        print(f"    {name:<34s} FAILED {type(e).__name__}: {str(e)[:70]}")
                        verdict.setdefault((label, name), []).append((I, D, "FAIL", 0.0))
                ttnn.deallocate(a)
                ttnn.deallocate(b)

    print("\n\n=== SUMMARY: configs that are EXACT at EVERY tested (I, D) ===")
    for (label, name), rows in sorted(verdict.items()):
        tags = {r[2] for r in rows}
        speeds = [r[3] for r in rows if r[2] == "EXACT"]
        if tags == {"EXACT"}:
            print(f"  EXACT-ALL  {label:<28s} {name:<34s} "
                  f"speedup min {min(speeds):.2f}x max {max(speeds):.2f}x")
    print("\n=== configs that break or fail somewhere ===")
    for (label, name), rows in sorted(verdict.items()):
        tags = {r[2] for r in rows}
        if tags != {"EXACT"}:
            detail = " ".join(f"I{r[0]}/D{r[1]}:{r[2]}" for r in rows)
            print(f"  MIXED      {label:<28s} {name:<34s} {detail}")


if __name__ == "__main__":
    main()
