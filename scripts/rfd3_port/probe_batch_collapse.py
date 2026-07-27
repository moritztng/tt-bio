"""Can RFD3's batched matmuls be made multi-core AND batch-invariant at once?

p13's device profile ranked RFD3's matmul time by shape. The top offenders are all
batched matmuls whose core count does not grow with the batch dimension:

    in0=(2000,4,32,32) in1=(2000,4,32,32)      1 core     171.5 ms   (D=8, 20 calls)
    in0=(8,250,256,128) in1=(128,512)         16 cores    101.0 ms
    in0=(8,250,256,128) in1=(128,256)         64 cores     82.8 ms
    in0=(8,250,256,512) in1=(512,128)        110 cores     55.4 ms

`core_grid=CORE_GRID_MAIN` is the obvious fix and is known to break RFD3's bit-exact
batch invariance (see the BATCH_INVARIANT_GRID comment in tt_bio/rfd3.py): ttnn derives
the program config from M = batch*rows, so the K-blocking (`in0_block_w`) moves with the
batch size and the fp32 accumulation regroups.

An explicit `program_config=` does not have that problem: `in0_block_w` is then a fixed
number, independent of M and of batch. Matmul output rows are independent of each other,
so M-blocking cannot change a row's value -- only K-blocking can. So a pinned program
config should be BOTH multi-core AND bit-exact at any batch size.

This probe measures that claim per shape. Every batch lane holds identical data, so a
correct op must return, for lane 0 at D=8, exactly the D=1 result.
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

REPEAT = 8


def ckc():
    dev = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def tt(x):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=get_device(),
                           dtype=ttnn.bfloat16)


def tiles(n):
    return (n + 31) // 32


def subblock(pm, pn, max_dest=4):
    """(h, w), h*w <= max_dest, h | pm, w | pn; prefer wide."""
    best = (1, 1)
    for h in range(1, max_dest + 1):
        for w in range(1, max_dest // h + 1):
            if pm % h == 0 and pn % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def bench(fn, *args, **kw):
    dev = get_device()
    out = fn(*args, **kw)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        out = fn(*args, **kw)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / REPEAT * 1e3, out


def lane0(out, n1):
    return ttnn.to_torch(out).float().flatten()[:n1]


def candidates(m_tiles, k_tiles, n_tiles, batch, grid):
    """Explicit program configs worth trying for this shape, deduped by label."""
    gx, gy = grid
    seen, out = set(), []

    def add(label, pc):
        if label not in seen:
            seen.add(label)
            out.append((label, pc))

    for by in {gy, min(gy, m_tiles)}:
        for bx in {gx, min(gx, n_tiles)}:
            if by < 1 or bx < 1:
                continue
            pm, pn = -(-m_tiles // by), -(-n_tiles // bx)
            sh, sw = subblock(pm, pn)
            add(f"2D mcast {bx}x{by} pm={pm} pn={pn} ibw={k_tiles}",
                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(bx, by),
                    in0_block_w=k_tiles, out_subblock_h=sh, out_subblock_w=sw,
                    per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                    fused_activation=None))

    if batch > 1:
        sh, sw = subblock(m_tiles, n_tiles)
        add(f"reuse(batch) {gx}x{gy} pm={m_tiles} pn={n_tiles} ibw={k_tiles}",
            ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=k_tiles, out_subblock_h=sh, out_subblock_w=sw,
                per_core_M=m_tiles, per_core_N=n_tiles))
    return out


SHAPES = [
    ("gca  [D*250,4,1,32]@[.,32,32]", lambda D: (D * 250, 4, 1, 32), lambda D: (D * 250, 4, 32, 32)),
    ("gca  [D*250,4,1,32]@[.,32,250]", lambda D: (D * 250, 4, 1, 32), lambda D: (D * 250, 4, 32, 250)),
    ("zpair [D,250,250,128]@[128,512]", lambda D: (D, 250, 250, 128), lambda D: (128, 512)),
    ("zpair [D,250,250,128]@[128,256]", lambda D: (D, 250, 250, 128), lambda D: (128, 256)),
    ("zpair [D,250,250,512]@[512,128]", lambda D: (D, 250, 250, 512), lambda D: (512, 128)),
    ("zpair [D,250,250,128]@[128,32]", lambda D: (D, 250, 250, 128), lambda D: (128, 32)),
    ("dit  [1,D,250,768]@[768,768]", lambda D: (1, D, 250, 768), lambda D: (768, 768)),
    ("dit  [1,D,250,384]@[384,768]", lambda D: (1, D, 250, 384), lambda D: (384, 768)),
    # atom-encoder QK^T: K=head_dim=32, so the single-K-tile rule binds. Already at
    # 120/130 cores in p13's profile, so this measures whether the rule is worth
    # applying beyond the 1-core decoder sites or is a no-op there.
    ("encqk [D,4,1959,32]@[D,4,32,1959]", lambda D: (D, 4, 1959, 32), lambda D: (D, 4, 32, 1959)),
]


def make(s0f, s1f, D):
    torch.manual_seed(0)
    s0, s1 = s0f(D), s1f(D)
    a1, b1 = s0f(1), s1f(1)
    a = torch.randn(*a1).repeat(*[s0[i] // a1[i] for i in range(len(s0))])
    b = torch.randn(*b1).repeat(*[s1[i] // b1[i] for i in range(len(s1))])
    return s0, s1, a, b


def run_shape(name, s0f, s1f, D, grid, refs):
    s0, s1, a, b = make(s0f, s1f, D)
    at, bt = tt(a), tt(b)
    kw = dict(compute_kernel_config=ckc(), dtype=ttnn.bfloat16)
    m_tiles, k_tiles, n_tiles = tiles(s0[-2]), tiles(s0[-1]), tiles(s1[-1])
    batch = 1
    for d in s0[:-2]:
        batch *= d
    n1 = 1
    for d in s0f(1)[:-1]:
        n1 *= d
    n1 *= s1f(1)[-1]

    print(f"\n### {name}  D={D}  M={s0[-2]}({m_tiles}t) K={s0[-1]}({k_tiles}t) "
          f"N={s1[-1]}({n_tiles}t) batch={batch}")
    base_t, base = bench(ttnn.matmul, at, bt, **kw)
    ref = refs.get(name)
    if ref is None:
        refs[name] = ref = lane0(base, n1)
    d0 = (lane0(base, n1) - ref).abs().max().item()
    print(f"    {'current core_grid=None':<40s} {base_t:8.2f} ms   1.00x  "
          f"lane0-vs-D1 {d0:.3e}")

    def report(label, fn, *args, **kwargs):
        try:
            t, o = bench(fn, *args, **kwargs)
            d = (lane0(o, n1) - ref).abs().max().item()
            print(f"    {label:<40s} {t:8.2f} ms  {base_t / t:5.2f}x  "
                  f"lane0-vs-D1 {d:.3e}{'  EXACT' if d == 0.0 else ''}")
        except Exception as e:
            print(f"    {label:<40s} FAILED {type(e).__name__}: {str(e)[:56]}")

    report("core_grid=CORE_GRID_MAIN", ttnn.matmul, at, bt, core_grid=CORE_GRID_MAIN, **kw)
    for label, pc in candidates(m_tiles, k_tiles, n_tiles, batch, grid):
        report(label, ttnn.matmul, at, bt, program_config=pc, **kw)
    ttnn.deallocate(at)
    ttnn.deallocate(bt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    grid = (CORE_GRID_MAIN.x, CORE_GRID_MAIN.y)
    print(f"device grid {g.x}x{g.y}, CORE_GRID_MAIN {grid[0]}x{grid[1]}")

    for name, s0f, s1f in SHAPES:
        if args.only and args.only not in name:
            continue
        refs = {}
        for D in args.batches:
            try:
                run_shape(name, s0f, s1f, D, grid, refs)
            except Exception as e:
                print(f"\n### {name} D={D}: SHAPE FAILED {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
