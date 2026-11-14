"""The biased gated in-projection (`trimul_tail.gin_ragged`) at a tile-aligned S against float64:
`[a | b] = (x @ Wp + bp) * sigmoid(x @ Wg + bg)`, a also times the pair mask m[i, j], both channel-moved to
[1, C, S, S] (out[c, i, j] = pre[i, j, c]). AF2's trimul in-projection carries both biases, so this is the
fused route chain 2 needs at BindCraft 2's S = 224. gin_ragged's gate refuses S % 32 == 0; the check calls
it with that gate lifted. The composed ops main runs today (biased linear, sigmoid gate multiply, mask,
permute) are graded against the same reference. One JSON line per case.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/bc4_fuse/gin_af2.py [--n 224] [--reps 10]
"""
import argparse, glob, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--k", type=int, default=128, help="c_z, the in-projection's input width")
ap.add_argument("--c", type=int, default=128, help="hidden, channels of a and of b")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
kcfg = T.trunk_compute_kernel_config(compute_kernel_config())
ckc, grid = MG.ckc_args(kcfg), tuple(T.COMPUTE_GRID_MAIN)
N, K, C = A.n, A.k, A.c
g = torch.Generator().manual_seed(A.seed)
bf = lambda t: t.to(torch.bfloat16).double()  # noqa: E731
x = bf(torch.randn(1, N, N, K, generator=g))
wp, wg = bf(torch.randn(K, 2 * C, generator=g) / K ** 0.5), bf(torch.randn(K, 2 * C, generator=g) / K ** 0.5)
bp, bgv = bf(0.3 * torch.randn(2 * C, generator=g)), bf(0.3 * torch.randn(2 * C, generator=g))
real = 195 if N == 224 else N - N // 8  # BC2's 80 + 115 real tokens of 224
t = (torch.arange(N) < real).double()
mask = (t[:, None] * t[None, :])[None]  # [1, S, S]

pre = (x @ wp + bp) * torch.sigmoid(x @ wg + bgv)  # [1, S, S, 2C]
pre[..., :C] *= mask[..., None]
ref_a, ref_b = (pre[..., :C].permute(0, 3, 1, 2), pre[..., C:].permute(0, 3, 1, 2))


def aiclk():
    out = {}
    for f in sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk")):
        try:
            out[f.split("/")[-2]] = open(f).read().strip()
        except OSError:
            pass
    return out


def up(v, dt=ttnn.bfloat16):
    return ttnn.from_torch(v.to(torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def rel(got, ref):
    return float((got.double() - ref).norm() / ref.norm())


xd, md = up(x), up(mask)
wpT, wgT = up(wp.t().contiguous()), up(wg.t().contiguous())
bias = TT.gin_ragged_bias(bp, bgv, N, dev)
TT.gin_ragged_ok = lambda *a, **k: True  # lift `S % TILE` for the check


def fused():
    return TT.gin_ragged(xd, wpT, wgT, bias, ckc, grid, N, mask=md)


wpd, wgd, bpd, bgd = up(wp), up(wg), up(bp[None]), up(bgv[None])
mu = ttnn.unsqueeze(md, -1)


def composed():
    p = ttnn.linear(xd, wpd, bias=bpd, compute_kernel_config=kcfg)
    q = ttnn.linear(xd, wgd, bias=bgd, compute_kernel_config=kcfg)
    v = ttnn.multiply(p, q, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    a, b = v[..., :C], v[..., C:]
    a = ttnn.multiply(a, mu)
    return ttnn.permute(a, (0, 3, 1, 2)), ttnn.permute(b, (0, 3, 1, 2))


for name, fn in (("gin_biased", fused), ("composed", composed)):
    out = fn()
    if out is None:
        print(json.dumps(dict(case=name, declined=True)), flush=True)
        continue
    ga, gb = (ttnn.to_torch(o).double()[..., :N, :N] for o in out)
    err = dict(rel_a=rel(ga, ref_a), rel_b=rel(gb, ref_b),
               pad_a_max=float(ga[..., real:, :].abs().max()) if real < N else 0.0)
    ts = []
    for _ in range(A.reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        for y in o:
            ttnn.deallocate(y)
    print(json.dumps(dict(case=name, n=N, k=K, c=C, real=real, **err, ms_median=1e3 * statistics.median(ts),
                          ms_min=1e3 * min(ts), aiclk=aiclk())), flush=True)
