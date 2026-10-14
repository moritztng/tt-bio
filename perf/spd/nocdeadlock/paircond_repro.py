"""Loop the Protenix-v2 pair-conditioning projection that deadlocked on qb2 chip 3 (2026-10-14).

fp32 [N,N,512] layer_norm -> ttnn.linear(x, w[512,256], core_grid=CORE_GRID_MAIN), HiFi4, fp32 dest,
packer_l1_acc: ttnn picks MatmulMultiCoreReuseMultiCast1DProgramConfig(mcast_in0=False) on 110 cores.
--grid none drops core_grid (the v1 fix for widths <= 4 tiles). Prints one line per --every iterations.
--kz/--kr/--nout/--odt give other models' shapes, e.g. OF3 linear_z: --kz 128 --kr 139 --nout 128 --odt bf16.
usage: TT_VISIBLE_DEVICES=3 timeout -s TERM 1800 python paircond_repro.py --iters 400 [--grid none] [--n 730]
"""
import argparse, time

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=730)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--grid", choices=["main", "none"], default="main")
ap.add_argument("--l1acc", type=int, default=1)
ap.add_argument("--kz", type=int, default=256)
ap.add_argument("--kr", type=int, default=256)
ap.add_argument("--nout", type=int, default=256)
ap.add_argument("--odt", choices=["fp32", "bf16"], default="fp32")
a = ap.parse_args()

dev = T.get_device()
ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                              fp32_dest_acc_en=True, packer_l1_acc=bool(a.l1acc))
g = torch.Generator().manual_seed(0)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
k = a.kz + a.kr
z = up(torch.randn(a.n, a.n, a.kz, generator=g))
rp = up(torch.randn(a.n, a.n, a.kr, generator=g))
lw = up(torch.rand(k, generator=g) + 0.5)
w = up(torch.randn(k, a.nout, generator=g) / k ** 0.5)
kw = dict(compute_kernel_config=ckc, dtype=ttnn.float32 if a.odt == "fp32" else ttnn.bfloat16)
if a.grid == "main":
    kw["core_grid"] = T.CORE_GRID_MAIN
ref = None
t0 = time.time()
for i in range(1, a.iters + 1):
    zc = ttnn.concat([z, rp], dim=-1)
    zc = ttnn.layer_norm(zc, weight=lw, epsilon=1e-5, compute_kernel_config=ckc)
    pb = ttnn.linear(zc, w, **kw)
    ttnn.deallocate(zc)
    if i % a.every == 0 or i == 1:
        h = ttnn.to_torch(pb)
        d = float(h.abs().sum())
        ref = d if ref is None else ref
        print(f"it {i} {time.time() - t0:.1f}s sum={d:.6f} same={d == ref}", flush=True)
    ttnn.deallocate(pb)
print("REPRO-DONE", flush=True)
