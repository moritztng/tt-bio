"""Where EPI 2's extra error comes from: the tail's residual add in DST against production's `add_`.

On Blackhole EPI 2 (`z + p * sigmoid(g)` added in DST) misses float64 by more than production does
(module rel_rms 0.00874 vs 0.00839, b1/m2) and fails test_trimul_tail_epi's element bound by 1.5
bf16 ULP; on Wormhole it matches production. This prints, per path, the error against the float64
`z + p` and its signed mean along sign(p) (a truncation of p toward zero shows as a negative bias),
for production (EPI 0 then ttnn.add), EPI 1 then ttnn.add, and EPI 2.

usage: TT_VISIBLE_DEVICES=<card> python perf/spd_trimul/epi2_bias.py [--n 160]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=160)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import mm_generic as MG, tenstorrent as T, trimul_tail as TTL
from tt_bio.af2 import compute_kernel_config

CZ, n = 256, A.n
dev = T.get_device()
g = torch.Generator().manual_seed(n)
bf = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).float()
xa, xb, z = bf(1, n, n, CZ), bf(1, n, n, CZ), bf(1, n, n, CZ)
wa, wb = bf(CZ, CZ, sc=CZ ** -0.5), bf(CZ, CZ, sc=CZ ** -0.5)
d = torch.float64
p64 = (xa.to(d) @ wa.to(d)) * torch.sigmoid(xb.to(d) @ wb.to(d))
ref = z.to(d) + p64
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
xa_d, xb_d, wa_d, wb_d = up(xa), up(xb), up(wa), up(wb)
ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
ulp = lambda t: 2.0 ** (torch.floor(torch.log2(t.abs().clamp_min(1e-30))) - 7)


def stats(name, y, yp=None):
    e = y.to(d) - ref
    r = {"path": name, "rel_rms": (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
         "bias_along_p": (e * torch.sign(p64)).mean().item(),
         "max_ulp": (e.abs() / ulp(ref)).max().item(), "mean_ulp": (e.abs() / ulp(ref)).mean().item()}
    if yp is not None:   # the product alone, as the add saw it
        ep = yp.to(d) - p64
        r["p_rel_rms"] = (ep.pow(2).mean().sqrt() / p64.pow(2).mean().sqrt()).item()
        r["p_bias_along_p"] = (ep * torch.sign(p64)).mean().item()
    print(json.dumps(r), flush=True)


for epi in (0, 1):
    TTL.set_epi(epi)
    y = TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid)
    z_d = up(z)
    s = ttnn.add(z_d, y)
    stats(f"epi{epi}+add", ttnn.to_torch(s).float(), ttnn.to_torch(y).float())
    for t in (y, z_d, s):
        ttnn.deallocate(t)
TTL.set_epi(2)
z_d = up(z)
y = TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid, resid=z_d)
stats("epi2", ttnn.to_torch(y).float())
# the same sum with p rounded toward zero to bf16 first, then z + p rounded to nearest: the
# signature a 16-bit DEST->SRCA move would leave
p32 = p64.float()
p_tz = (p32.view(torch.int32) & ~0xFFFF).view(torch.float32)
stats("model:trunc(p)+z,rne", (z + p_tz).to(torch.bfloat16).float())
stats("model:rne(p)+z,rne", (z + p32.to(torch.bfloat16).float()).to(torch.bfloat16).float())
print(json.dumps({"arch": str(dev.arch()), "n": n}))
