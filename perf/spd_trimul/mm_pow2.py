"""Isolate the -2^k outliers seen in trimul's input projections (in_t1 --save at N 768).

Both the incumbent path and T1 return g_out = LN(z) @ Wg_out with the same ~12 wrong elements, each off by exactly
-2, -4 or -8. This runs that one matmul (z and weights regenerated exactly as in_t1 draws them) on a host-normalised
bf16 input, so the reference is exact for what the device sees, across fidelity x fp32 dest x packer L1 acc, plus
once on the device's own layer_norm output. Per config: elements off by > 0.5 and how many of them are a -2^k step.
"""
import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=768)
ap.add_argument("--fids", default="HiFi4,HiFi3,HiFi2,LoFi")
ap.add_argument("--out", help="also append each JSON line here")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
base = T.trunk_compute_kernel_config(compute_kernel_config())
C = h = 256
gen = torch.Generator().manual_seed(0)            # in_t1's draw order: weights, then (z, mask) per --ns entry
w = {"g_in": torch.randn(2 * h, C, generator=gen) / C ** 0.5, "p_in": torch.randn(2 * h, C, generator=gen) / C ** 0.5,
     "g_out": torch.randn(C, C, generator=gen) / C ** 0.5, "gamma": 1 + 0.1 * torch.randn(C, generator=gen),
     "beta": 0.1 * torch.randn(C, generator=gen)}
w = {k: v.to(torch.bfloat16) for k, v in w.items()}
for n in (512, A.n):
    z = (torch.randn(1, n, n, C, generator=gen) * 2 + torch.randn(1, n, n, 1, generator=gen)).to(torch.bfloat16)
    torch.rand(1, n, n, generator=gen)
n = A.n
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
zn = torch.nn.functional.layer_norm(z.double(), (C,), w["gamma"].double(), w["beta"].double(), 1e-5).to(torch.bfloat16)
wt = w["g_out"].t().contiguous()
ref = zn.double() @ wt.double()
dz, dzn, dw = up(z), up(zn), up(wt)


def emit(rec):
    print(json.dumps(rec), flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def grade(name, out, r, **kw):
    e = out.double().reshape(r.shape) - r
    bad = (e.abs() > 0.5).nonzero().tolist()
    steps = [round(math.log2(-float(e[tuple(i)]))) for i in bad if float(e[tuple(i)]) < 0
             and abs(math.log2(-float(e[tuple(i)])) - round(math.log2(-float(e[tuple(i)])))) < 0.01]
    emit({"case": name, **kw, "bad": len(bad), "pow2_neg": len(steps), "k": sorted(steps),
          "rel_rms": float((e.square().mean() / r.square().mean()).sqrt()), "first": bad[:6]})


for fid in A.fids.split(","):
    for f32 in (True, False):
        for l1 in (True, False):
            cfg = type(base)(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=base.math_approx_mode,
                             fp32_dest_acc_en=f32, packer_l1_acc=l1)
            o = ttnn.matmul(dzn, dw, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            grade("host_ln", ttnn.to_torch(o), ref, fid=fid, fp32_dest=f32, packer_l1=l1)
            ttnn.deallocate(o)
xn = ttnn.layer_norm(dz, weight=up(w["gamma"]), bias=up(w["beta"]), epsilon=1e-5,
                     compute_kernel_config=T.ln_compute_kernel_config(base))
xh = ttnn.to_torch(xn).reshape(zn.shape)
emit({"case": "device_ln_vs_host_ln", "max_abs": float((xh.double() - zn.double()).abs().max()),
      "differ": int((xh != zn).sum())})
o = ttnn.matmul(xn, dw, compute_kernel_config=base, memory_config=ttnn.DRAM_MEMORY_CONFIG)
grade("device_ln", ttnn.to_torch(o), xh.double() @ wt.double(), fid=str(base.math_fidelity),
      fp32_dest=base.fp32_dest_acc_en, packer_l1=base.packer_l1_acc)
o2 = ttnn.matmul(dzn, dw, compute_kernel_config=base, memory_config=ttnn.DRAM_MEMORY_CONFIG)
o3 = ttnn.matmul(dzn, dw, compute_kernel_config=base, memory_config=ttnn.DRAM_MEMORY_CONFIG)
emit({"case": "repeat_bit_equal", "equal": bool(torch.equal(ttnn.to_torch(o2), ttnn.to_torch(o3)))})
