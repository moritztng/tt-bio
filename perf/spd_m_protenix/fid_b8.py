"""Is a matmul whose operands are both bfp8 bit-identical at HiFi2 and HiFi3/HiFi4?

bfp8_b carries 7 mantissa bits per element under a shared exponent. If the fidelity phases only split mantissa bits
that bfp8 does not have, the extra phases of HiFi3/HiFi4 add zeros and HiFi2 is exact for it. Protenix-v2 --fast runs
its outer-product-mean contraction this way at HiFi3 (opm_b8, 1.62 s per 512 fold on Wormhole, math-bound).

Shapes are that contraction's: x [M, K] times y [M, K]^T, both bfp8. Also runs bf16 x bfp8 and bfp8 x bf16, so the
output says which operand the extra phases read. Prints per arm ms/call (median of --reps after 2 warm calls),
torch.equal against the first arm's output (HiFi4 by default), and rel-RMS against float64.
"""
import argparse, json, statistics, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=16384)
ap.add_argument("--k", type=int, default=4128)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--cfg", action="store_true", help="bfp8 x bfp8 only: ttnn's default program vs opm_contract_config, "
                "HiFi2 / LoFi, bf16 / fp32 dest")
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402

gen = torch.Generator().manual_seed(0)
dev = T.get_device()
x_h = torch.randn(a.m, a.k, generator=gen).bfloat16()
y_h = torch.randn(a.m, a.k, generator=gen).bfloat16()
up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
F = ttnn.MathFidelity
B8, B16 = ttnn.bfloat8_b, ttnn.bfloat16

CFG = T.opm_contract_config(a.m // 32, a.m // 32, a.k // 32, dev.compute_with_storage_grid_size())
ARMS = ([(cfg, fid, f32) for cfg in (None, CFG) for fid in (F.HiFi2, F.LoFi) for f32 in (False, True)] if a.cfg
        else [(None, fid, False) for fid in (F.HiFi4, F.HiFi3, F.HiFi2, F.LoFi)])
for xd, yd in ((B8, B8),) if a.cfg else ((B8, B8), (B16, B8), (B8, B16)):
    x, y = up(x_h, xd), up(y_h, yd)
    # f64 reference of the values the device holds (bfp8 rounding included).
    xr, yr = ttnn.to_torch(x).double(), ttnn.to_torch(y).double()
    rows = slice(0, 1024)
    ref = xr[rows] @ yr.T
    outs = {}
    for cfg, fid, f32 in ARMS:
        ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=f32,
                                               packer_l1_acc=True)
        ms = []
        try:
            ttnn.deallocate(ttnn.matmul(x, y, transpose_b=True, dtype=B16, program_config=cfg, compute_kernel_config=ckc))
        except RuntimeError as exc:
            print(json.dumps({"fid": str(fid), "cfg": cfg is not None, "fp32_dest": f32, "refused": str(exc)[:160]}),
                  flush=True)
            continue
        for i in range(2 + a.reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            z = ttnn.matmul(x, y, transpose_b=True, dtype=B16, program_config=cfg, compute_kernel_config=ckc)
            ttnn.synchronize_device(dev)
            if i >= 2:
                ms.append((time.perf_counter() - t0) * 1e3)
            if i < 1 + a.reps:
                ttnn.deallocate(z)
        zt = ttnn.to_torch(z)
        ttnn.deallocate(z)
        outs.setdefault("first", zt)
        d = zt[rows].double()
        rel = ((d - ref).pow(2).mean() / ref.pow(2).mean()).sqrt().item()
        print(json.dumps({"x": str(xd), "y": str(yd), "fid": str(fid), "cfg": cfg is not None, "fp32_dest": f32, "ms": round(statistics.median(ms), 3),
                          "equal_first": bool(torch.equal(zt, outs["first"])), "rel_rms_f64": round(rel, 6)}),
              flush=True)
    ttnn.deallocate(x)
    ttnn.deallocate(y)
