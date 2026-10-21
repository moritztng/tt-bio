#!/usr/bin/env python3
"""OpenFold3's outer product mean at the page cell (36 MSA rows, 512 tokens), as it runs today, against contraction
orders that never build the [I*C, J*D] outer product.

pcen2 (WH --fast fold census, 1000 MHz) prices today's path at 21.7 ms a call, 16 calls a fold: contract 7.7 ms
(0.28 of roof) + opm_relayout 5.5 + proj_o linear 8.5 (0.29), around a 512 MB intermediate. With S = 36 rows the
a-first order (fold proj_o into a's projection, then contract over (s, d)) is fewer flops (77 vs 88 GFLOP) and moves
~0.4 GB instead of ~2 GB, but needs one relayout of a 151 MB tensor that ttnn has no single op for:

  site      OuterProductMean(x) on main: layer_norm + projections + contract + relayout + proj_o
  proj      layer_norm + the two projections alone (subtract from site to compare the contraction)
  aperm     A = a W [(s i), (d z)]; permute to (i, z, s, d) tiled; [(i z), (s d)] @ b^T [(s d), j]; transpose to (i, j, z)
  aperm_rm  the same with the permute in ROW_MAJOR
  bfirst    B = b W [(s j), (c z)]; permute to (s, c, j, z); a^T [i, (s c)] @ B [(s c), (j z)]; view to (i, j, z)
  joined    `_small_depth_joined` (repeat b over i, one batched [J, S*D] x [S*D, c_z] per i)

ms a call (median of --reps after two warm calls, synchronized, untraced: every arm is >= 1 ms), rel. L2 of the
alternative arms against float64 on the same rounded a, b and weights. One chip, random operands and weights.

    TT_VISIBLE_DEVICES=N python perf/spd/of3_opm_probe.py --out r.json
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=36)
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", type=Path, required=True)
a_ = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio.openfold3_fold import WH_FAST_LEVERS

# OF3's shipped Wormhole --fast set, so `site` takes the fold's path (opm_jd and its relayout).
T.levers(WH_FAST_LEVERS).__enter__()

S, I, CM, C, Z = a_.depth, a_.tokens, 64, 32, 128
J, D = I, C
dev = T.get_device()
torch.manual_seed(0)
res = {"card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename, "arch": str(dev.arch()),
       "shape": {"S": S, "I": I, "c_m": CM, "C": C, "c_z": Z}, "loadavg": open("/proc/loadavg").read().split()[:3],
       "cells": []}
# The trunk's config (trunk_compute_kernel_config: HiFi3 + fp32 accumulation), which the MSA module runs under.
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
sd = {"norm.weight": 1 + 0.1 * torch.randn(CM), "norm.bias": 0.1 * torch.randn(CM),
      "proj_a.weight": torch.randn(C, CM) / CM ** 0.5, "proj_b.weight": torch.randn(D, CM) / CM ** 0.5,
      "proj_o.weight": torch.randn(Z, C * D) / (C * D) ** 0.5, "proj_o.bias": 0.1 * torch.randn(Z)}
opm = T.OuterProductMean(sd, ckc, scale_bias=True)
x = ttnn.from_torch(torch.randn(1, S, I, CM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
scale = 1.0 / S
d64 = lambda t: ttnn.to_torch(t).double()  # noqa: E731


def proj():
    """project_ab of the module, op for op."""
    x3 = ttnn.reshape(x, (S, I, CM))
    mc = ttnn.layer_norm(x3, weight=opm.norm_weight, bias=opm.norm_bias, epsilon=1e-5, compute_kernel_config=ckc)
    a = ttnn.linear(mc, opm.a_weight, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
    b = ttnn.linear(mc, opm.b_weight, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
    ttnn.deallocate(mc)
    return a, b


wf = opm._proj_o_folded(C, D)                       # [C, D*Z], row c, column d*Z + z
w_dcz = ttnn.from_torch(d64(opm.o_weight).reshape(C, D, Z).permute(1, 0, 2).reshape(D, C * Z).float(),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)   # [D, C*Z]
bias_s = ttnn.multiply(opm.o_bias, scale)


def aperm(rm=False):
    a, b = proj()
    a2 = ttnn.multiply(ttnn.reshape(a, (S * I, C)), scale)
    ttnn.deallocate(a)
    A = ttnn.matmul(a2, wf, compute_kernel_config=ckc)          # [(s i), (d z)]
    ttnn.deallocate(a2)
    if rm:
        A = ttnn.to_layout(A, ttnn.ROW_MAJOR_LAYOUT)
    A = ttnn.permute(ttnn.reshape(A, (S, I, D, Z)), (1, 3, 0, 2))  # (i, z, s, d)
    A = ttnn.reshape(A, (I * Z, S * D))
    if rm:
        A = ttnn.to_layout(A, ttnn.TILE_LAYOUT)
    bT = ttnn.reshape(ttnn.permute(b, (0, 2, 1)), (S * D, J))  # (s d), j
    ttnn.deallocate(b)
    o = ttnn.matmul(A, bT, compute_kernel_config=ckc)          # [(i z), j]
    ttnn.deallocate(A); ttnn.deallocate(bT)
    o = ttnn.transpose(ttnn.reshape(o, (I, Z, J)), -2, -1)       # (i, j, z)
    return ttnn.add(o, bias_s)


def bfirst():
    a, b = proj()
    B = ttnn.matmul(ttnn.reshape(b, (S * J, D)), w_dcz, compute_kernel_config=ckc)  # [(s j), (c z)]
    ttnn.deallocate(b)
    B = ttnn.permute(ttnn.reshape(B, (S, J, C, Z)), (0, 2, 1, 3))  # (s, c, j, z)
    B = ttnn.reshape(B, (S * C, J * Z))
    aT = ttnn.reshape(ttnn.permute(ttnn.multiply(a, scale), (1, 0, 2)), (I, S * C))  # i, (s c)
    ttnn.deallocate(a)
    o = ttnn.matmul(aT, B, compute_kernel_config=ckc)           # [i, (j z)]
    ttnn.deallocate(aT); ttnn.deallocate(B)
    return ttnn.add(ttnn.reshape(o, (I, J, Z)), bias_s)


def joined():
    a, b = proj()
    o = opm._small_depth_joined(a, b, wf, scale, S, I, C, J, D, Z)
    ttnn.deallocate(a); ttnn.deallocate(b)
    return ttnn.add(o, bias_s)


def site():
    return opm(x, None, None)


def proj_only():
    a, b = proj()
    ttnn.deallocate(b)
    return a


def timed(fn, n):
    for _ in range(2):
        ttnn.deallocate(fn())
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    return round(statistics.median(ts), 3), round(min(ts), 3)


a0, b0 = proj()
A64, B64 = d64(a0).reshape(S, I, C), d64(b0).reshape(S, J, D)
W64 = d64(opm.o_weight)                                   # [C*D, Z], row c*D + d
Af = torch.einsum("sic,cdz->sidz", A64, W64.reshape(C, D, Z))   # a-first in float64: no 2 GB outer product
ref = (torch.einsum("sidz,sjd->ijz", Af, B64) + d64(opm.o_bias).reshape(1, 1, Z)) * scale
del Af
ttnn.deallocate(a0); ttnn.deallocate(b0)

for arm, fn in (("site", site), ("proj", proj_only), ("aperm", aperm), ("aperm_rm", lambda: aperm(True)),
                ("bfirst", bfirst), ("joined", joined)):
    c = {"arm": arm, "calls_per_fold": 16}
    try:
        o = fn()
        if arm != "proj":
            got = d64(o).reshape(I, J, Z)
            c["rel_l2"] = float((got - ref).norm() / ref.norm())
        ttnn.deallocate(o)
        c["ms"], c["ms_min"] = timed(fn, a_.reps)
    except Exception as e:  # noqa: BLE001
        c["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["cells"].append(c)
    print(json.dumps(c), flush=True)
    a_.out.write_text(json.dumps(res, indent=1))
res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
res["done"] = time.strftime("%FT%TZ", time.gmtime())
a_.out.write_text(json.dumps(res, indent=1))
T.cleanup()
