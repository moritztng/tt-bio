#!/usr/bin/env python3
"""D0 + D1: the landed floor executed, on the input that produced the NaN, forward and backward.

`of3t-orchestrator` wrote the floor without a card. Three things were therefore unverified:
whether `ttnn.clamp(d, -60.0, None)` runs at all, whether the SHIPPED function reproduces the
`accurate_clamp60` reading that justified it, and whether the BACKWARD past a fully-masked row is
finite -- the sum there is ~1e-24 and a reciprocal squared is ~8e47, past fp32's 3.4e38.

The input is rank5b.py's verbatim: same seed, same shape, same seven fully-masked blocks, so the
numbers are comparable to `perf/of3t_softgrad/FULLY_MASKED_ROW_OVERFLOW.json` entry for entry.

Everything is scored against a float64 softmax and its float64 analytic backward on the host, not
against another device arm.
"""
import json
import os
import sys

import torch
import ttnn

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, W)
from tt_bio.tenstorrent import (get_device, _accurate_softmax,  # noqa: E402
                                _SOFTMAX_PRECISE_CKC)

OUT = os.path.join(W, "perf", "of3t_nanfloor", "FLOOR_EXECUTED.json")
dev = get_device()
rep = {"what": "D0 the landed form executes; D1 the floor forward and backward on the "
               "fully-masked input, float64 reference",
       "input": "rank5b.py verbatim: seed 0, [1,14,4,32,128] fp32, blocks 7..13 at -1e9"}


def t2d(y):
    return ttnn.to_torch(y).double()


def stats(t, ref):
    fin = bool(torch.isfinite(t).all())
    return {"finite": fin, "n_nonfinite": int((~torch.isfinite(t)).sum()),
            "rel_l2": (float((t - ref).norm() / ref.norm()) if fin else None)}


# ---------------------------------------------------------------- D0: does it run at all
probe_t = torch.randn(1, 1, 32, 128) * 3.0
probe = ttnn.from_torch(probe_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
d0 = {}
for name, fn in (
        ("clamp_lower_only", lambda: ttnn.clamp(probe, -60.0, None)),
        ("accurate_softmax_positional_ckc", lambda: _accurate_softmax(probe, _SOFTMAX_PRECISE_CKC)),
        ("accurate_softmax_positional_none", lambda: _accurate_softmax(probe, None)),
        ("accurate_softmax_bf16_in", lambda: _accurate_softmax(
            ttnn.from_torch(probe_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16),
            _SOFTMAX_PRECISE_CKC))):
    try:
        y = fn()
        d0[name] = {"ok": True, "error": None, "dtype": str(y.dtype)}
        ttnn.deallocate(y)
    except Exception as e:                                                     # noqa: BLE001
        d0[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
    print("D0", name, json.dumps(d0[name]), flush=True)
rep["d0"] = d0

# ---------------------------------------------------------------- the adversarial input
torch.manual_seed(0)
shape = [1, 14, 4, 32, 128]
x = torch.randn(*shape) * 3.0
x[:, 7:, :, :, :] -= 1e9
xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
ref = torch.softmax(x.double(), dim=-1)

# the row sum the backward has to survive, measured rather than argued
_m = ttnn.max(xt, dim=-1, keepdim=True)
_d = ttnn.subtract(xt, _m)
_dc = ttnn.clamp(_d, -60.0, None)
ttnn.exp(_dc, output_tensor=_dc)
_s = ttnn.sum(_dc, dim=-1, keepdim=True, compute_kernel_config=_SOFTMAX_PRECISE_CKC)
s_h = ttnn.to_torch(_s).double()
rep["row_sum_after_floor"] = {
    "masked_rows_min": float(s_h[:, 7:].min()), "masked_rows_max": float(s_h[:, 7:].max()),
    "live_rows_min": float(s_h[:, :7].min()), "live_rows_max": float(s_h[:, :7].max())}
print("row sums", json.dumps(rep["row_sum_after_floor"]), flush=True)

# ---------------------------------------------------------------- D1 forward
fwd = {}
fwd["shipped_accurate_ckc"] = stats(t2d(_accurate_softmax(xt, _SOFTMAX_PRECISE_CKC)), ref)
fwd["shipped_accurate_nockc"] = stats(t2d(_accurate_softmax(xt, None)), ref)


def measured_form(clamp):
    """rank5b.py's arm: clamps the TOP at 0 as well as the bottom. Not what shipped."""
    m = ttnn.max(xt, dim=-1, keepdim=True)
    d = ttnn.subtract(xt, m)
    d = ttnn.maximum(ttnn.minimum(d, 0.0), float(clamp))
    ttnn.exp(d, output_tensor=d)
    s = ttnn.sum(d, dim=-1, keepdim=True, compute_kernel_config=_SOFTMAX_PRECISE_CKC)
    return ttnn.divide(d, s)


def sum_floor(tiny):
    """The alternative argued in the orchestrator's commit and never run: floor the SUM."""
    m = ttnn.max(xt, dim=-1, keepdim=True)
    d = ttnn.subtract(xt, m)
    ttnn.exp(d, output_tensor=d)
    s = ttnn.sum(d, dim=-1, keepdim=True, compute_kernel_config=_SOFTMAX_PRECISE_CKC)
    s = ttnn.clamp(s, float(tiny), None)
    return ttnn.divide(d, s), s


fwd["measured_form_clamp60"] = stats(t2d(measured_form(-60.0)), ref)
p_sf, s_sf = sum_floor(1e-18)
fwd["sum_floor_1e-18"] = stats(t2d(p_sf), ref)
rep["d1_forward"] = fwd
for k, v in fwd.items():
    print("D1fwd", k, json.dumps(v), flush=True)

# ---------------------------------------------------------------- D1 backward
torch.manual_seed(1)
g = torch.randn(*shape)
gt = ttnn.from_torch(g, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
y64 = ref
dx64 = y64 * (g.double() - (g.double() * y64).sum(-1, keepdim=True))

bw = {}
# (a) the arm's backward: analytic, y only, no division by the row sum anywhere.
y_dev = _accurate_softmax(xt, _SOFTMAX_PRECISE_CKC)
inner = ttnn.sum(ttnn.multiply(gt, y_dev), dim=-1, keepdim=True,
                 compute_kernel_config=_SOFTMAX_PRECISE_CKC)
dx_an = ttnn.multiply(y_dev, ttnn.subtract(gt, inner))
bw["analytic_arm_rule"] = stats(t2d(dx_an), dx64)

# (b) differentiating the CHAIN, which is what a tape over the five ops would do.
#     Two op orders, because 1/s**2 formed as a product of two reciprocals overflows and the
#     same quantity formed by dividing twice does not.
for order in ("rs_squared", "divide_twice"):
    m = ttnn.max(xt, dim=-1, keepdim=True)
    d = ttnn.subtract(xt, m)
    u = ttnn.clamp(d, -60.0, None)
    e = ttnn.exp(u)
    s = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=_SOFTMAX_PRECISE_CKC)
    gd = ttnn.sum(ttnn.multiply(gt, e), dim=-1, keepdim=True,
                  compute_kernel_config=_SOFTMAX_PRECISE_CKC)
    step = {}
    if order == "rs_squared":
        rs = ttnn.reciprocal(s)
        rs2 = ttnn.multiply(rs, rs)
        step["reciprocal_s_max"] = float(ttnn.to_torch(rs).double().abs().max())
        step["reciprocal_s_squared_finite"] = bool(
            torch.isfinite(ttnn.to_torch(rs2)).all())
        de = ttnn.subtract(ttnn.multiply(gt, rs), ttnn.multiply(gd, rs2))
    else:
        de = ttnn.subtract(ttnn.divide(gt, s), ttnn.divide(ttnn.divide(gd, s), s))
    du = ttnn.multiply(de, e)
    # clamp backward: the floor kills the gradient where it bound
    dx_ch = ttnn.multiply(du, ttnn.gt(d, -60.0))
    step.update(stats(t2d(dx_ch), dx64))
    bw[f"chain_differentiated_{order}"] = step

rep["d1_backward"] = bw
for k, v in bw.items():
    print("D1bw", k, json.dumps(v), flush=True)

json.dump(rep, open(OUT, "w"), indent=1, sort_keys=True)
print("wrote", OUT, flush=True)
