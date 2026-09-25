#!/usr/bin/env python3
"""D1, the parts the first pass left open.

  - is the chain's `ttnn.sum` config inert at this shape, to the BIT rather than to a rel_l2;
  - does the floor touch a row that has a live maximum, to the BIT;
  - does flooring the SUM instead fix the BACKWARD too, and what it costs on the forward;
  - the immunity claim measured directly: a bf16-valued input has no overshoot to floor.

Same input as floor.py, same float64 reference.
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

OUT = os.path.join(W, "perf", "of3t_nanfloor", "FLOOR_ALTERNATIVES.json")
dev = get_device()
rep = {}

torch.manual_seed(0)
shape = [1, 14, 4, 32, 128]
x = torch.randn(*shape) * 3.0
x[:, 7:, :, :, :] -= 1e9
xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
ref = torch.softmax(x.double(), dim=-1)
torch.manual_seed(1)
g = torch.randn(*shape)
gt = ttnn.from_torch(g, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
dx64 = ref * (g.double() - (g.double() * ref).sum(-1, keepdim=True))


def h(y):
    return ttnn.to_torch(y).float()


def bits(t):
    return t.contiguous().numpy().tobytes()


def split_rel(t, r):
    t, r = t.double(), r.double()
    out = {"rel_l2_all": float((t - r).norm() / r.norm()),
           "rel_l2_live_rows": float((t - r)[:, :7].norm() / r[:, :7].norm()),
           "rel_l2_masked_rows": float((t - r)[:, 7:].norm() / r[:, 7:].norm()),
           "finite": bool(torch.isfinite(t).all()),
           "n_nonfinite": int((~torch.isfinite(t)).sum())}
    return out


# ---- is the chain's reduction config inert here, to the bit -------------------------------
a = h(_accurate_softmax(xt, _SOFTMAX_PRECISE_CKC))
b = h(_accurate_softmax(xt, None))
rep["sum_ckc_bit_identity_on_this_input"] = {
    "bit_identical": bits(a) == bits(b),
    "max_abs_diff": float((a - b).abs().max()),
    "note": "the two _fp32_softmax_tail call sites pass sm_ckc, which is None unless "
            "TT_BIO_SOFTMAX_CKC is set; this is what that argument buys on this shape"}
print("ckc", json.dumps(rep["sum_ckc_bit_identity_on_this_input"]), flush=True)


def chain(floor=None, sfloor=None, clamp_top=False):
    m = ttnn.max(xt, dim=-1, keepdim=True)
    d = ttnn.subtract(xt, m)
    if clamp_top:
        d = ttnn.minimum(d, 0.0)
    if floor is not None:
        d = ttnn.clamp(d, float(floor), None)
    e = ttnn.exp(d)
    s = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=_SOFTMAX_PRECISE_CKC)
    if sfloor is not None:
        s = ttnn.clamp(s, float(sfloor), None)
    return ttnn.divide(e, s), e, s, d


# ---- the floor against the unfloored chain, on the rows that have a live maximum ----------
p_fl, _, _, _ = chain(floor=-60.0)
p_un, _, _, _ = chain(floor=None)
A, B = h(p_fl), h(p_un)
rep["floor_vs_unfloored"] = {
    "live_rows_bit_identical": bits(A[:, :7]) == bits(B[:, :7]),
    "live_rows_max_abs_diff": float((A[:, :7] - B[:, :7]).abs().max()),
    "masked_rows_unfloored_nonfinite": int((~torch.isfinite(B[:, 7:])).sum()),
    "masked_rows_floored_nonfinite": int((~torch.isfinite(A[:, 7:])).sum())}
print("floorvsun", json.dumps(rep["floor_vs_unfloored"]), flush=True)

# ---- the SUM-floor alternative: forward, live-row identity, and the backward --------------
alt = {}
for tiny in (1e-18, 1e-20, 1e-30):
    p, e, s, d = chain(floor=None, sfloor=tiny)
    P = h(p)
    rec = split_rel(P, ref)
    rec["live_rows_bit_identical_to_unfloored"] = bits(P[:, :7]) == bits(B[:, :7])
    # the backward a tape over this chain would run, in the safe op order
    gd = ttnn.sum(ttnn.multiply(gt, e), dim=-1, keepdim=True,
                  compute_kernel_config=_SOFTMAX_PRECISE_CKC)
    de = ttnn.subtract(ttnn.divide(gt, s), ttnn.divide(ttnn.divide(gd, s), s))
    dx = ttnn.multiply(de, e)
    DX = h(dx)
    rec["backward_divide_twice"] = {"finite": bool(torch.isfinite(DX).all()),
                                    "n_nonfinite": int((~torch.isfinite(DX)).sum())}
    # and the order that overflowed with the exponent floor
    rs = ttnn.reciprocal(s)
    rs2 = ttnn.multiply(rs, rs)
    de2 = ttnn.subtract(ttnn.multiply(gt, rs), ttnn.multiply(gd, rs2))
    DX2 = h(ttnn.multiply(de2, e))
    rec["backward_rs_squared"] = {"finite": bool(torch.isfinite(DX2).all()),
                                  "n_nonfinite": int((~torch.isfinite(DX2)).sum()),
                                  "reciprocal_s_max": float(h(rs).double().abs().max())}
    # the analytic backward the arm rule actually installs, on this forward
    inner = ttnn.sum(ttnn.multiply(gt, p), dim=-1, keepdim=True,
                     compute_kernel_config=_SOFTMAX_PRECISE_CKC)
    DXA = h(ttnn.multiply(p, ttnn.subtract(gt, inner)))
    rec["backward_analytic"] = {"finite": bool(torch.isfinite(DXA).all()),
                                "n_nonfinite": int((~torch.isfinite(DXA)).sum()),
                                "rel_l2": float((DXA.double() - dx64).norm() / dx64.norm())}
    alt[f"sum_floor_{tiny:g}"] = rec
    print("sumfloor", tiny, json.dumps(rec), flush=True)
rep["sum_floor_alternative"] = alt

# ---- the immunity claim, measured: a bf16-valued input has nothing to overshoot -----------
xb = x.to(torch.bfloat16).to(torch.float32)          # exactly the bf16 grid, stored fp32
xbt = ttnn.from_torch(xb, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
mb = ttnn.to_torch(ttnn.max(xbt, dim=-1, keepdim=True)).double()
wantb = xb.double().amax(-1, keepdim=True)
yb = h(_accurate_softmax(xbt, _SOFTMAX_PRECISE_CKC))
refb = torch.softmax(xb.double(), dim=-1)
rep["bf16_grid_input_immunity"] = {
    "max_overshoot_over_true_row_max": float((mb - wantb).max()),
    "max_shortfall_under_true_row_max": float((wantb - mb).max()),
    "shipped_accurate_nonfinite": int((~torch.isfinite(yb)).sum()),
    "shipped_accurate_rel_l2": float((yb.double() - refb).norm() / refb.norm())}
print("immunity", json.dumps(rep["bf16_grid_input_immunity"]), flush=True)

json.dump(rep, open(OUT, "w"), indent=1, sort_keys=True)
print("wrote", OUT, flush=True)
