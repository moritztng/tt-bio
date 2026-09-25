#!/usr/bin/env python3
"""D55's surviving line: does a compute kernel config on `inner = sum(g*y)` move the gradient?

`tt_bio/autograd.py:softmax_bw_inner` threads its `config` to the row-sum DENOMINATOR and not
to the numerator reduction the near-cancellation `dx = y*(g - inner)` is built on. This scores
the numerator's config against a float64 reference on the SAME operand values the card saw,
and the reference's rule is validated by central finite differences first, in the same run.

Three layers, cheapest first:
  fd      central differences on a small float64 softmax, against the analytic VJP. Validates
          the RULE. If this does not pass nothing below it may be read.
  inner   the reduction alone and the whole `dx`, arms ship / fix / fix+fp32 product.
  triatt  `autograd.triangle_attention`'s own backward, dq/dk/dv/dbias, against a float64
          torch autograd reference. The op the trunk actually runs.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio  # noqa: E402
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT), (
    f"tt_bio came from {_tt_bio.__file__}, not {_ROOT}")

import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import autograd as ag  # noqa: E402
from tt_bio.autograd import precise_config  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


# ---------------------------------------------------------------- 1. validate the rule


def fd_check(seed=0, n=12, rows=6, h=1e-5):
    """Central differences on `L = sum(g * softmax(s))`, float64, against `y*(g - sum(g y))`.

    The rule the card evaluates is the renormalised one, `y*(g - sum(g y)/sum(y))`. On an
    exact softmax sum(y) is 1 and the two coincide; the renorm exists for a y the card
    produced that does not sum to one, which is a separate (already decided, D116) question.
    So what needs validating here is the rule at sum(y)=1, and that is what this does.
    """
    torch.manual_seed(seed)
    s = torch.randn(rows, n, dtype=torch.float64) * 2.0
    g = torch.randn(rows, n, dtype=torch.float64)
    y = torch.softmax(s, dim=-1)
    analytic = y * (g - (g * y).sum(-1, keepdim=True) / y.sum(-1, keepdim=True))
    fd = torch.zeros_like(s)
    for i in range(rows):
        for j in range(n):
            sp = s.clone(); sp[i, j] += h
            sm = s.clone(); sm[i, j] -= h
            fd[i, j] = ((g[i] * torch.softmax(sp[i], -1)).sum()
                        - (g[i] * torch.softmax(sm[i], -1)).sum()) / (2 * h)
    return dict(h=h, shape=[rows, n], rel_l2=rel_l2(analytic, fd), cos=cos(analytic, fd),
                max_abs_diff=float((analytic - fd).abs().max()),
                ref_max_abs=float(fd.abs().max()))


# ---------------------------------------------------------------- 2. the reduction itself

PRECISE = None  # filled at run time (needs ttnn enums, cheap but keep it explicit)


def _inner_arms(dev, y_tt, g_tt, cfg):
    """`inner` as shipped, as fixed, and with an fp32 product as the upper bound.

    The third arm exists because a compute kernel config sets the ACCUMULATOR, not the
    output dtype of `ttnn.multiply`: `g_j*y_j` is rounded to bf16 before the sum sees it
    either way. If ship and fix agree and fix32 does not, the config is not the defect.
    """
    out = {}
    # SHIP: no config on the product, no config on the numerator sum, precise denominator.
    p_ship = ttnn.multiply(g_tt, y_tt)
    out["ship"] = ttnn.divide(ttnn.sum(p_ship, dim=-1, keepdim=True),
                              ttnn.sum(y_tt, dim=-1, keepdim=True, compute_kernel_config=cfg))
    # FIX: the one-line change -- `compute_kernel_config=config or precise_config()` on the
    # numerator reduction too.
    out["fix"] = ttnn.divide(ttnn.sum(p_ship, dim=-1, keepdim=True, compute_kernel_config=cfg),
                             ttnn.sum(y_tt, dim=-1, keepdim=True, compute_kernel_config=cfg))
    # FIX32: same, and the product kept in fp32 so the summand is not pre-rounded.
    try:
        p32 = ttnn.multiply(g_tt, y_tt, dtype=ttnn.float32, compute_kernel_config=cfg)
    except TypeError:
        p32 = ttnn.multiply(g_tt, y_tt, dtype=ttnn.float32)
    out["fix32"] = ttnn.divide(ttnn.sum(p32, dim=-1, keepdim=True, compute_kernel_config=cfg),
                               ttnn.sum(y_tt, dim=-1, keepdim=True, compute_kernel_config=cfg))
    ttnn.deallocate(p32)
    return out


def inner_case(dev, shape, std, seed, dtype=ttnn.bfloat16):
    """One reduction case on operands shaped and built the way the trunk builds them.

    `p` is an EXACT softmax rounded to the device dtype, which is what the shipped default
    hands `triangle_attention`'s backward (`_exact_softmax_raw`). `dp` comes off a device
    matmul, so its magnitudes and its sign structure are the ones the real backward sees,
    not a standard normal.
    """
    torch.manual_seed(seed)
    B, H, nq, nk = shape
    hd = 64
    s = (torch.randn(B, H, nq, nk) * std)
    # exactly `host_f64_softmax_values`: float64 softmax, stored back at device precision
    y64 = torch.softmax(s.to(torch.bfloat16).double(), dim=-1)
    y_tt = ttnn.from_torch(y64.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=dev)
    go = ttnn.from_torch(torch.randn(B, H, nq, hd).to(torch.bfloat16), dtype=dtype,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    # already transposed, so the probe needs no transpose flag and no permute
    v = ttnn.from_torch((torch.randn(B, H, hd, nk) * 0.5).to(torch.bfloat16), dtype=dtype,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = precise_config()
    g_tt = ttnn.matmul(go, v, compute_kernel_config=cfg)
    ttnn.deallocate(go); ttnn.deallocate(v)

    # the reference is float64 ON THE VALUES THE CARD HOLDS, for both operands
    y = ttnn.to_torch(y_tt).double()
    g = ttnn.to_torch(g_tt).double()
    inner_ref = (g * y).sum(-1, keepdim=True) / y.sum(-1, keepdim=True)
    dx_ref = y * (g - inner_ref)

    arms = _inner_arms(dev, y_tt, g_tt, cfg)
    res = {}
    for name, i_tt in arms.items():
        i = ttnn.to_torch(i_tt).double()
        dx = ttnn.to_torch(ttnn.multiply(y_tt, ttnn.subtract(g_tt, i_tt))).double()
        res[name] = dict(
            inner_rel_l2=rel_l2(i, inner_ref),
            inner_max_abs_err=float((i - inner_ref).abs().max()),
            dx_rel_l2=rel_l2(dx, dx_ref),
            dx_cos=cos(dx, dx_ref),
            dx_rowsum_rms=float(dx.sum(-1).pow(2).mean().sqrt()),
        )
        ttnn.deallocate(i_tt)
    # how tight the cancellation actually is: |g - inner| against |g|
    spread = (g - inner_ref).abs()
    res["_cancellation"] = dict(
        mean_abs_g=float(g.abs().mean()),
        mean_abs_g_minus_inner=float(spread.mean()),
        tightness_mean=float(spread.mean() / g.abs().mean()),
        p01_tightness=float(torch.quantile(
            (spread / g.abs().clamp_min(1e-30)).flatten().float()[:4_000_000], 0.01)),
        inner_abs_mean=float(inner_ref.abs().mean()),
        dx_ref_rowsum_rms=float(dx_ref.sum(-1).pow(2).mean().sqrt()),
    )
    ttnn.deallocate(y_tt); ttnn.deallocate(g_tt)
    return dict(shape=list(shape), std=std, seed=seed, arms=res)


# ---------------------------------------------------------------- 3. the op the trunk runs


def triatt_case(dev, B, H, n, hd, seed, fix):
    """`autograd.triangle_attention`'s backward, both arms, vs a float64 autograd reference.

    `fix` is applied by MONKEYPATCHING `softmax_bw_inner` rather than by editing the file,
    so both arms run in one process against one device open and the only difference between
    them is the config on that reduction.
    """
    torch.manual_seed(seed)
    q0 = (torch.randn(B, H, n, hd) * 0.5).to(torch.bfloat16)
    k0 = (torch.randn(B, H, n, hd) * 0.5).to(torch.bfloat16)
    v0 = (torch.randn(B, H, n, hd) * 0.5).to(torch.bfloat16)
    b0 = (torch.randn(1, H, n, n) * 0.5).to(torch.bfloat16)
    g0 = torch.randn(B, H, n, hd).to(torch.bfloat16)
    scale = hd ** -0.5

    # float64 reference on exactly these values
    qd = q0.double().requires_grad_(True); kd = k0.double().requires_grad_(True)
    vd = v0.double().requires_grad_(True); bd = b0.double().requires_grad_(True)
    o = torch.softmax((qd @ kd.transpose(-2, -1)) * scale + bd, dim=-1) @ vd
    o.backward(g0.double())
    ref = dict(q=qd.grad, k=kd.grad, v=vd.grad, bias=bd.grad)

    def to_dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    real_inner = ag.softmax_bw_inner

    def patched(y, g, dim=-1, config=None):
        cfg = config or precise_config()
        inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True,
                         compute_kernel_config=cfg)
        if not ag.SOFTMAX_BW_RENORM:
            return inner
        return ttnn.divide(inner, ttnn.sum(y, dim=dim, keepdim=True,
                                           compute_kernel_config=cfg))

    ag.softmax_bw_inner = patched if fix else real_inner
    try:
        with ag.tape():
            qt = ag.Tensor(to_dev(q0), requires_grad=True)
            kt = ag.Tensor(to_dev(k0), requires_grad=True)
            vt = ag.Tensor(to_dev(v0), requires_grad=True)
            bt = ag.Tensor(to_dev(b0), requires_grad=True)
            out = ag.triangle_attention(qt, kt, vt, bias=bt, scale=scale)
            ag.backward(out, to_dev(g0))
        got = {}
        for name, t in (("q", qt), ("k", kt), ("v", vt), ("bias", bt)):
            got[name] = ttnn.to_torch(t.grad).double()
    finally:
        ag.softmax_bw_inner = real_inner
    return {n_: dict(rel_l2=rel_l2(got[n_], ref[n_]), cos=cos(got[n_], ref[n_]),
                     norm_ratio=float(torch.linalg.vector_norm(got[n_])
                                      / torch.linalg.vector_norm(ref[n_])))
            for n_ in got}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_innercfg/VJP.json")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--skip-device", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    rep = dict(
        generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=platform.node(),
        commit=subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        renorm_flag=bool(ag.SOFTMAX_BW_RENORM),
        fused_flag=bool(ag.SOFTMAX_BW_FUSED),
    )
    rep["fd_validates_the_rule"] = fd_check()
    print("FD:", json.dumps(rep["fd_validates_the_rule"]))

    if not a.skip_device:
        dev = get_device()
        rep["inner"] = []
        for shape, std in (([8, 4, 384, 384], 3.0), ([8, 4, 384, 384], 1.0),
                           ([8, 4, 128, 128], 3.0), ([2, 16, 384, 384], 3.0)):
            c = inner_case(dev, shape, std, seed=17)
            rep["inner"].append(c)
            print("INNER", shape, std, json.dumps(c["arms"]))
        rep["triatt"] = {}
        for fix in (False, True):
            rep["triatt"]["fix" if fix else "ship"] = triatt_case(
                dev, B=4, H=4, n=256, hd=64, seed=11, fix=fix)
            print("TRIATT", "fix" if fix else "ship",
                  json.dumps(rep["triatt"]["fix" if fix else "ship"]))
    rep["seconds"] = round(time.time() - t0, 1)
    p = pathlib.Path(_ROOT) / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=1))
    print("wrote", p, rep["seconds"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
