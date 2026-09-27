#!/usr/bin/env python3
"""of3t-p10grad: OF3T's SHIPPED backward, op by op, against a float64 reference.

The reference is validated by central finite differences before any op is graded, and the bar
is `perf/of3t_p10grad/PREREGISTRATION.md`, committed before this file ran once.

    python3 perf/of3t_p10grad/opgrad.py --out PEROP.json [--draws 32] [--ops layer_norm,softmax]

What is compared, precisely: the device holds inputs in bfloat16 or float32; those inputs are
read BACK to the host and used as the reference's inputs, so the input rounding is not charged
to the op. What is left is the op's own arithmetic.

Each case names the VERB chain it drives, because "the layer norm backward" is not one thing:
`ttnn.moreh_layer_norm_backward` is a kernel in the wheel, and OF3T does not call it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())

import torch                                                          # noqa: E402
import ttnn                                                           # noqa: E402
import tt_bio                                                         # noqa: E402
from tt_bio import autograd as ag                                     # noqa: E402
from tt_bio import taped_ttnn as tp                                   # noqa: E402
from tt_bio.tenstorrent import get_device                             # noqa: E402

# The shared env carries an EDITABLE install of tt_bio pointed at /home/ttuser/tt-bio-dev, which
# sits on main. An arm that measures main while running from this worktree announces it only in a
# config field nobody re-reads. (`of3t-vjpln`'s D190 trap, paid for once already.)
_WT = os.path.realpath(os.getcwd())
if not os.path.realpath(tt_bio.__file__).startswith(_WT):
    raise SystemExit(f"tt_bio resolved to {tt_bio.__file__}, not this worktree {_WT}")

DTYPES = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}

# --- the bar, read out of the pre-registration rather than restated loosely ------------------
BAR = {"float32": {"pass": 1.0e-3, "wrong": 1.0e-1},
       "bfloat16": {"pass": 3.0e-2, "wrong": 1.0}}
FD_BAR = 1.0e-6
BIAS_SIGMA = 5.0
BIAS_FLOOR = 1.0e-3


def grade(rel, dtype):
    b = BAR[dtype]
    if not math.isfinite(rel):
        return "WRONG"
    if rel <= b["pass"]:
        return "PASS"
    if rel >= b["wrong"]:
        return "WRONG"
    return "MARGINAL"


def relL2(a, b):
    n = torch.linalg.vector_norm(b).item()
    return torch.linalg.vector_norm(a - b).item() / n if n > 0 else float("nan")


dev = None


def to_card(t, dtype):
    return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def back(v):
    return ttnn.to_torch(v)


# ---------------------------------------------------------------------------------------------
# The cases. `host` is the mathematics in float64; `device` drives the SHIPPED taped verb.
# ---------------------------------------------------------------------------------------------

EPS = 1e-5          # every layer norm on tt-bio's inference path uses 1e-5


def _ln_host(ins, eps=EPS):
    x, gamma, beta = ins["x"], ins["gamma"], ins["beta"]
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma + beta


def _ln_device(dv, g):
    x = ag.Tensor(dv["x"], requires_grad=True)
    gamma = ag.Tensor(dv["gamma"], requires_grad=True)
    beta = ag.Tensor(dv["beta"], requires_grad=True)
    out = ag._taped_layer_norm(ttnn.layer_norm, (x,),
                               {"weight": gamma, "bias": beta, "epsilon": EPS,
                                "compute_kernel_config": ag.precise_config()})
    out.backward(seed=g)
    return {"dx": x.grad, "dgamma": gamma.grad, "dbeta": beta.grad}, out.value


def _ln_diag(mean_cfg, rsqrt_split, sum_scale=False):
    """DIAGNOSTIC, not shipped: the shipped layer-norm backward closure with one knob moved.

    `autograd._taped_layer_norm`'s closure passes `precise_config()` to exactly one of its four
    reductions (`var`) and to none of `mean`, `dn_mean`, `dn_norm_mean`; and it takes `rstd`
    from `ttnn.rsqrt`, whose approximation lands on `dx` as a pure per-row SCALE because `dx`
    ends in `multiply(dx, rstd)`. One arm per knob.
    """
    def impl(dv, g):
        cfg = ag.precise_config()
        xv, gamma, beta = dv["x"], dv["gamma"], dv["beta"]
        mk = {"compute_kernel_config": cfg} if mean_cfg else {}
        K = int(xv.shape[-1])

        def _mean(v):
            # `sum_scale`: the reduction, then an explicit multiply by 1/K, instead of
            # `ttnn.mean`. 1/128 is a power of two and exact in every float; 1/384 is not,
            # and if `ttnn.mean` carries its 1/K in bf16 the constant is 9.8e-4 low. Every
            # term of the closure that is a mean then inherits the same signed constant,
            # which is what a K-dependent systematic bias looks like.
            if not sum_scale:
                return ttnn.mean(v, dim=-1, keepdim=True, **mk)
            return ttnn.multiply(ttnn.sum(v, dim=-1, keepdim=True, compute_kernel_config=cfg),
                                 1.0 / K)

        mean = _mean(xv)
        centered = ttnn.subtract(xv, mean)
        var = (_mean(ttnn.multiply(centered, centered)) if sum_scale else
               ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                         compute_kernel_config=cfg))
        vpe = ttnn.add(var, EPS)
        rstd = ttnn.reciprocal(ttnn.sqrt(vpe)) if rsqrt_split else ttnn.rsqrt(vpe)
        norm = ttnn.multiply(centered, rstd)
        dgamma = ttnn.sum(ttnn.multiply(g, norm), dim=list(range(len(g.shape) - 1)),
                          compute_kernel_config=cfg)
        dbeta = ttnn.sum(g, dim=list(range(len(g.shape) - 1)), compute_kernel_config=cfg)
        dnorm = ttnn.multiply(g, gamma)
        dn_mean = _mean(dnorm)
        dn_norm_mean = _mean(ttnn.multiply(dnorm, norm))
        dx = ttnn.subtract(ttnn.subtract(dnorm, dn_mean), ttnn.multiply(norm, dn_norm_mean))
        out = ttnn.layer_norm(xv, weight=gamma, bias=beta, epsilon=EPS,
                              compute_kernel_config=cfg)
        return {"dx": ttnn.multiply(dx, rstd), "dgamma": dgamma, "dbeta": dbeta}, out
    return impl


def _sm_host(ins):
    return torch.softmax(ins["x"], dim=-1)


def _sm_device(dv, g):
    x = ag.Tensor(dv["x"], requires_grad=True)
    out = tp._v_softmax(ttnn.softmax, (x,), {"dim": -1})
    out.backward(seed=g)
    return {"dx": x.grad}, out.value


def _sm_host_bw(ins, y, cot):
    """The SHIPPED backward expression, evaluated in float64 on the device's own y.

    `y (g - sum(g y) / sum(y))`, renorm included, because that is the expression
    `autograd.softmax_bw` writes -- not the textbook `y (g - sum(g y))`. Grading the closure
    against the textbook form would charge it for a deliberate repair (`of3t-d116`, up to
    13.09x on ||dq||). What is left is the closure's ARITHMETIC.
    """
    inner = (cot * y).sum(dim=-1, keepdim=True) / y.sum(dim=-1, keepdim=True)
    return {"dx": y * (cot - inner)}


def _sm_diag(fwd_cfg, sum_cfg):
    """DIAGNOSTIC, not shipped: the shipped softmax algebra with one knob moved at a time.

    Two knobs, and they have to be separated or neither is measured. `taped_ttnn._v_softmax`
    calls `ttnn.softmax(x, dim)` with NO `compute_kernel_config`, and
    `autograd.softmax_bw_inner:166` runs `ttnn.sum(ttnn.multiply(g, y), dim, keepdim=True)`
    with none either, while the renorm denominator one line below it gets `precise_config()`.
    One arm per knob says which of the two the shipped error lives in.
    """
    def impl(dv, g):
        cfg = ag.precise_config()
        y = (ttnn.softmax(dv["x"], dim=-1, compute_kernel_config=cfg) if fwd_cfg
             else ttnn.softmax(dv["x"], dim=-1))
        num = ttnn.multiply(g, y)
        inner = (ttnn.sum(num, dim=-1, keepdim=True, compute_kernel_config=cfg) if sum_cfg
                 else ttnn.sum(num, dim=-1, keepdim=True))
        inner = ttnn.divide(inner, ttnn.sum(y, dim=-1, keepdim=True, compute_kernel_config=cfg))
        return {"dx": ttnn.multiply(y, ttnn.subtract(g, inner))}, y
    return impl


def _lin_host(ins):
    # `ttnn.linear(x, w)` is `x @ w` with w [in, out] -- NOT torch's `x @ w.T`. Writing the
    # torch convention here graded the op WRONG at rel L2 1.407 in both dtypes, which is
    # sqrt(2), the signature of two uncorrelated vectors of equal norm. The forward rel L2
    # read 1.407 too, which is what caught it: a reference that disagrees with the card on
    # the FORWARD is a broken reference, because the forward is the one thing both sides
    # compute the same way.
    return ins["x"] @ ins["w"] + ins["bias"]


def _lin_device(dv, g):
    x = ag.Tensor(dv["x"], requires_grad=True)
    w = ag.Tensor(dv["w"], requires_grad=True)
    b = ag.Tensor(dv["bias"], requires_grad=True)
    out = ag._taped_linear(ttnn.linear, (x, w), {"bias": b,
                                                 "compute_kernel_config": ag.precise_config()})
    out.backward(seed=g)
    return {"dx": x.grad, "dw": w.grad, "dbias": b.grad}, out.value


def _mm_host(ins):
    return ins["a"] @ ins["b"]


def _mm_device(dv, g):
    a = ag.Tensor(dv["a"], requires_grad=True)
    b = ag.Tensor(dv["b"], requires_grad=True)
    out = tp._v_matmul(ttnn.matmul, (a, b),
                       {"compute_kernel_config": ag.precise_config()})
    out.backward(seed=g)
    return {"da": a.grad, "db": b.grad}, out.value


def _mul_host(ins):
    return ins["a"] * ins["b"]


def _mul_device(dv, g):
    a = ag.Tensor(dv["a"], requires_grad=True)
    b = ag.Tensor(dv["b"], requires_grad=True)
    out = tp._VERBS["multiply"](ttnn.multiply, (a, b), {})
    out.backward(seed=g)
    return {"da": a.grad, "db": b.grad}, out.value


def _sig_host(ins):
    return torch.sigmoid(ins["x"])


def _sig_device(dv, g):
    x = ag.Tensor(dv["x"], requires_grad=True)
    out = tp._VERBS["sigmoid"](ttnn.sigmoid, (x,), {})
    out.backward(seed=g)
    return {"dx": x.grad}, out.value


# The `w` of `ttnn.linear` is [out, in] and the torch reference transposes it, which is what
# `_lin_host` does. `dw` therefore comes back [out, in] from both sides.
CASES = [
    # key                       site in the trunk                    verb chain the backward runs
    dict(key="layer_norm/pair_K128",
         site="TRI_MUL / TRI_ATT / TRANS on z: [1,N,N,128], 816 of the step's 816+240 backwards",
         verb="ttnn.mean, subtract, multiply, mean, rsqrt, add, multiply, sum "
              "(autograd._taped_layer_norm) -- NOT ttnn.moreh_layer_norm_backward",
         shapes=dict(x=(1, 64, 64, 128), gamma=(128,), beta=(128,)),
         host=_ln_host, device=_ln_device, diff="x"),
    dict(key="layer_norm/single_K384",
         site="APB / TRANS on s: [1,N,384]",
         verb="same closure as layer_norm/pair_K128",
         shapes=dict(x=(1, 384, 384), gamma=(384,), beta=(384,)),
         host=_ln_host, device=_ln_device, diff="x"),
    dict(key="layer_norm/single_K384_CONTROL",
         site="DIAGNOSTIC control: the shipped closure rewritten here, both knobs left alone. "
              "Must reproduce layer_norm/single_K384, or the two arms below measure nothing",
         verb="unconfigured ttnn.mean x3 + ttnn.rsqrt -- as shipped",
         shapes=dict(x=(1, 384, 384), gamma=(384,), beta=(384,)),
         host=_ln_host, device=_ln_diag(False, False), diff="x"),
    dict(key="layer_norm/single_K384_MEANCFG",
         site="DIAGNOSTIC: precise_config() on all three unconfigured means, rsqrt as shipped",
         verb="ttnn.mean(precise_config()) x3 + ttnn.rsqrt",
         shapes=dict(x=(1, 384, 384), gamma=(384,), beta=(384,)),
         host=_ln_host, device=_ln_diag(True, False), diff="x"),
    dict(key="layer_norm/single_K384_RSQRTSPLIT",
         site="DIAGNOSTIC: rstd as reciprocal(sqrt(v)) instead of rsqrt(v), means as shipped",
         verb="unconfigured ttnn.mean x3 + ttnn.reciprocal(ttnn.sqrt())",
         shapes=dict(x=(1, 384, 384), gamma=(384,), beta=(384,)),
         host=_ln_host, device=_ln_diag(False, True), diff="x"),
    dict(key="layer_norm/single_K384_SUMSCALE",
         site="DIAGNOSTIC: every mean as sum * (1/K) with the scale an fp32 python float",
         verb="ttnn.sum(precise_config()) then ttnn.multiply by 1/K, instead of ttnn.mean",
         shapes=dict(x=(1, 384, 384), gamma=(384,), beta=(384,)),
         host=_ln_host, device=_ln_diag(False, False, True), diff="x"),
    dict(key="layer_norm/pair_K128_SUMSCALE",
         site="DIAGNOSTIC control on an EXACT 1/K: at K=128 the constant is a power of two, so "
              "this arm must be indistinguishable from the shipped one",
         verb="ttnn.sum(precise_config()) then ttnn.multiply by 1/128",
         shapes=dict(x=(1, 64, 64, 128), gamma=(128,), beta=(128,)),
         host=_ln_host, device=_ln_diag(False, False, True), diff="x"),
    dict(key="softmax/triatt_K64",
         site="TRI_ATT scores [1,heads=4,N,N], 192 of the step's 240 softmax backwards",
         verb="ttnn.multiply, sum, sum, divide, subtract, multiply "
              "(autograd.softmax_bw with TT_BIO_SOFTMAX_BW_RENORM on) "
              "-- NOT ttnn.moreh_softmax_backward, which is off by default",
         shapes=dict(x=(1, 4, 64, 64)),
         host=_sm_host, device=_sm_device, host_bw=_sm_host_bw, diff="x"),
    dict(key="softmax/apb_K64",
         site="AttentionPairBias scores [1,heads=16,N,N], 48 of the 240",
         verb="same closure as softmax/triatt_K64",
         shapes=dict(x=(1, 16, 64, 64)),
         host=_sm_host, device=_sm_device, host_bw=_sm_host_bw, diff="x"),
    dict(key="softmax/triatt_K384",
         site="TRI_ATT at the n384 scope: the longest softmax reduction the trunk runs",
         verb="same closure as softmax/triatt_K64",
         shapes=dict(x=(1, 4, 384, 384)),
         host=_sm_host, device=_sm_device, host_bw=_sm_host_bw, diff="x"),
    dict(key="softmax/triatt_K64_CONTROL",
         site="DIAGNOSTIC control: the shipped algebra rewritten here, both knobs left alone. "
              "Must reproduce softmax/triatt_K64, or the two diagnostic arms measure nothing",
         verb="ttnn.softmax(no cfg) + unconfigured numerator sum -- as shipped",
         shapes=dict(x=(1, 4, 64, 64)),
         host=_sm_host, device=_sm_diag(False, False), host_bw=_sm_host_bw, diff="x"),
    dict(key="softmax/triatt_K64_SUMCFG",
         site="DIAGNOSTIC: only the numerator reduction configured, forward as shipped",
         verb="ttnn.softmax(no cfg) + precise_config() on the numerator sum",
         shapes=dict(x=(1, 4, 64, 64)),
         host=_sm_host, device=_sm_diag(False, True), host_bw=_sm_host_bw, diff="x"),
    dict(key="softmax/triatt_K64_FWDCFG",
         site="DIAGNOSTIC: only the FORWARD configured, both reductions as shipped",
         verb="ttnn.softmax(precise_config()) + unconfigured numerator sum",
         shapes=dict(x=(1, 4, 64, 64)),
         host=_sm_host, device=_sm_diag(True, False), host_bw=_sm_host_bw, diff="x"),
    dict(key="linear/pair_128_128",
         site="the pair-track projections, x [1,N*N,128] w [128,128]",
         verb="ttnn.matmul(g, w, transpose_b) for dx; matmul(x^T, g, dtype=float32) for dw; "
              "sum for dbias (autograd._taped_linear)",
         shapes=dict(x=(1, 4096, 128), w=(128, 128), bias=(128,)),
         host=_lin_host, device=_lin_device, diff="x"),
    dict(key="matmul/pair_4096x128x128",
         site="a bare batched matmul on the pair track",
         verb="ag.bmm both sides (taped_ttnn._v_matmul)",
         shapes=dict(a=(1, 4096, 128), b=(128, 128)),
         host=_mm_host, device=_mm_device, diff="a"),
    dict(key="multiply/pair",
         site="every gate and every residual scale on the pair track",
         verb="ttnn.multiply (taped_ttnn._binary)",
         shapes=dict(a=(1, 64, 64, 128), b=(1, 64, 64, 128)),
         host=_mul_host, device=_mul_device, diff="a"),
    dict(key="sigmoid/pair",
         site="the gates",
         verb="ttnn.sigmoid_bw (taped_ttnn._unary fused)",
         shapes=dict(x=(1, 64, 64, 128)),
         host=_sig_host, device=_sig_device, diff="x"),
]


def draw(case, seed):
    g = torch.Generator().manual_seed(seed)
    ins = {}
    for k, s in case["shapes"].items():
        if k == "gamma":
            # A trained layer-norm scale sits near 1, not near 0. Drawing it N(0,1) would make
            # the reference's own gradient unrepresentative of the one the trunk takes.
            ins[k] = (1.0 + 0.1 * torch.randn(s, generator=g, dtype=torch.float64))
        elif k in ("beta", "bias"):
            ins[k] = 0.1 * torch.randn(s, generator=g, dtype=torch.float64)
        else:
            ins[k] = torch.randn(s, generator=g, dtype=torch.float64)
    return ins


def fd_validate(case, ins, cot, n_probe=4096):
    """Bar 1: is the float64 reference trustworthy.

    Central difference of `L(x) = <cot, f(x)>` along a random direction, against the reference's
    own `<dx, v>`. Restricted to `n_probe` random coordinates of the differentiated input so the
    step stays well-conditioned on a big tensor: a direction spread over 3.1M coordinates makes
    every per-coordinate perturbation `h/sqrt(n)`, which is where the difference stops resolving.
    """
    name = case["diff"]
    x0 = ins[name]
    gen = torch.Generator().manual_seed(9001)
    flat = x0.reshape(-1)
    idx = torch.randperm(flat.numel(), generator=gen)[:min(n_probe, flat.numel())]
    v = torch.zeros_like(flat)
    v[idx] = torch.randn(idx.numel(), generator=gen, dtype=torch.float64)
    v = (v / torch.linalg.vector_norm(v)).reshape(x0.shape)

    req = {k: t.clone().requires_grad_(k == name) for k, t in ins.items()}
    out = case["host"](req)
    loss = (out * cot).sum()
    (dx_ref,) = torch.autograd.grad(loss, req[name])
    analytic = (dx_ref * v).sum().item()

    best = None
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        up = dict(ins); up[name] = x0 + h * v
        dn = dict(ins); dn[name] = x0 - h * v
        num = ((case["host"](up) * cot).sum() - (case["host"](dn) * cot).sum()).item() / (2 * h)
        err = abs(num - analytic) / max(abs(analytic), 1e-300)
        if best is None or err < best[1]:
            best = (h, err, num)
    return {"analytic": analytic, "h": best[0], "fd": best[2], "rel_err": best[1],
            "coords": int(idx.numel()), "validated": bool(best[1] <= FD_BAR)}


def ref_grads(case, ins, cot):
    req = {k: t.clone().requires_grad_(True) for k, t in ins.items()}
    out = case["host"](req)
    loss = (out * cot).sum()
    names = list(req)
    gs = torch.autograd.grad(loss, [req[n] for n in names], allow_unused=True)
    return out.detach(), {n: g for n, g in zip(names, gs) if g is not None}


# The device's gradient names are the op's; the reference's are the INPUT's. One map, so a
# rename on either side is a KeyError rather than a silently skipped gradient.
GRADMAP = {"dx": "x", "dgamma": "gamma", "dbeta": "beta", "dw": "w", "dbias": "bias",
           "da": "a", "db": "b"}


def run_case(case, dtype_name, draws, want_det=True):
    tt = DTYPES[dtype_name]
    rows = {}
    det = None
    fd = None
    fwd = []
    rowsum = []
    for i in range(draws):
        ins = draw(case, 1000 + i)
        dv = {k: to_card(t, tt) for k, t in ins.items()}
        # The reference's inputs are the DEVICE's inputs, read back. Input rounding is not
        # charged to the op.
        exact = {k: back(v).double() for k, v in dv.items()}
        cot_h = torch.randn(tuple(int(d) for d in case["host"](exact).shape),
                            generator=torch.Generator().manual_seed(7000 + i),
                            dtype=torch.float64)
        gdev = to_card(cot_h, tt)
        cot = back(gdev).double()

        if fd is None:
            fd = fd_validate(case, exact, cot)
        _, ref = ref_grads(case, exact, cot)

        grads, out_v = case["device"](dv, gdev)
        got = {k: back(v) for k, v in grads.items() if v is not None}
        y_dev = back(out_v).double()
        fwd.append(relL2(y_dev, case["host"](exact)))
        if case["key"].startswith("softmax"):
            rs = y_dev.sum(dim=-1)
            rowsum.append((rs.mean().item(), rs.min().item(), rs.max().item()))
        # The forward/backward split. A closure that reads its own output carries the
        # forward's error into dx, and the two are different defects with different owners:
        # `ref` grades the op, `refbw` grades the backward closure alone on the y the card
        # actually produced.
        refbw = case["host_bw"](exact, y_dev, cot) if case.get("host_bw") else None

        if want_det and det is None:
            dv2 = {k: to_card(t, tt) for k, t in ins.items()}
            g2 = to_card(cot_h, tt)
            grads2, _ = case["device"](dv2, g2)
            det = {k: bool(torch.equal(got[k], back(grads2[k])))
                   for k in got if grads2.get(k) is not None}

        for k, gv in got.items():
            r = ref[GRADMAP[k]].reshape(gv.shape)
            d = gv.double()
            rows.setdefault(k, {"rel": [], "s": [], "mean_over_rms": [], "rel_bw": []})
            rows[k]["rel"].append(relL2(d, r))
            denom = (r * r).sum().item()
            rows[k]["s"].append(((d - r) * r).sum().item() / denom if denom > 0 else float("nan"))
            rms = math.sqrt((r * r).mean().item())
            rows[k]["mean_over_rms"].append((d - r).mean().item() / rms if rms > 0 else
                                            float("nan"))
            if refbw is not None and k in refbw:
                rows[k]["rel_bw"].append(relL2(d, refbw[k].reshape(gv.shape)))
        for v in list(dv.values()) + [gdev]:
            ttnn.deallocate(v)

    out = {}
    for k, acc in rows.items():
        rel = acc["rel"]
        s = torch.tensor(acc["s"], dtype=torch.float64)
        m = s.mean().item()
        se = (s.std(unbiased=True) / math.sqrt(len(s))).item() if len(s) > 1 else float("nan")
        sigma = abs(m) / se if se and math.isfinite(se) and se > 0 else float("inf")
        bias = bool(sigma >= BIAS_SIGMA and abs(m) >= BIAS_FLOOR)
        out[k] = {
            "rel_l2_median": float(torch.tensor(rel).median().item()),
            "rel_l2_max": max(rel),
            "grade": grade(float(torch.tensor(rel).median().item()), dtype_name),
            "systematic_mean": m, "systematic_se": se, "systematic_sigma": sigma,
            "mean_over_rms": float(torch.tensor(acc["mean_over_rms"]).mean().item()),
            "verdict_bias": "BIAS" if bias else "NOISE",
            "draws": len(rel),
            "bitexact_repeat": det.get(k) if det else None,
        }
        if acc["rel_bw"]:
            m2 = float(torch.tensor(acc["rel_bw"]).median().item())
            out[k]["rel_l2_bw_only_median"] = m2
            out[k]["grade_bw_only"] = grade(m2, dtype_name)
    fwd_med = float(torch.tensor(fwd).median().item()) if fwd else None
    suspect = bool(dtype_name == "float32" and fwd_med is not None and fwd_med >= 1.0e-1)
    res = {"instrument_suspect": suspect, "grads": out, "fd": fd,
           "forward_rel_l2_median": fwd_med}
    if rowsum:
        t = torch.tensor(rowsum, dtype=torch.float64)
        # A softmax row must sum to 1. How far the card's does is the whole of the leading
        # factor's error, because dx carries y multiplicatively.
        res["device_row_sum"] = {"mean": t[:, 0].mean().item(), "min": t[:, 1].min().item(),
                                 "max": t[:, 2].max().item(),
                                 "mean_minus_one": t[:, 0].mean().item() - 1.0}
    return res


def aiclk(card):
    try:
        raw = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"], capture_output=True,
                             text=True, timeout=60).stdout
        d = json.loads(raw)
        return float(d["device_info"][card]["telemetry"]["aiclk"].strip())
    except Exception as e:                                          # noqa: BLE001
        return f"unavailable: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--ops", default="")
    ap.add_argument("--dtypes", default="bfloat16,float32")
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    a = ap.parse_args()

    global dev
    dev = get_device()

    want = [c for c in CASES if not a.ops or any(c["key"].startswith(p)
                                                 for p in a.ops.split(","))]
    rep = {"instrument": "of3t-p10grad per-op float64 gradient reference",
           "bar": "perf/of3t_p10grad/PREREGISTRATION.md",
           "host": socket.gethostname(), "card": a.card,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
           "renorm": ag.SOFTMAX_BW_RENORM, "fused_softmax_bw": ag.SOFTMAX_BW_FUSED,
           "exact_layer_norm_installed": ag.exact_layer_norm_installed(),
           "draws": a.draws, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "aiclk_start": aiclk(a.card), "aiclk_during": [], "cases": {}}

    for c in want:
        rep["cases"][c["key"]] = {"site": c["site"], "verb": c["verb"],
                                  "shapes": {k: list(v) for k, v in c["shapes"].items()},
                                  "dtypes": {}}
        for dt in a.dtypes.split(","):
            t0 = time.time()
            try:
                r = run_case(c, dt, a.draws)
                r["seconds"] = round(time.time() - t0, 1)
            except Exception as e:                                  # noqa: BLE001
                import traceback
                r = {"error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()[-2000:]}
            rep["cases"][c["key"]]["dtypes"][dt] = r
            rep["aiclk_during"].append(aiclk(a.card))
            print(f"[{c['key']} {dt}] " + json.dumps(r.get("grads", r))[:600], flush=True)
            with open(a.out, "w") as fh:
                json.dump(rep, fh, indent=1)

    clk = [x for x in rep["aiclk_during"] if isinstance(x, float)]
    rep["aiclk_summary"] = ({"n": len(clk), "min": min(clk), "median": sorted(clk)[len(clk) // 2],
                             "max": max(clk)} if clk else "NO SAMPLES")
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("AICLK during:", rep["aiclk_summary"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
