"""What does the DEVICE layer-norm backward actually lose, and where?

`of3t-p10exact` step 2. `autograd.py:1325` (and its `_taped_layer_norm` twin at 2723) recompute
mean and rstd in the backward and form

    dx = rstd * (dnorm - mean(dnorm) - norm * mean(dnorm * norm))

entirely in the forward dtype, which is bf16 for this model. `exactln.exact_layer_norm()` does the
same algebra in float64 and rounds ONCE at the end, and that rung clears the clause while this one
does not. So the difference is intermediate arithmetic, not output dtype, and this probe measures
it on the REAL tensors the trunk produces rather than on synthetic ones: for every call it
samples, it computes the same three gradients three ways,

    dev_bf16   exactly what production runs today
    dev_fp32   the same composite with x, g and every intermediate in fp32, on device
    float64    the host reference

and reports rel_l2 and cos of the two device arms against float64. The taped value handed back to
the tape is ALWAYS dev_bf16 unless --fp32 is given, so a probing run is bit-identical to an
unprobed one and its .pt is still a valid rung.

The `ttnn.sum` accumulator is NOT the suspect: `sum_accumulator.py` measured it flat in N
(1.752e-03 at 64 rows, 1.848e-03 at 147,456), which is one bf16 output rounding and no sqrt(N).
"""
from __future__ import annotations

import contextlib
import json

import torch
import ttnn

from tt_bio import autograd as ag
from tt_bio import taped_ttnn as tt

STATS = {"verb": 0, "bw": 0, "probed": 0, "fp32_returned": 0}
CALLS = []
_CFG = {"limit": 32, "stride": 37, "fp32": False}


def _m(a, ref):
    a, ref = a.reshape(-1).double(), ref.reshape(-1).double()
    na, nr = a.norm(), ref.norm()
    return {"rel_l2": float((a - ref).norm() / nr) if nr > 0 else None,
            "cos": float(a @ ref / (na * nr)) if na > 0 and nr > 0 else None}


def _composite(xv, g, gamma_v, eps, cfg):
    """The layer-norm backward from `autograd.py`, verbatim, on whatever dtype it is handed."""
    mean = ttnn.mean(xv, dim=-1, keepdim=True)
    centered = ttnn.subtract(xv, mean)
    var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                    compute_kernel_config=cfg)
    rstd = ttnn.rsqrt(ttnn.add(var, eps))
    norm = ttnn.multiply(centered, rstd)
    dgamma = (ag._sum_leading(ttnn.multiply(g, norm), [int(gamma_v.shape[-1])])
              if gamma_v is not None else None)
    dbeta = ag._sum_leading(g, [int(g.shape[-1])])
    dnorm = ttnn.multiply(g, gamma_v) if gamma_v is not None else g
    dn_mean = ttnn.mean(dnorm, dim=-1, keepdim=True)
    dn_norm_mean = ttnn.mean(ttnn.multiply(dnorm, norm), dim=-1, keepdim=True)
    dx = ttnn.subtract(ttnn.subtract(dnorm, dn_mean), ttnn.multiply(norm, dn_norm_mean))
    return dgamma, dbeta, ttnn.multiply(dx, rstd)


def _f64(x64, g64, gamma64, eps):
    d = x64.shape[-1]
    xc = x64 - x64.mean(-1, keepdim=True)
    rstd = torch.rsqrt((xc * xc).mean(-1, keepdim=True) + eps)
    n = xc * rstd
    dgamma = (g64 * n).reshape(-1, d).sum(0) if gamma64 is not None else None
    dbeta = g64.reshape(-1, d).sum(0)
    dn = g64 if gamma64 is None else g64 * gamma64
    dx = rstd * (dn - dn.mean(-1, keepdim=True) - n * (dn * n).mean(-1, keepdim=True))
    return dgamma, dbeta, dx


def _verb(shipped, args, kwargs):
    args = list(args) + [None] * (3 - len(args))
    x, gamma, beta = (ag._wrap(args[0]), ag._wrap(args[1]), ag._wrap(args[2]))
    kw = dict(kwargs)
    for nm, slot in (("weight", 1), ("bias", 2)):
        if kw.get(nm) is not None:
            v = ag._wrap(kw.pop(nm))
            gamma, beta = (v, beta) if slot == 1 else (gamma, v)
        else:
            kw.pop(nm, None)
    kw.pop("l1_headroom", None)
    eps = kw.pop("epsilon", 1e-5)
    cfg = kw.pop("compute_kernel_config", None)
    STATS["verb"] += 1
    out_v = shipped(x.value, weight=(gamma.value if gamma is not None else None),
                    bias=(beta.value if beta is not None else None),
                    epsilon=eps, compute_kernel_config=cfg, **kw)
    parents = [t for t in (x, gamma, beta) if t is not None]

    def make():
        def bw(g):
            STATS["bw"] += 1
            k = STATS["bw"] - 1
            probe = STATS["probed"] < _CFG["limit"] and (k < 8 or k % _CFG["stride"] == 0)
            bwcfg = ag.precise_config()
            xv = x.value
            gv = gamma.value if gamma is not None else None

            dg, db, dx = _composite(xv, g, gv, eps, bwcfg)
            alt = None
            if probe or _CFG["fp32"]:
                xf = ttnn.typecast(xv, ttnn.float32)
                gf = ttnn.typecast(g, ttnn.float32)
                gvf = ttnn.typecast(gv, ttnn.float32) if gv is not None else None
                alt = _composite(xf, gf, gvf, eps, bwcfg)
                for t in (xf, gf) + ((gvf,) if gvf is not None else ()):
                    ttnn.deallocate(t)

            if probe:
                STATS["probed"] += 1
                x64 = ttnn.to_torch(xv).double()
                g64 = ttnn.to_torch(g).double()
                gv64 = ttnn.to_torch(gv).double().reshape(-1) if gv is not None else None
                r = _f64(x64, g64, gv64, eps)
                rec = {"call": k, "shape": list(x64.shape),
                       "rows": int(x64.numel() // x64.shape[-1]), "d": int(x64.shape[-1]),
                       "g_dtype": str(g.dtype), "x_dtype": str(xv.dtype), "eps": eps,
                       "sum_leading_branch": "tree" if xv.dtype == ttnn.float32 else "ttnn.sum"}
                for nm, dev, dev32, ref in (("dgamma", dg, alt[0], r[0]),
                                            ("dbeta", db, alt[1], r[1]),
                                            ("dx", dx, alt[2], r[2])):
                    if ref is None or dev is None:
                        continue
                    rec[nm] = {"bf16": _m(ttnn.to_torch(dev), ref),
                               "fp32": _m(ttnn.to_torch(dev32), ref)}
                CALLS.append(rec)

            if alt is not None:
                keep, drop = ((alt, (dg, db, dx)) if _CFG["fp32"] else ((dg, db, dx), alt))
                for t in drop:
                    if t is not None:
                        ttnn.deallocate(t)
                dg, db, dx = keep
                if _CFG["fp32"]:
                    STATS["fp32_returned"] += 1

            if gamma is not None and gamma.requires_grad:
                gamma.add_grad(ttnn.reshape(dg, [int(v) for v in gamma.value.shape]))
            if beta is not None and beta.requires_grad:
                beta.add_grad(ttnn.reshape(db, [int(v) for v in beta.value.shape]))
            if x.requires_grad:
                x.add_grad(dx)
        return bw

    return ag._tape(out_v, parents, make)


@contextlib.contextmanager
def ln_probe(limit=32, stride=37, fp32=False):
    _CFG.update(limit=limit, stride=stride, fp32=fp32)
    saved = (tt._VERBS["layer_norm"], ag._TAPED["layer_norm"])
    tt._VERBS["layer_norm"] = _verb
    ag._TAPED["layer_norm"] = _verb
    tt.forget_shim_bindings("layer_norm")
    try:
        yield
    finally:
        tt._VERBS["layer_norm"], ag._TAPED["layer_norm"] = saved
        tt.forget_shim_bindings("layer_norm")


def dump(path):
    with open(path, "w") as f:
        json.dump({"stats": dict(STATS), "cfg": dict(_CFG), "calls": CALLS}, f, indent=1)
