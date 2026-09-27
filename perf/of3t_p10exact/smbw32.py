"""The softmax backward in fp32, with the shipped identity kept exactly as it is.

`of3t-bwdaccum`s `softmax_fp32` lever is not this. It replaces the taped softmax VERB with its
own forward and a backward `y (g - sum(g y))`, dropping the row-sum correction that
`tt_bio.autograd.softmax_bw_inner` applies under `TT_BIO_SOFTMAX_BW_RENORM=1`. Measured on the
model frame, `--lever ceiling` (that lever on top of `all`) reads clause 3.1578980864825232,
x_bar 20.76, cos 0.0552 against `all`s 0.8476, and it is FASTER than `all` (42.5 s against
83.2 s). A precision lever cannot buy time, so that arm is a different function, not a more
precise one.

This scope changes the arithmetic and nothing else. `softmax_bw` is replaced by a copy that
computes the SAME expression -- `y (g - sum(g y) / sum(y))` -- in fp32 and casts the result back
to the gradients dtype, so the forward, the verb, the tape and the identity are untouched:

    inner = ttnn.sum(ttnn.multiply(g, y), dim, keepdim=True)     # autograd.py:166, NO kernel
                                                                 # config and bf16 operands
    inner = ttnn.divide(inner, ttnn.sum(y, ..., precise_config()))
    dx    = ttnn.multiply(y, ttnn.subtract(g, inner))            # a near-cancellation in bf16

`g - inner` cancels: at a converged row `inner` approaches `g`s mass, so the difference keeps
only the low bits of two bf16 numbers. That is where the precision goes, and the reduction that
feeds it is the one reduction in the file carrying no `compute_kernel_config` at all.

All three callers are reached by patching the module global: `taped_ttnn.py:227` looks the name
up on the module at call time, and `autograd.py:1366` (the taped softmax verb) and `:2097`
(`triangle_attention`s chunked backward) resolve it in `tt_bio.autograd`s own globals, which is
the object this rebinds.
"""
from __future__ import annotations

import contextlib

STATS = {"fired": 0, "elements": 0, "installed": False}


@contextlib.contextmanager
def softmax_bw_fp32():
    import ttnn
    from tt_bio import autograd as ag

    shipped = ag.softmax_bw

    def softmax_bw(y, g, dim=-1, config=None):
        STATS["fired"] += 1
        STATS["elements"] += int(y.volume() if hasattr(y, "volume") else 0)
        out_dtype = g.dtype
        y32 = ttnn.typecast(y, ttnn.float32)
        g32 = ttnn.typecast(g, ttnn.float32)
        cfg = config or ag.precise_config()
        prod = ttnn.multiply(g32, y32)
        inner = ttnn.sum(prod, dim=dim, keepdim=True, compute_kernel_config=cfg)
        ttnn.deallocate(prod)
        if ag.SOFTMAX_BW_RENORM:
            s = ttnn.sum(y32, dim=dim, keepdim=True, compute_kernel_config=cfg)
            inner = ttnn.divide(inner, s)
            ttnn.deallocate(s)
        diff = ttnn.subtract(g32, inner)
        ttnn.deallocate(inner)
        ttnn.deallocate(g32)
        dx = ttnn.multiply(y32, diff)
        ttnn.deallocate(diff)
        ttnn.deallocate(y32)
        out = ttnn.typecast(dx, out_dtype) if dx.dtype != out_dtype else dx
        if out is not dx:
            ttnn.deallocate(dx)
        return out

    ag.softmax_bw = softmax_bw
    STATS["installed"] = ag.softmax_bw is softmax_bw
    try:
        yield
    finally:
        ag.softmax_bw = shipped
