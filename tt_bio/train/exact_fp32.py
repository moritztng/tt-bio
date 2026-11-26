"""An fp32 control whose reductions are genuinely fp32: `with exact_fp32(): ...`.

The Blackhole matrix engine has no fp32 dot product. A single product is exact, but a 32-term
dot of operands that are exactly representable in bf16 already reads 3.1e-4 against float64 at
HiFi4 (`perf/r4t_tape/kacc.py`), splitting operands into bf16 parts does not move it
(`split_mm.py`), and fp32 `layer_norm`, `sum` and tile-face transposes read 1.2e-3, 4.3e-4 and
4.2e-4 (`fp32_ops.py`). Elementwise fp32 ops are exact to 5e-8. So an "fp32" arm built from
the card's own verbs grades a gradient against a 1e-3 floor per reduction, and a reduction over
n^2 pair rows with cancellation turns that into 5e-2 (r4t D49).

Inside this scope those verbs, handed float32 device operands only, compute on the host in
float64 and come back rounded once to float32: the arithmetic of an exact fp32 machine.
Everything else, every elementwise op, every taped gradient rule and the order in which the
tape composes them, still runs on the card as shipped. It is a verification instrument for the
gradient check's fp32 arm: it is never entered by a served path, it changes nothing outside the
`with`, and a call it does not cover (a non-float32 operand, an unknown keyword that changes the
math) goes to the real verb untouched, counted under "passed:<operand dtypes>".
"""
from __future__ import annotations

import contextlib

import torch
import ttnn

# Covered: matmul, linear, experimental.minimal_matmul (the trimul in-projection; uncovered it alone
# held the PairUpdate forward at 6.1e-4 against float64, covered 1.2e-7), sum, mean, layer_norm,
# rms_norm, transpose, permute. Computing the elementwise verbs on the host as well moved nothing.
EXACT_STATS: dict = {}
_ACT = {None: lambda t: t, "relu": torch.relu, "silu": torch.nn.functional.silu,
        "gelu": torch.nn.functional.gelu, "sigmoid": torch.sigmoid}


def _f32_dev(*ts) -> bool:
    return all(isinstance(t, ttnn.Tensor) and t.dtype == ttnn.float32
               and t.storage_type() == ttnn.StorageType.DEVICE for t in ts if t is not None)


def _host(t):
    return ttnn.to_torch(t).double()


def _back(r, like, memory_config=None):
    kw = {"memory_config": memory_config} if memory_config is not None else {}
    try:
        return ttnn.from_torch(r.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                               device=like.device(), **kw)
    except Exception:  # noqa: BLE001  a sharded config needs a shard spec it cannot take from here
        return ttnn.from_torch(r.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                               device=like.device())


def _why(*ts):
    return ",".join("-" if t is None else str(getattr(t, "dtype", type(t).__name__)).split(".")[-1]
                    for t in ts)


def _note(name, served, why=""):
    k = (name, "exact" if served else "passed" + (":" + why if why else ""))
    EXACT_STATS[k] = EXACT_STATS.get(k, 0) + 1


def _mm(real):
    def matmul(a, b, *args, transpose_a=False, transpose_b=False, memory_config=None, **kw):
        if args or not _f32_dev(a, b) or kw.get("dtype") not in (None, ttnn.float32, ttnn.bfloat16):
            _note("matmul", False, _why(a, b) + ",args=%d,dtype=%s" % (len(args), kw.get("dtype")))
            return real(a, b, *args, transpose_a=transpose_a, transpose_b=transpose_b,
                        memory_config=memory_config, **kw)
        A, B = _host(a), _host(b)
        A = A.transpose(-1, -2) if transpose_a else A
        B = B.transpose(-1, -2) if transpose_b else B
        _note("matmul", True)
        return _back(A @ B, a, memory_config)
    return matmul


def _mmm(real):
    """`ttnn.experimental.minimal_matmul`: the trimul in-projection's matmul (`config` is blocking only)."""
    def minimal_matmul(x, w, *args, bias_tensor=None, memory_config=None, dtype=None, **kw):
        if args or not _f32_dev(x, w, bias_tensor) or dtype not in (None, ttnn.float32) \
                or set(kw) - {"compute_kernel_config", "config"}:
            _note("minimal_matmul", False, _why(x, w, bias_tensor) + ",args=%d,dtype=%s" % (len(args), dtype))
            return real(x, w, *args, bias_tensor=bias_tensor, memory_config=memory_config, dtype=dtype, **kw)
        y = _host(x) @ _host(w)
        if bias_tensor is not None:
            y = y + _host(bias_tensor).reshape(-1)[: y.shape[-1]]
        _note("minimal_matmul", True)
        return _back(y, x, memory_config)
    return minimal_matmul


def _lin(real):
    def linear(x, w, *args, bias=None, transpose_a=False, transpose_b=False, activation=None,
               memory_config=None, **kw):
        if len(args) == 1 and bias is None:
            bias, args = args[0], ()
        if args or not _f32_dev(x, w, bias) or activation not in _ACT:
            _note("linear", False, _why(x, w, bias) + ",args=%d,act=%s" % (len(args), activation))
            return real(x, w, *args, bias=bias, transpose_a=transpose_a, transpose_b=transpose_b,
                        activation=activation, memory_config=memory_config, **kw)
        X, W = _host(x), _host(w)
        X = X.transpose(-1, -2) if transpose_a else X
        W = W.transpose(-1, -2) if transpose_b else W
        y = X @ W
        if bias is not None:
            y = y + _host(bias).reshape(-1)[: y.shape[-1]]
        _note("linear", True)
        return _back(_ACT[activation](y), x, memory_config)
    return linear


def _reduce(real, name, op):
    def reduce(x, dim=None, keepdim=False, *args, memory_config=None, **kw):
        if args or not _f32_dev(x) or kw.get("scalar") is not None:
            _note(name, False, _why(x))
            return real(x, dim, keepdim, *args, memory_config=memory_config, **kw)
        X = _host(x)
        dims = tuple(range(X.dim())) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
        _note(name, True)
        return _back(op(X, dim=dims, keepdim=keepdim), x, memory_config)
    return reduce


def _norm(real, name, center):
    def norm(x, *args, epsilon=1e-12, weight=None, bias=None, residual_input_tensor=None,
             memory_config=None, **kw):
        if args or not _f32_dev(x, weight, bias, residual_input_tensor):
            _note(name, False, _why(x, weight, bias, residual_input_tensor))
            return real(x, *args, epsilon=epsilon, weight=weight, bias=bias,
                        residual_input_tensor=residual_input_tensor, memory_config=memory_config, **kw)
        X = _host(x)
        if residual_input_tensor is not None:
            X = X + _host(residual_input_tensor)
        if center:
            X = X - X.mean(-1, keepdim=True)
        y = X * torch.rsqrt((X * X).mean(-1, keepdim=True) + epsilon)
        if weight is not None:
            y = y * _host(weight).reshape(-1)[: y.shape[-1]]
        if bias is not None:
            y = y + _host(bias).reshape(-1)[: y.shape[-1]]
        _note(name, True)
        return _back(y, x, memory_config)
    return norm


def _move(real, name):
    def move(x, *dims, memory_config=None, **kw):
        if kw or not _f32_dev(x):
            _note(name, False, _why(x) + ",kw=" + "/".join(kw))
            return real(x, *dims, memory_config=memory_config, **kw)
        X = _host(x)
        _note(name, True)
        out = X.transpose(*dims) if name == "transpose" else X.permute(*dims[0])
        return _back(out, x, memory_config)
    return move


@contextlib.contextmanager
def exact_fp32():
    """Swap the reducing verbs on `ttnn` for exact ones for the length of the `with`."""
    from .. import taped_ttnn
    swap = {"matmul": _mm, "linear": _lin,
            "sum": lambda r: _reduce(r, "sum", torch.sum), "mean": lambda r: _reduce(r, "mean", torch.mean),
            "layer_norm": lambda r: _norm(r, "layer_norm", True), "rms_norm": lambda r: _norm(r, "rms_norm", False),
            "transpose": lambda r: _move(r, "transpose"), "permute": lambda r: _move(r, "permute")}
    real = {k: getattr(ttnn, k) for k in swap}
    real_mmm = ttnn.experimental.minimal_matmul
    try:
        for k, f in swap.items():
            setattr(ttnn, k, f(real[k]))
        ttnn.experimental.minimal_matmul = _mmm(real_mmm)
        taped_ttnn.forget_shim_bindings()
        yield EXACT_STATS
    finally:
        for k, v in real.items():
            setattr(ttnn, k, v)
        ttnn.experimental.minimal_matmul = real_mmm
        taped_ttnn.forget_shim_bindings()
