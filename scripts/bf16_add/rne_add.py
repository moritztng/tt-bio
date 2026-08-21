"""Repo-wide round-to-nearest-even bfloat16 ``ttnn.add``, as a measurement arm.

``ttnn.add``/``ttnn.add_`` break bfloat16 ties AWAY FROM ZERO; torch and JAX break them to
even, and ttnn's bfloat16 datapath is narrower than float32, so at unequal operand magnitudes
it also drops bits an exact float32 sum keeps. Measured in
``scripts/af2_port/residual_add_probe.py``: 11.16% of elements differ from torch by 1 ulp at
equal operand magnitudes. Routing the add through float32 and narrowing the sum with
``typecast`` is bit-identical to torch's bfloat16 add (0 of 5,537,792 elements differ).

``install()`` swaps both ops for the float32-routed version, so a harness runs A/B without a
line of model-code change. Every call is counted and classified. That matters: a screen whose
instrument silently no-ops reads exactly like a model that clears comfortably.

Use ``run_with_rne.py <script.py> [args...]`` to wrap an existing harness.
"""
from __future__ import annotations

import collections

import ttnn

_orig_add = None
_orig_add_ = None

#: call classification -> count. ``patched`` is the only class that changed arithmetic.
counts: collections.Counter = collections.Counter()


def _both_bf16(a, b) -> bool:
    return (
        isinstance(a, ttnn.Tensor)
        and isinstance(b, ttnn.Tensor)
        and a.dtype == ttnn.bfloat16
        and b.dtype == ttnn.bfloat16
    )


def _wide_add(a, b, memory_config):
    """bfloat16 ``a + b`` rounded the way torch rounds it."""
    wa = ttnn.typecast(a, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    wb = ttnn.typecast(b, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s = _orig_add(wa, wb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(wa)
    ttnn.deallocate(wb)
    out = ttnn.typecast(s, ttnn.bfloat16, memory_config=memory_config)
    ttnn.deallocate(s)
    return out


def _skip(kwargs) -> str | None:
    """A kwarg this arm must not silently emulate away."""
    if kwargs.get("activations"):
        return "skip_activations"
    dtype = kwargs.get("dtype")
    if dtype is not None and dtype != ttnn.bfloat16:
        return "skip_dtype"
    return None


def add(a, b, *args, **kwargs):
    if args or not _both_bf16(a, b):
        counts["skip_not_bf16"] += 1
        return _orig_add(a, b, *args, **kwargs)
    why = _skip(kwargs)
    if why:
        counts[why] += 1
        return _orig_add(a, b, **kwargs)
    out_t = kwargs.get("output_tensor")
    mcfg = kwargs.get("memory_config") or (
        out_t.memory_config() if isinstance(out_t, ttnn.Tensor) else a.memory_config())
    try:
        out = _wide_add(a, b, mcfg)
    except Exception as exc:  # L1/DRAM pressure from the float32 intermediates
        counts["fallback:" + type(exc).__name__] += 1
        return _orig_add(a, b, **kwargs)
    counts["patched"] += 1
    if isinstance(out_t, ttnn.Tensor):
        ttnn.copy(out, out_t)
        ttnn.deallocate(out)
        return out_t
    return out


def add_(a, b, *args, **kwargs):
    """In place, so the result is written back into ``a``: callers use both the return value
    and the mutated operand, and a patch that only fixed one of them would fire silently."""
    if args or not _both_bf16(a, b) or _skip(kwargs):
        counts["skip_inplace_other"] += 1
        return _orig_add_(a, b, *args, **kwargs)
    try:
        out = _wide_add(a, b, a.memory_config())
    except Exception as exc:
        counts["fallback:" + type(exc).__name__] += 1
        return _orig_add_(a, b, **kwargs)
    ttnn.copy(out, a)
    ttnn.deallocate(out)
    counts["patched"] += 1
    return a


def install() -> None:
    global _orig_add, _orig_add_
    if _orig_add is not None:
        return
    _orig_add, _orig_add_ = ttnn.add, ttnn.add_
    ttnn.add, ttnn.add_ = add, add_


def uninstall() -> None:
    global _orig_add, _orig_add_
    if _orig_add is None:
        return
    ttnn.add, ttnn.add_ = _orig_add, _orig_add_
    _orig_add = _orig_add_ = None


def report() -> str:
    return "rne_add: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "rne_add: idle"
