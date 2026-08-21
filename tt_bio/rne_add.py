"""Round bfloat16 ``ttnn.add`` to nearest even, repo-wide. Off unless ``TT_BIO_RNE_ADD=1``.

``ttnn.add``/``ttnn.add_`` break bfloat16 ties AWAY FROM ZERO; torch and JAX break them to
even, and ttnn's bfloat16 datapath is narrower than float32, so at unequal operand magnitudes
it also drops bits an exact float32 sum keeps. 11.14% of elements differ from torch by 1 ulp at
equal operand magnitudes, 6.55% at a ratio of 0.01
(``scripts/bf16_add/patch_probe.py``). Routing the add through float32 and narrowing the sum
with ``typecast`` is bit-identical to torch's bfloat16 add: 0 of 96141312 elements differ.

Every model in the repo builds its residual trunk out of these two calls, so the lever is
global: ``install()`` swaps both ops and any harness or fold then runs the other arm without a
line of model-code change. ``tt_bio.tenstorrent`` installs it at import when the env var is set,
which is what carries it into a gate's fold subprocesses.

Calls are counted and classified. That matters: a screen whose instrument silently no-ops reads
exactly like a model that clears its bar.
"""
from __future__ import annotations

import collections
import os

import ttnn

ENV = "TT_BIO_RNE_ADD"

_orig_add = None
_orig_add_ = None

#: call classification -> count. ``patched`` is the only class whose arithmetic changed.
counts: collections.Counter = collections.Counter()


def enabled() -> bool:
    return os.environ.get(ENV, "0") != "0"


def _both_bf16(a, b) -> bool:
    return (isinstance(a, ttnn.Tensor) and isinstance(b, ttnn.Tensor)
            and a.dtype == ttnn.bfloat16 and b.dtype == ttnn.bfloat16)


def _wide_add(a, b, memory_config):
    """bfloat16 ``a + b``, rounded the way torch rounds it."""
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
    except Exception as exc:  # the float32 intermediates are 2x the operands
        counts["fallback:" + type(exc).__name__] += 1
        return _orig_add(a, b, **kwargs)
    counts["patched"] += 1
    if isinstance(out_t, ttnn.Tensor):
        ttnn.copy(out, out_t)
        ttnn.deallocate(out)
        return out_t
    return out


def add_(a, b, *args, **kwargs):
    """In place, so the sum goes back into ``a``: call sites use the mutated operand as well as
    the return value, and a patch that fixed only one of them would fire silently."""
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
    if not counts:
        return "rne_add: no add reached the lever"
    return "rne_add: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
