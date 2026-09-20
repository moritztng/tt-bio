"""Count calls to any one function, in EVERY process of a fold, spawn children included.

Generalises `perf/c14_matmul_ceiling/statshim`, which does the same job for one hard-coded
function. Two reasons a fold needs this rather than a counter read in the driver:

* `tt_bio.main predict` folds in a WORKER process, not the one that parsed argv, so a wrap
  installed by a driver script never sees the calls that matter
  (`in-process-patch-never-reaches-a-spawn-child`);
* a production counter is usually incremented after the function's own early returns, so 0/0
  cannot separate "never called" from "called and rejected before the count".

`sitecustomize` is imported by every interpreter, and wrapping `builtins.__import__` catches the
target module whenever it lands, in whichever process. Records flush every 200 calls, so nothing
depends on how the process exits.

    C14_GI_WRAP=tt_bio.tenstorrent:_short_m_proj_config   count calls to a function
    C14_GI_COUNTER=tt_bio.tenstorrent:APB_CONCAT_HEADS_STATS   snapshot a production counter
    C14_GI_OUT=/tmp/gi_firing        -> writes /tmp/gi_firing.<pid>

`C14_GI_COUNTER` is for a lever decided once at construction rather than per call, where there is
nothing useful to wrap but the model already keeps a `[served, declined]` list. It needs no patch
and no import hook of its own: the list is read out of the live module at flush. Either variable
works alone; a run that sets neither records nothing and says so.
"""
import atexit
import builtins
import json
import os
import sys

_WRAP = os.environ.get("C14_GI_WRAP")
_COUNTER = os.environ.get("C14_GI_COUNTER")
_OUT = os.environ.get("C14_GI_OUT")
_CALLS = [0]
_DONE = []


def _counter_value():
    """The live [served, declined] list, or None if the module never loaded in this process."""
    if not _COUNTER:
        return None
    mod, attr = _COUNTER.split(":")
    m = sys.modules.get(mod)
    if m is None:
        return None
    v = getattr(m, attr, None)
    return list(v) if isinstance(v, (list, tuple)) else v


def _flush():
    rec = {"pid": os.getpid(), "argv": " ".join(sys.argv)[:120],
           "wrap": _WRAP, "patched": bool(_DONE), "calls": _CALLS[0],
           "counter_name": _COUNTER, "counter": _counter_value(),
           "env_flags": {k: v for k, v in os.environ.items() if k.startswith("TT_BIO_")}}
    with open("%s.%d" % (_OUT, os.getpid()), "w") as f:
        json.dump(rec, f)


if _COUNTER and _OUT and not _WRAP:
    atexit.register(_flush)

if _WRAP and _OUT:
    _mod, _attr = _WRAP.split(":")

    def _patch(m):
        orig = getattr(m, _attr)

        def wrapped(*a, _o=orig, **kw):
            _CALLS[0] += 1
            if _CALLS[0] % 200 == 1:
                _flush()
            return _o(*a, **kw)

        setattr(m, _attr, wrapped)

    _real_import = builtins.__import__

    def _imp(name, *a, **kw):
        mod = _real_import(name, *a, **kw)
        if not _DONE:
            m = sys.modules.get(_mod)
            if m is not None and hasattr(m, _attr):
                _DONE.append(1)
                _patch(m)
        return mod

    builtins.__import__ = _imp
    atexit.register(_flush)
