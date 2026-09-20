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

    C14_GI_WRAP=tt_bio.tenstorrent:_short_m_proj_config
    C14_GI_OUT=/tmp/gi_firing        -> writes /tmp/gi_firing.<pid>
"""
import atexit
import builtins
import json
import os
import sys

_WRAP = os.environ.get("C14_GI_WRAP")
_OUT = os.environ.get("C14_GI_OUT")
_CALLS = [0]
_DONE = []


def _flush():
    rec = {"pid": os.getpid(), "argv": " ".join(sys.argv)[:120],
           "wrap": _WRAP, "patched": bool(_DONE), "calls": _CALLS[0]}
    with open("%s.%d" % (_OUT, os.getpid()), "w") as f:
        json.dump(rec, f)


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
