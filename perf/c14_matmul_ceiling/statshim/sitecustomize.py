"""Record every `_short_m_proj_config` call in EVERY process of a fold, spawn children included.

Two reasons this is not just a read of `MM_SHORT_M_STATS`:

* the counter is incremented only after the dtype and rank screens, so 0/0 cannot tell "never
  called" from "called and rejected before the count";
* `tt_bio.main predict` folds in a worker process, not in the process that parsed the argv, so a
  wrap installed by a driver script never sees the calls that matter.

`sitecustomize` is imported by every interpreter, and wrapping `builtins.__import__` is what
catches `tt_bio.tenstorrent` whenever it lands, in whichever process. Records flush every 200
calls, so nothing depends on how the process exits. Enabled only when C14_MM_STATS_OUT is set.
"""
import atexit
import builtins
import json
import os
import sys

_OUT = os.environ.get("C14_MM_STATS_OUT")
_SEEN = {}
_DONE = []


def _flush():
    m = sys.modules.get("tt_bio.tenstorrent")
    rec = {"pid": os.getpid(), "argv": " ".join(sys.argv)[:100],
           "counter": [int(v) for v in getattr(m, "MM_SHORT_M_STATS", [])] if m else None,
           "flag": bool(getattr(m, "_MM_SHORT_M_BW", False)) if m else None,
           "groups": [{"x": list(k[0]), "w": list(k[1]), "x_dtype": k[2], "w_dtype": k[3],
                       "calls": v[0], "served": v[1]} for k, v in _SEEN.items()]}
    with open("%s.%d" % (_OUT, os.getpid()), "w") as f:
        json.dump(rec, f)


def _patch(m):
    orig = m._short_m_proj_config

    def wrapped(x, w, _o=orig):
        pc = _o(x, w)
        key = (tuple(int(v) for v in x.shape), tuple(int(v) for v in w.shape),
               str(x.dtype), str(w.dtype))
        e = _SEEN.setdefault(key, [0, 0])
        e[0] += 1
        e[1] += int(pc is not None)
        if e[0] % 200 == 1:
            _flush()
        return pc

    m._short_m_proj_config = wrapped


if _OUT:
    _real_import = builtins.__import__

    def _imp(name, *a, **kw):
        mod = _real_import(name, *a, **kw)
        if not _DONE:
            m = sys.modules.get("tt_bio.tenstorrent")
            if m is not None and hasattr(m, "_short_m_proj_config"):
                _DONE.append(1)
                _patch(m)
        return mod

    builtins.__import__ = _imp
    atexit.register(_flush)
