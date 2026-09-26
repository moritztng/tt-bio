"""Force one (q_chunk, k_chunk) on the fused-large-S route, and record what actually fired.

`_tri_att_fused_large_s` serves OpenFold3's trunk at 1088, 1216 and 1472, and its own comment says
its pair is "a NARROW q against a WIDE k" which "the stock ladder never offers". It is gated to
q_len > _Q_SPLIT_MAX_S (1024), so 832 never reaches it. The cap is env-settable, so the gate opens
without touching the repo.

Opening it is not enough by itself: at 832 the kernel's own order prefers (416, 416), the same
chunked k the dividing-k lever lands on. TT_FORCE_FUSED_PAIR pins the pair instead, so the
single-chunk (64, 832) can be measured against it.

Two hazards this file has already hit, both recorded so the next reader does not repeat them:
`tenstorrent.py` does `from . import triatt_sdpa`, so keying the patch on the imported NAME never
fires; and calling `import sys` inside the patch re-enters the wrapped __import__ and recurses
until the stack ends. sys is bound once at module scope and __import__ is guarded re-entrantly.

TT_STATS_DUMP writes the firing counters after each trunk call. hifi_picks is the proof: a leg
that reports (416, 416) did not take the pin, whatever TT_FORCE_FUSED_PAIR says.
"""
import builtins
import json
import os
import sys

_real_import = builtins.__import__
_state = {"pairs": False, "trunk": False, "n": 0, "busy": False}


def _patch_pairs():
    TS = sys.modules.get("tt_bio.triatt_sdpa")
    # Present in sys.modules is not the same as finished importing. Patching on every import means
    # this also runs DURING triatt_sdpa's own import, where the attribute does not exist yet.
    # Return without latching so a later import retries.
    if TS is None or _state["pairs"] or not hasattr(TS, "fused_pairs"):
        return
    force = os.environ.get("TT_FORCE_FUSED_PAIR")
    if not force:
        _state["pairs"] = True
        return
    qc, kc = (int(x) for x in force.split(","))
    real = TS.fused_pairs

    def wrapped(q_len, heads, head_dim, cores, dtype):
        # Only at the length whose k is being pinned; every other shape keeps its own order.
        if q_len == kc:
            return ((qc, kc),)
        return real(q_len, heads, head_dim, cores, dtype)

    TS.fused_pairs = wrapped
    _state["pairs"] = True


def _patch_trunk():
    mod = sys.modules.get("tt_bio.openfold3_trunk")
    out = os.environ.get("TT_STATS_DUMP")
    if mod is None or _state["trunk"] or not out:
        return
    cls = getattr(mod, "OF3Trunk", None)
    if cls is None:
        return
    real = cls.__call__

    def wrapped(self, *a, **k):
        r = real(self, *a, **k)
        _state["n"] += 1
        T = sys.modules.get("tt_bio.tenstorrent")
        TS = sys.modules.get("tt_bio.triatt_sdpa")
        rec = {
            "call": _state["n"],
            "forced_pair": os.environ.get("TT_FORCE_FUSED_PAIR"),
            "pin_installed": _state["pairs"] and bool(os.environ.get("TT_FORCE_FUSED_PAIR")),
            "q_split_max": getattr(TS, "_Q_SPLIT_MAX_S", None),
            "hifi_stats": dict(getattr(T, "TRIATT_FUSED_HIFI_STATS", {}) or {}),
            "hifi_picks": {str(x): y for x, y in
                           (getattr(T, "TRIATT_FUSED_HIFI_PICKS", {}) or {}).items()},
            "large_s_stats": list(getattr(T, "SDPA_FUSED_LARGE_S_STATS", []) or []),
            "kernel_stats": list(getattr(TS, "STATS", []) or []),
            "kernel_rejects": {str(x): y for x, y in (getattr(TS, "REJECTS", {}) or {}).items()},
        }
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return r

    cls.__call__ = wrapped
    _state["trunk"] = True


def _imp(name, *a, **k):
    m = _real_import(name, *a, **k)
    if _state["busy"]:
        return m
    _state["busy"] = True
    try:
        if not _state["pairs"]:
            _patch_pairs()
        if not _state["trunk"]:
            _patch_trunk()
    finally:
        _state["busy"] = False
    return m


if os.environ.get("TT_FORCE_FUSED_PAIR") or os.environ.get("TT_STATS_DUMP"):
    builtins.__import__ = _imp
