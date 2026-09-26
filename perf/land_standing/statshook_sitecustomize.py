"""Dump the STOCK triangle-attention ladder's counters at process exit.

`TT_BIO_SDPA_WIDE_K_UP` is consumed by `_tri_att_k_chunks`, i.e. the stock ladder, so the
question "does it fire on a real model" is answered by `SDPA_K_CHUNK_STATS` -- [calls served at a
k wider than the shipped pick, calls that fell back to the shipped pick] -- and by which pair
`SDPA_CHUNK_PICKS` records per shape. The module's own comment says why the fold has to say it:
"a silently-declined config is indistinguishable from an absent one, so an A/B on this path can
only be believed if the fold itself says which pair it ran."

atexit rather than a class patch, because the counters live on the module and the fold runs in a
spawned worker: PYTHONPATH reaches the worker, so this registers there too and each process
writes its own line keyed by pid.
"""
import atexit
import json
import os
import sys

OUT = os.environ.get("TT_STOCK_STATS_DUMP")


def _dump():
    T = sys.modules.get("tt_bio.tenstorrent")
    if T is None or not OUT:
        return
    rec = {
        "pid": os.getpid(),
        "wide_k_up_flag": bool(getattr(T, "_SDPA_WIDE_K_UP", None)),
        # [served at a wider k, fell back to the shipped k]
        "k_chunk_stats": list(getattr(T, "SDPA_K_CHUNK_STATS", []) or []),
        "route_counts": dict(getattr(T, "SDPA_ROUTE_COUNTS", {}) or {}),
        "fused_large_s": list(getattr(T, "SDPA_FUSED_LARGE_S_STATS", []) or []),
        "picks": {str(k): v for k, v in (getattr(T, "SDPA_CHUNK_PICKS", {}) or {}).items()},
        "hifi_stats": dict(getattr(T, "TRIATT_FUSED_HIFI_STATS", {}) or {}),
        "mask_trans": dict(getattr(T, "MASK_TRANS_STATS", {}) or {}),
    }
    if not rec["picks"] and not any(rec["k_chunk_stats"]):
        rec["note"] = "no triangle-attention pick recorded in this process"
    try:
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:                                              # noqa: BLE001
        pass


if OUT:
    atexit.register(_dump)
