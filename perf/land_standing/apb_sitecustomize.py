"""Dump APB_CONCAT_HEADS_STATS at process exit, one line per pid.

The fused-HiFi gate was run without counters and had to be re-run purely to answer "did the lever
serve a single call", which turned out to be zero for a structural reason. A gate arm that cannot
say whether the flag executed proves only that nothing broke, so this rides along from the start.

atexit rather than a class patch: the fold runs in a spawned worker, PYTHONPATH reaches it, and
each process writes its own line keyed by pid. A release_gate run exits normally, so atexit fires
(unlike the BC2 pytest case, where a teardown SIGABRT destroys every end-of-run hook).
"""
import atexit
import json
import os
import sys

OUT = os.environ.get("TT_APB_STATS_DUMP")


def _dump():
    T = sys.modules.get("tt_bio.tenstorrent")
    if T is None or not OUT:
        return
    stats = list(getattr(T, "APB_CONCAT_HEADS_STATS", []) or [])
    rec = {
        "pid": os.getpid(),
        "apb_flag": bool(getattr(T, "_APB_CONCAT_HEADS", None)),
        # [served, declined] at the token head re-assembly
        "apb_stats": stats,
        "served": stats[0] if len(stats) > 0 else None,
        "declined": stats[1] if len(stats) > 1 else None,
    }
    if not any(stats):
        rec["note"] = "no APB head re-assembly recorded in this process"
    try:
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:                                              # noqa: BLE001
        pass


if OUT:
    atexit.register(_dump)
