"""Dump reblock_permute's own call counters at interpreter exit, per process.

`tt-bio predict` folds in a multiprocessing-spawn worker, so a counter read in the launching
process is always 0 and cannot say whether E6 ran inside the fold. A spawn child re-execs python
and therefore imports `sitecustomize`, so putting this directory first on PYTHONPATH gets the
counter out of every process in the fold, keyed by pid.

Set TT_BIO_REBLOCK_STATS_DIR to turn it on. Reads nothing and changes nothing if the module was
never imported.
"""
import atexit
import json
import os
import sys

_DIR = os.environ.get("TT_BIO_REBLOCK_STATS_DIR")


def _dump():
    RB = sys.modules.get("tt_bio.reblock_permute")
    if RB is None:
        return
    T = sys.modules.get("tt_bio.tenstorrent")
    rec = {
        "pid": os.getpid(),
        "gated_moves": RB.STATS_GATED[0], "gated_rejects": RB.STATS_GATED[1],
        "plain_moves": RB.STATS[0], "plain_rejects": RB.STATS[1],
        "back_moves": RB.STATS_BACK[0], "back_rejects": RB.STATS_BACK[1],
        "mask_after_move": getattr(T, "_TRIMUL_MASK_AFTER_MOVE", None),
        "rejects": {f"{k[0]}{list(k[1])}": v for k, v in RB.REJECTS.items()},
    }
    with open(os.path.join(_DIR, f"reblock_{os.getpid()}.json"), "w") as f:
        json.dump(rec, f, indent=2)


if _DIR:
    os.makedirs(_DIR, exist_ok=True)
    atexit.register(_dump)
