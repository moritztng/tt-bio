"""Read the pad-up counters in the process that folds.

`tt-bio predict` spawns its worker with `mp.get_context("spawn")` and stops it with
`terminate()`, so the child never runs `atexit`: the first attempt printed the PARENT's counters,
which are all zeros because the parent never folds. The dump is therefore registered at import
(so the spawned child registers it too), on SIGTERM and SIGINT as well as at exit, and it writes a
file per pid so nothing depends on the child's stdout still being attached.
"""
import atexit
import json
import os
import signal
import sys

from tt_bio import tenstorrent as T

OUT = os.environ.get("INERT_DUMP_DIR", "/tmp")
TAG = os.environ.get("INERT_TAG", "leg")
_done = set()


def dump(why="exit"):
    if why in _done:
        return
    _done.add(why)
    rec = {"pid": os.getpid(), "ppid": os.getppid(), "why": why, "tag": TAG,
           "stats": dict(T.TRIATT_FUSED_HIFI_STATS),
           "padded": {str(k): v for k, v in T.TRIATT_FUSED_HIFI_PADDED.items()},
           "picks": {str(k): v for k, v in T.TRIATT_FUSED_HIFI_PICKS.items()},
           "pad_up_tiles": T._TRIATT_HIFI_PAD_UP_TILES}
    line = "INERT_DUMP " + json.dumps(rec)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(os.path.join(OUT, f"inert_{TAG}_{os.getpid()}_{why}.json"), "w") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _sig(signum, frame):
    dump(f"signal{signum}")
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


atexit.register(dump)
for _s in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_s, _sig)
    except (ValueError, OSError):
        pass

if __name__ == "__main__":
    sys.argv = ["tt-bio", "predict"] + sys.argv[1:]
    from tt_bio.main import cli
    cli()
