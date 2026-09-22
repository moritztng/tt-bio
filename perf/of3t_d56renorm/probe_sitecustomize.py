"""Per-process witness for D56, loaded by EVERY interpreter without touching shipped code.

Python imports `sitecustomize` automatically in every interpreter it starts, including the
workers `tt_bio predict` spawns. That is the only hook that reaches those workers without
putting verification machinery in the engine: a `runpy` wrapper breaks `multiprocessing`
spawn, and an `atexit` registered in `tt_bio.autograd` cannot fire in a process that never
imports `tt_bio.autograd` -- which, on an inference fold, is every process.

That last point is the measurement. `imported` False is not a missing reading; it is the
result, and it is stronger than a zero counter: the module holding the flag and the branch was
never loaded, so nothing could have read either.

Writes one json per pid into $TT_BIO_D56_PROBE_DIR. Inert when that is unset.
"""
import atexit
import json
import os
import sys

_DIR = os.environ.get("TT_BIO_D56_PROBE_DIR")

if _DIR:

    @atexit.register
    def _witness():
        rec = {"pid": os.getpid(), "argv": sys.argv[:4]}
        ag = sys.modules.get("tt_bio.autograd")
        rec["imported"] = ag is not None
        if ag is not None:
            rec["flag"] = bool(getattr(ag, "SOFTMAX_BW_RENORM", None))
            rec["stats"] = dict(getattr(ag, "SOFTMAX_BW_RENORM_STATS", {}))
        rec["taped_ttnn_imported"] = "tt_bio.taped_ttnn" in sys.modules
        try:
            os.makedirs(_DIR, exist_ok=True)
            with open(os.path.join(_DIR, "%d.json" % os.getpid()), "w") as f:
                json.dump(rec, f)
        except Exception:
            pass
