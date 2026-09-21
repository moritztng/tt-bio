"""Arms perf/of3t_auxgrad/exposure.py inside a spawned `tt-bio predict` worker.

`tt-bio predict` folds in an `mp.get_context("spawn")` child, which is a fresh interpreter: a
patch installed in the parent reaches nothing, and the first attempt at this measurement
produced a completed 89.8 s fold with an empty record (fleet memory
`in-process-patch-never-reaches-a-spawn-child`). `site` imports `sitecustomize` in EVERY
interpreter including that child, so this is the one place the hook can be armed from.

It cannot import tt_bio here: that would pull ttnn into every interpreter on the box and open a
device in processes that want none. Instead it wraps `__import__` and arms the patch the moment
both target modules are in `sys.modules`, then restores the original import.

Enabled only when EXPOSURE_HOOK=1, so having this directory on PYTHONPATH is inert by default.
"""
import os
import sys

if os.environ.get("EXPOSURE_HOOK") == "1":
    import builtins

    _orig = builtins.__import__
    # Attribute presence, not module presence. A module is in sys.modules from the moment its
    # import STARTS, so a membership test fires mid-initialisation and the patch then imports a
    # half-built module: measured here as "cannot import name ConfidenceHead from partially
    # initialized module tt_bio.protenix", which killed a fold at the confidence step.
    _need = (("tt_bio.openfold3_fold", "OpenFold3"),
             ("tt_bio.openfold3_confidence", "OF3ConfidenceHead"))

    def _ready():
        return all(hasattr(sys.modules.get(m), attr) for m, attr in _need)

    def _hooked(name, *a, **k):
        m = _orig(name, *a, **k)
        if _ready():
            builtins.__import__ = _orig
            sys.path.insert(0, os.environ["EXPOSURE_SRC"])
            import exposure
            exposure.install()
        return m

    builtins.__import__ = _hooked
