"""Arm the ragged-tile-tail census in every process of a tt-bio run, workers included.

Put this directory first on PYTHONPATH and set TOKEN_AXIS_CENSUS_DIR; CPython imports
`sitecustomize` at interpreter startup, so a `tt-bio predict` worker subprocess gets armed too.
That is the whole point -- `predict`/`design` fork a per-card `mp.get_context("spawn")` worker, so
patching only the parent counted zero calls on a fold that made 1307 of them. Does nothing at all
when the env var is unset.

Each armed process also drops an empty `armed-<pid>` file next to its counters. A census that reads
zero and a census that never reached the model process look identical in the counters, and the
`armed-*` files are how you tell them apart.
"""
import os

if os.environ.get("TOKEN_AXIS_CENSUS_DIR"):
    import sys

    _here = os.path.abspath(__file__)          # <repo>/perf/bucketing_audit/censusenv/
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_here))))
    sys.path.insert(0, os.path.join(_repo, "tests"))
    import token_axis_probe

    token_axis_probe.enable()
    _d = os.environ["TOKEN_AXIS_CENSUS_DIR"]
    os.makedirs(_d, exist_ok=True)
    open(os.path.join(_d, "armed-%d" % os.getpid()), "w").close()
