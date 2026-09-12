"""Seed every RNG, and optionally count `_bias_stack` calls, in this process and in every child.

Two jobs, both of which have to happen before user code runs and in EVERY process, which is what
makes sitecustomize the right place: `tt-bio predict` and the BoltzGen CLI both fan their work
into child processes, so a monkeypatch installed in the launcher is never the one that runs. A
first attempt did exactly that and read 0 calls on a Boltz-2 fold that certainly makes them.

  BG_AB_SEED        seed torch, numpy and random. BoltzGen has no --seed of its own.
  BG_COUNT_DIR      wrap `tt_bio.boltz2._bias_stack` and write <pid>.json there at exit.

Not shipped and not importable by the package: it exists only for `perf/b2z_levers`.
"""
import atexit
import json
import os
import pathlib
import random

import numpy as np
import torch

_s = int(os.environ.get("BG_AB_SEED", "1234"))
random.seed(_s)
np.random.seed(_s)
torch.manual_seed(_s)

_countdir = os.environ.get("BG_COUNT_DIR")
if _countdir:
    import tt_bio.boltz2 as _B2

    _calls = {"n": 0, "fused": 0}
    _orig = _B2._bias_stack

    def _counting(layers, x):
        _calls["n"] += 1
        if _B2._fuse_bias_stacks() and len(layers) > 1:
            _calls["fused"] += 1
        return _orig(layers, x)

    _B2._bias_stack = _counting

    def _dump():
        d = pathlib.Path(_countdir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{os.getpid()}.json").write_text(json.dumps(
            {"pid": os.getpid(), "argv0": os.environ.get("BG_LEG", ""), **_calls}))

    atexit.register(_dump)
