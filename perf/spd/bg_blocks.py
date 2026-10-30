"""bg_blocks.py DESIGN_ARGS... : `tt-bio design` with a device-synced wall clock on every block of the diffusion step.

BoltzGen at 512 aa on a p150a spends ~23 s per design in 500 diffusion steps (45 ms/step) against 21.7 s for a whole
design on a stock A100, so the step is where the gap lives. This splits each step into the shared `Diffusion` module's
blocks (conditioner, atom encoder, token transformer and its attention / transition halves, atom decoder) so the
45 ms has owners before anyone reaches for Tracy.

Every timed call is bracketed by synchronize_device, which removes the pipelining across block boundaries; the totals
are an upper bound per block and the step itself reads slower than in an unwrapped run. Labels nest: a call made
inside another timed block is keyed "outer/inner". One line per design on stdout, `BGBLOCKS {json}`, with
label -> [seconds, calls] (also appended to $BGBLOCKS_OUT when set), printed when a design's diffusion ends (the next non-diffusion op after a run of steps).

    python perf/spd/bg_blocks.py design FIXTURE --protocol ... --num_designs 3 --out_dir OUT
"""
import collections
import json
import os
import runpy
import sys
import time

import ttnn

from tt_bio import tenstorrent as T

ACC = collections.defaultdict(float)
CNT = collections.Counter()
STACK = []
NAMES = {}


def _sync():
    try:
        ttnn.synchronize_device(T.get_device())
    except Exception:
        pass


def flush():
    if ACC:
        line = "BGBLOCKS " + json.dumps({k: [round(v, 3), CNT[k]] for k, v in sorted(ACC.items())})
        print(line, flush=True)
        if os.environ.get("BGBLOCKS_OUT"):
            with open(os.environ["BGBLOCKS_OUT"], "a") as f:
                f.write(line + "\n")
        ACC.clear()
        CNT.clear()


def timed(cls, label):
    fn = cls.__dict__["__call__"]

    def wrapper(self, *a, **kw):
        name = NAMES.get(id(self), label)
        _sync()
        t0 = time.monotonic()
        STACK.append(name)
        key = "/".join(STACK)
        try:
            return fn(self, *a, **kw)
        finally:
            STACK.pop()
            _sync()
            ACC[key] += time.monotonic() - t0
            CNT[key] += 1

    cls.__call__ = wrapper


_diff_call = T.Diffusion.__dict__["__call__"]


def _diffusion(self, *a, **kw):
    for attr in ("encoder", "decoder", "token_transformer"):
        m = getattr(self, attr, None)
        if m is not None:
            NAMES[id(m)] = attr
    for attr in ("conditioner_transition_0", "conditioner_transition_1"):
        m = getattr(self, attr, None)
        if m is not None:
            NAMES[id(m)] = "cond"
    if CNT["step"] >= 500:
        flush()
    _sync()
    t0 = time.monotonic()
    STACK.append("step")
    try:
        return _diff_call(self, *a, **kw)
    finally:
        STACK.pop()
        _sync()
        ACC["step"] += time.monotonic() - t0
        CNT["step"] += 1


T.Diffusion.__call__ = _diffusion
timed(T.DiffusionTransformer, "dt")
timed(T.DiffusionTransformerLayer, "layer")
timed(T.AttentionPairBias, "attn")
timed(T.ConditionedTransitionBlock, "trans")
timed(T.AdaLN, "adaln")
if hasattr(T, "Transition"):
    timed(T.Transition, "transition")

sys.argv = ["tt_bio.main"] + sys.argv[1:]
try:
    runpy.run_module("tt_bio.main", run_name="__main__")
finally:
    flush()
