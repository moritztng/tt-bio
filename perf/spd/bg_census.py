"""bg_census.py DESIGN_ARGS... : device kernel time per ttnn op of the BoltzGen diffusion step, from the device profiler.

Runs `tt-bio design` in-process with every ttnn operation hooked, and arms the hook for a window of diffusion steps
inside a warm design (default steps 600-619 of the run = design 2, steps 100-119). Each op call's device operation
ids are matched to the profiler's programs, so every kernel nanosecond lands on (op, tt_bio call site, block). Needs a
profiler build of tt-metal and TT_METAL_DEVICE_PROFILER=1 (perf/spd_census/run.sh sets up the same environment).

Writes $BGCENSUS_OUT (json): window, per-step device kernel ns, and rows [op, site, block, calls, kernel_ns] sorted
by kernel time. Blocks are the shared Diffusion module's (encoder, token_transformer, decoder, attn, trans, adaln).

    TT_METAL_DEVICE_PROFILER=1 BGCENSUS_OUT=out.json python perf/spd/bg_census.py design FIXTURE ... --num_designs 2
"""
import collections
import json
import os
import runpy
import sys

import ttnn
import ttnn.decorators as D

from tt_bio import tenstorrent as T

OUT = os.environ.get("BGCENSUS_OUT", "bg_census.json")
S0 = int(os.environ.get("BGCENSUS_FROM", "600"))
NSTEP = int(os.environ.get("BGCENSUS_STEPS", "20"))
OPID = ttnn._ttnn.get_device_operation_id
ST = {"step": -1, "on": False, "depth": 0, "done": False, "flushed": 0}
KERN = {}                   # device operation id -> kernel ns (a mesh program's runtime id is (op id << 10) | sub)
REG = []
CALLS = []                  # (id0, id1, op, site, block)
NAMES = {}


def site():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn and "perf/spd" not in fn:
            return f"{fn.split('/tt_bio/')[1]}:{f.f_lineno}:{f.f_code.co_name}"
        f = f.f_back
    return "?"


def hook(orig):
    def call(self, *a, **k):
        if not ST["on"] or ST["depth"]:
            return orig(self, *a, **k)
        ST["depth"] += 1
        try:
            id0 = OPID()
            out = orig(self, *a, **k)
            id1 = OPID()
            if id1 > id0:
                CALLS.append((id0, id1, self.python_fully_qualified_name, site(), "/".join(REG)))
            if id1 - ST["flushed"] >= 1000:      # the device profiler buffer holds a few thousand programs
                drain()
            return out
        finally:
            ST["depth"] -= 1
    return call


D.FastOperation.__call__ = hook(D.FastOperation.__call__)
D.Operation.__call__ = hook(D.Operation.__call__)


def region(cls, label):
    fn = cls.__dict__["__call__"]

    def w(self, *a, **k):
        REG.append(NAMES.get(id(self), label))
        try:
            return fn(self, *a, **k)
        finally:
            REG.pop()
    cls.__call__ = w


for cls, label in ((T.DiffusionTransformer, "dt"), (T.AttentionPairBias, "attn"),
                   (T.ConditionedTransitionBlock, "trans"), (T.AdaLN, "adaln")):
    region(cls, label)


def drain(keep=True):
    dev = T.get_device()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    ST["flushed"] = OPID()
    for _chip, progs in ttnn.get_latest_programs_perf_data().items():
        for p in progs:
            r = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
            if keep and r is not None:
                i = p.program_execution_uid.runtime_id >> 10
                KERN[i] = KERN.get(i, 0) + r.duration


def finish():
    drain()
    kern = KERN
    agg = collections.defaultdict(lambda: [0, 0])
    total = missing = 0
    for id0, id1, op, st, blk in CALLS:
        ns = sum(kern.get(i, 0) for i in range(id0, id1))
        missing += sum(1 for i in range(id0, id1) if i not in kern)
        agg[(op, st, blk)][0] += 1
        agg[(op, st, blk)][1] += ns
        total += ns
    rows = sorted(([*key, c, ns] for key, (c, ns) in agg.items()), key=lambda r: -r[4])
    json.dump(dict(window=[S0, S0 + NSTEP], steps=NSTEP, calls=len(CALLS), ids_missing=missing,
                   kernel_ns_per_step=total / NSTEP, rows=rows), open(OUT, "w"), indent=0)
    print(f"BGCENSUS {OUT}: {len(CALLS)} calls, {total / NSTEP / 1e6:.3f} ms device kernel per step, "
          f"{missing} ids without a program", flush=True)


_diff_call = T.Diffusion.__dict__["__call__"]


def _diffusion(self, *a, **kw):
    for attr in ("encoder", "decoder", "token_transformer"):
        if getattr(self, attr, None) is not None:
            NAMES[id(getattr(self, attr))] = attr
    ST["step"] += 1
    if ST["step"] == S0:
        drain(keep=False)                              # drop everything before the window
        ST["on"] = True
    if ST["step"] == S0 + NSTEP and not ST["done"]:
        ST["on"] = False
        ST["done"] = True
        finish()
    return _diff_call(self, *a, **kw)


T.Diffusion.__call__ = _diffusion
sys.argv = ["tt_bio.main"] + sys.argv[1:]
runpy.run_module("tt_bio.main", run_name="__main__")
