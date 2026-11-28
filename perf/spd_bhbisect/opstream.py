#!/usr/bin/env python3
"""Load-independent comparison of two trees: the ttnn op stream one xmodel_ab run dispatches.

A wall read on a shared host at load/core > 1 is not a measurement. This wraps every ttnn operation call and records,
for the whole run (cold + warm folds): calls per op name, a running digest over (name, tensor-argument shapes and
dtypes) in dispatch order, and main-thread CPU seconds inside and outside ttnn calls. Two trees with the same digest
and the same CIF send the device the same work in the same order; the CPU split then says whether the host glue
between ops changed.

    python perf/spd_bhbisect/opstream.py OUT.json perf/of3_4xpd/xmodel_ab.py <its args...>
"""
import hashlib, importlib.metadata as im, json, runpy, sys, time
from collections import Counter
from pathlib import Path

out, sys.argv = Path(sys.argv[1]), sys.argv[2:]
import ttnn
from ttnn import decorators as D

calls, digest, state = Counter(), hashlib.sha256(), {"depth": 0, "in": 0.0}


def sig(x):
    if isinstance(x, ttnn.Tensor):
        return f"{tuple(x.shape)}{x.dtype}"
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(sig(y) for y in x) + "]"
    return ""


def wrap(cls):
    inner = cls.__call__

    def call(self, *a, **k):
        if state["depth"]:
            return inner(self, *a, **k)
        name = getattr(self, "python_fully_qualified_name", None) or type(self).__name__
        calls[name] += 1
        digest.update(f"{name}|{sig(a)}|{sig(list(k.values()))}\n".encode())
        state["depth"], t = 1, time.thread_time()
        try:
            return inner(self, *a, **k)
        finally:
            state["depth"] = 0
            state["in"] += time.thread_time() - t

    cls.__call__ = call


for c in {getattr(D, n) for n in ("FastOperation", "Operation") if hasattr(D, n)}:
    wrap(c)
t0 = time.thread_time()
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    main = time.thread_time() - t0
    out.write_text(json.dumps({
        "ops_total": sum(calls.values()), "stream_sha": digest.hexdigest()[:16], "calls": dict(sorted(calls.items())),
        "main_cpu_in_ttnn_s": round(state["in"], 2), "main_cpu_outside_ttnn_s": round(main - state["in"], 2),
        "process_cpu_s": round(time.process_time(), 1), "loadavg": open("/proc/loadavg").read().split()[:3],
        "versions": {p: im.version(p) for p in ("ttnn", "torch", "numpy")}}, indent=1))
