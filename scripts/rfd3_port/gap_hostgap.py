#!/usr/bin/env python3
"""RFD3 drained-device gaps: where the card sits idle because the host is working after a blocking read.

    gap_hostgap.py ARM OUT.json [STEPS] [--fixture=perf/dsfix/fixtures/rfd3_R4.json] [--fast=1]

ARM is gap_screen.py's '+'-joined knob set (absolute: every knob not named is off). One warm design, then one timed
design of STEPS steps. Every ttnn.to_torch drains the queue, so from its return until the next ttnn op is enqueued
the card has nothing to run: that window is device idle by construction, a lower bound on the step's idle time
(the rest is the host enqueueing slower than the card executes, which Tracy's busy share measures). Each window is
keyed by the read that opened it and the first op that closed it (file:line in tt_bio), summed per step. Also times
every host-only call the trace plan names (atom and DiT neighbour index, distogram bins, scatter_mean) and the
step's waits on the worker-thread index (Future.result).
"""
import collections
import json
import os
import pathlib
import shutil
import sys
import time
import traceback

from gap_knobs import apply                                               # noqa: E402
_arm = apply(sys.argv[1])

sys.path.insert(0, os.getcwd())
import ttnn                                                               # noqa: E402
import ttnn.decorators as D                                               # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                             # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                               # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ARM, OUT = args[0], pathlib.Path(args[1])
STEPS = int(args[2]) if len(args) > 2 else 60
FIXTURE = opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json")
FAST = opts.get("fast", "0") == "1"

ST = {"on": False, "open": None, "depth": 0}
GAPS = collections.defaultdict(lambda: [0.0, 0])     # (read site, next op site) -> [s, n]
HOST = collections.defaultdict(lambda: [0.0, 0])     # host-only function -> [s, n]
READS = collections.defaultdict(lambda: [0.0, 0])    # read site -> [s blocked in to_torch, n]


def site():
    for fr in reversed(traceback.extract_stack()[:-2]):
        if "/tt_bio/" in fr.filename and "/ttnn/" not in fr.filename:
            return "%s:%d" % (fr.filename.split("/tt_bio/")[-1], fr.lineno)
    return "?"


def close():
    if ST["open"] is not None:
        t0, where = ST["open"]
        g = GAPS[(where, site())]
        g[0] += time.perf_counter() - t0
        g[1] += 1
        ST["open"] = None


def hook(orig):
    def call(self, *a, **k):
        if ST["on"] and not ST["depth"]:
            close()
        ST["depth"] += 1
        try:
            return orig(self, *a, **k)
        finally:
            ST["depth"] -= 1
    return call


D.FastOperation.__call__ = hook(D.FastOperation.__call__)
D.Operation.__call__ = hook(D.Operation.__call__)
_to_torch = ttnn.to_torch


def to_torch(t, *a, **k):
    if not ST["on"] or ST["depth"] or not isinstance(t, ttnn.Tensor) or t.storage_type() != ttnn.StorageType.DEVICE:
        return _to_torch(t, *a, **k)
    close()
    where = site()
    t0 = time.perf_counter()
    out = _to_torch(t, *a, **k)
    t1 = time.perf_counter()
    READS[where][0] += t1 - t0
    READS[where][1] += 1
    ST["open"] = (t1, where)
    return out


ttnn.to_torch = to_torch     # tt_bio calls it as ttnn.to_torch, so the module attribute is enough


def timed(owner, name):
    fn = getattr(owner, name)

    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if ST["on"]:
                h = HOST[name]
                h[0] += time.perf_counter() - t0
                h[1] += 1
    setattr(owner, name, w)


for _n in ("_create_attention_indices", "_neighbours_kdtree", "_scaled_distogram_bins", "_scatter_mean",
           "_dense_attention_mask"):
    if hasattr(M, _n):
        timed(M, _n)

import concurrent.futures as _cf                                          # noqa: E402
timed(_cf.Future, "result")   # the step waiting on the worker-thread index (RFD3_ATTN_THREAD): host-bound if large

WALL = []
_sample = RFD3Sampler.sample


def _timed(self, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, *a, **k)
    WALL.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def fold(tag):
    specs = json.loads(pathlib.Path(FIXTURE).read_text())
    out_dir = "/tmp/gap_hostgap_%s" % tag
    shutil.rmtree(out_dir, ignore_errors=True)
    WALL.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=1, fast=FAST, verbose=False)
    return WALL[0]


warm_s = fold("warm")
ST["on"] = True
wall_s = fold("timed")
ST["on"] = False
n = STEPS - 1                     # the sampler runs STEPS-1 denoiser steps
ms = lambda s: 1e3 * s / n        # noqa: E731
gaps = sorted(((k, v) for k, v in GAPS.items()), key=lambda kv: -kv[1][0])
idle = sum(v[0] for v in GAPS.values())
print("[hostgap] %s: wall %.2f s for %d steps = %.1f ms/step; drained idle %.1f ms/step" % (ARM, wall_s, n, ms(wall_s), ms(idle)))
for (r, nx), (s, c) in gaps[:25]:
    print("  %7.2f ms/step %5.1f/step  read %-34s -> %s" % (ms(s), c / n, r, nx))
for k, (s, c) in sorted(HOST.items(), key=lambda kv: -kv[1][0]):
    print("  host %-28s %7.2f ms/step %5.1f calls/step" % (k, ms(s), c / n))
for k, (s, c) in sorted(READS.items(), key=lambda kv: -kv[1][0]):
    print("  read %-34s %7.2f ms/step blocked %5.1f/step" % (k, ms(s), c / n))
rec = {"arm": ARM, "fast": FAST, "fixture": FIXTURE, "steps": STEPS, "warm_s": warm_s, "wall_s": wall_s,
       "wall_ms_per_step": ms(wall_s), "drained_idle_ms_per_step": ms(idle),
       "gaps": [{"read": r, "next": nx, "ms_per_step": ms(s), "per_step": c / n} for (r, nx), (s, c) in gaps],
       "host_calls": {k: {"ms_per_step": ms(s), "per_step": c / n} for k, (s, c) in HOST.items()},
       "reads": {k: {"blocked_ms_per_step": ms(s), "per_step": c / n} for k, (s, c) in READS.items()},
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=1))
print("wrote", OUT)
