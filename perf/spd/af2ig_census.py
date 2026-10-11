"""Where does an AF2-IG fold spend its time? Per-stage wall clock, then one cProfiled fold.

Loads the model the way the worker does (`load_af2_device_model(..., template=True)`), folds one
fixture `--warm` times untimed, then `--reps` times with every stage of `AF2Model.forward`
wrapped in a timer that synchronises the device on exit, so each stage's number includes the
device work it queued. A last fold runs unwrapped under cProfile for the host-side breakdown
(ttnn call count and the Python cost around it). AICLK is sampled from sysfs during the reps.

    TT_VISIBLE_DEVICES=<card> PYTHONPATH=. python perf/spd/af2ig_census.py \\
        --input perf/kitcmp/fixtures/af2ig_pdl1_l80_s0.yaml --params ~/.boltz/af2/params/params_model_1_ptm.npz \\
        --out out/af2ig_census
"""
from __future__ import annotations

import argparse
import cProfile
import glob
import hashlib
import io
import json
import pstats
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--params", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--warm", type=int, default=2)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--trace", action="store_true",
               help="replay the two device stacks from a trace (AF2DeviceModel.trace_stacks)")
p.add_argument("--trace-stacks", default="",
               help="comma list of stacks to trace (extra_msa,evoformer); default both")
p.add_argument("--trace-blocks", type=int, default=None, help="trace only the first N Evoformer blocks")
p.add_argument("--trace-blocking", action="store_true", help="execute the replay blocking")
p.add_argument("--trimul-dram", action="store_true",
               help="keep every trimul tensor in DRAM (no L1 residency), for the trace-hang bisection")
p.add_argument("--ops", action="store_true",
               help="end with one fold timed per ttnn call between two syncs (perf/hallgrad/census.py's "
                    "Recorder), aggregated by op, call site and shapes into census.json 'ops'")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio.af2 import load_af2_device_model  # noqa: E402
from tt_bio.af2_weights import load_af2_state_dict  # noqa: E402
from tt_bio.af2ig import fold, token_count  # noqa: E402
from tt_bio.af2ig_input import read_af2ig_input  # noqa: E402

if a.trace:
    from tt_bio.tenstorrent import get_device
    get_device(trace="af2")
model = load_af2_device_model(load_af2_state_dict(a.params), template=True).eval()
model.trace_stacks = (set(a.trace_stacks.split(",")) if a.trace and a.trace_stacks else a.trace)
model.trace_blocks, model.trace_blocking = a.trace_blocks, a.trace_blocking
if a.trimul_dram:
    from tt_bio import tenstorrent as _tt
    _tt._TRIMUL_DRAM_SHAPES.update(range(1, 4097))
if a.trace:  # say where a hang sits: capture, first replay or a later one
    _replay = model._replay

    def _logged(name, run, *inputs):
        key = (name, *(tuple(t.shape) for t in inputs))
        state = ("eager" if key not in model._traces else
                 "capture" if model._traces[key] is None else "replay")
        print(f"trace {name} {state} t={time.time():.1f}", flush=True)
        out = _replay(name, run, *inputs)
        ttnn.synchronize_device(model._device)
        print(f"trace {name} done t={time.time():.1f}", flush=True)
        return out
    model._replay = _logged
spec = read_af2ig_input(a.input)
dev = model._device

T: dict[str, list[float]] = defaultdict(list)


def timed(name, fn):
    def wrapper(*args, **kw):
        t0 = time.perf_counter()
        r = fn(*args, **kw)
        ttnn.synchronize_device(dev)
        T[name].append(time.perf_counter() - t0)
        return r
    return wrapper


STAGES = ("template_embedding", "extra_msa_stack", "evoformer_stack", "single_activations",
          "structure", "relative_encoding")

clocks: list[int] = []
stop = threading.Event()


def sample():
    nodes = glob.glob("/sys/class/tenstorrent/tenstorrent!*/tt_aiclk")
    while not stop.is_set():
        for n in nodes:
            try:
                clocks.append((n.split("!")[1].split("/")[0], int(open(n).read().strip(), 0)))
            except (OSError, ValueError):
                pass
        time.sleep(0.25)


for _ in range(a.warm):
    fold(model, spec)
ttnn.synchronize_device(dev)

plain, digests = [], []
for _ in range(a.reps):
    t0 = time.perf_counter()
    pred = fold(model, spec)
    ttnn.synchronize_device(dev)
    plain.append(time.perf_counter() - t0)
    digests.append(hashlib.sha256(np.ascontiguousarray(pred.coords).tobytes()).hexdigest()[:16])

# A stage is a bound method or a submodule; a submodule is timed through its forward, because
# nn.Module refuses a plain function in its place.
def target(s):
    o = getattr(model, s)
    return (o, "forward") if isinstance(o, torch.nn.Module) else (model, s)


saved = {s: getattr(*target(s)) for s in STAGES}
for s in STAGES:
    setattr(*target(s), timed(s, saved[s]))
th = threading.Thread(target=sample, daemon=True)
th.start()
staged = []
for _ in range(a.reps):
    t0 = time.perf_counter()
    fold(model, spec)
    ttnn.synchronize_device(dev)
    staged.append(time.perf_counter() - t0)
stop.set()
th.join()
for s in STAGES:
    obj, attr = target(s)
    if obj is model:
        setattr(model, s, saved[s])
    else:
        del obj.forward  # back to the class's forward

per_fold = {k: sum(v) / a.reps for k, v in T.items()}
per_fold["rest (embeddings, recycle, heads, confidence, host glue)"] = (
    statistics.median(staged) - sum(per_fold.values()))

prof = cProfile.Profile()
t0 = time.perf_counter()
prof.enable()
fold(model, spec)
ttnn.synchronize_device(dev)
prof.disable()
prof_s = time.perf_counter() - t0
buf = io.StringIO()
st = pstats.Stats(prof, stream=buf)
st.sort_stats("tottime").print_stats(40)
ttnn_calls = sum(v[1] for k, v in st.stats.items() if "ttnn" in k[0] or "ttnn" in str(k[2]))
by_node = defaultdict(list)
for n, c in clocks:
    by_node[n].append(c)
res = {"input": a.input, "tokens": token_count(spec), "trace": a.trace, "coords_sha16": digests,
       "plain_fold_s": plain, "staged_fold_s": staged, "per_fold_stage_s": per_fold,
       "calls_per_fold": {k: len(v) // a.reps for k, v in T.items()},
       "profiled_fold_s": prof_s, "ttnn_frames_calls": ttnn_calls,
       "aiclk": {n: {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}
                 for n, v in by_node.items()}}
if a.ops and not a.trace:  # a synced call inside a capture is exactly what a trace refuses
    from perf.hallgrad.census import CLASS, Recorder
    class SiteRecorder(Recorder):
        def _origin(self):  # no autograd here: name the innermost tt_bio frame and the AF2 block
            f, inner, block = sys._getframe(2), None, None
            while f is not None and block is None:
                fn = f.f_code.co_filename
                if "/tt_bio/" in fn:
                    site = f"{Path(fn).name}:{f.f_code.co_name}"
                    inner = inner or site
                    if fn.endswith(("/af2.py", "/af2_reference.py")):
                        block = site
                f = f.f_back
            return f"{block} > {inner}" if block and block != inner else (inner or "harness")

    rec = SiteRecorder(ttnn, dev, 0)
    rec.rec = []
    rec.install_all()
    rec.active = True
    t0 = time.perf_counter()
    fold(model, spec)
    ttnn.synchronize_device(dev)
    rec.active = False
    agg: dict = defaultdict(lambda: {"calls": 0, "synced_s": 0.0})
    for r in rec.rec:
        d = agg[(r["name"], r["origin"], r["sig"])]
        d["calls"] += 1
        d["synced_s"] += r["synced_s"]
    rows = sorted(({"name": n, "origin": o, "sig": g, "cls": CLASS.get(n, "other"), **d}
                   for (n, o, g), d in agg.items()), key=lambda r: -r["synced_s"])
    by_cls: dict = defaultdict(float)
    for r in rows:
        by_cls[r["cls"]] += r["synced_s"]
    res["ops"] = {"fold_s": time.perf_counter() - t0, "calls": len(rec.rec),
                  "synced_total_s": sum(by_cls.values()), "by_class": dict(by_cls), "top": rows[:80]}
(a.out / "census.json").write_text(json.dumps(res, indent=1))
(a.out / "profile.txt").write_text(buf.getvalue())
st.dump_stats(str(a.out / "profile.pstats"))
print(json.dumps(res, indent=1))
print(buf.getvalue()[:6000])
