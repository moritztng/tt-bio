"""Split a cold `tt-bio predict` model load into the parts a disk tile cache could remove.

Runs perf/spd/bench.py unchanged (same CLI config, same `Worker.load_model`, same `predict_one`) with timers
around the load's moving parts, and appends one JSON record per phase to <out>/split.jsonl:

    import    process start -> torch, ttnn, tt_bio imported (python import cost, paid by every call)
    load      the model build: ckpt_s (torch.load / safetensors), tile_s (host tiling inside Module.torch_to_tt,
              transform included), upload_s (its to_device), other_ft_s / other_ft_n (ttnn.from_torch calls
              outside torch_to_tt: weights a module derives and pushes itself, which no torch_to_tt cache sees),
              rest_s (python construction and torch work around them)
    rep       per fold rep the same counters (inputs and derived tensors pushed during the fold)

Same arguments as bench.py:

    python perf/spd_coldstart/split.py --out RUN/b2 --chip 3 --model boltz2 --inputs u100 --warm 1 --samples 1

Timing the host tile and the upload apart calls from_torch on host and then to_device, which is what
torch_to_tt's cache path already does (bit-exact by construction); the rep's digest in bench.jsonl is the check.
"""
import json
import os
import runpy
import sys
import time
from collections import defaultdict
from pathlib import Path

T_START = time.monotonic()
with open("/proc/self/stat") as f:
    _ticks = int(f.read().rsplit(")", 1)[1].split()[19])
with open("/proc/uptime") as f:
    AGE0 = float(f.read().split()[0]) - _ticks / os.sysconf("SC_CLK_TCK")  # process age when this line ran

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "spd" / "bench.py"
sys.path.insert(0, str(HERE.parents[1]))
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
REC = open(out / "split.jsonl", "a", buffering=1)


def rec(**kw):
    REC.write(json.dumps(dict(t=time.time(), **kw)) + "\n")


imp = {}
t = time.monotonic(); import torch; imp["torch_s"] = time.monotonic() - t  # noqa: E401,E702
t = time.monotonic(); import tt_bio.main; imp["tt_bio_main_s"] = time.monotonic() - t  # noqa: E401,E702,F401
t = time.monotonic(); import ttnn; imp["ttnn_s"] = time.monotonic() - t  # noqa: E401,E702  (after main: TT_METAL_CACHE)
t = time.monotonic()
import tt_bio.tenstorrent as T  # noqa: E402
import tt_bio.worker as W  # noqa: E402
imp["tt_bio_rest_s"] = time.monotonic() - t
rec(ev="import", age_at_start_s=AGE0, total_s=AGE0 + time.monotonic() - T_START, **imp)

acc = defaultdict(float)
_in_t2t = [0]


def _timed(name, fn):
    def w(*a, **kw):
        t = time.monotonic()
        try:
            return fn(*a, **kw)
        finally:
            acc[name + "_s"] += time.monotonic() - t
            acc[name + "_n"] += 1
    return w


torch.load = _timed("ckpt", torch.load)
try:
    import safetensors.torch as _st
    _st.load_file = _timed("ckpt", _st.load_file)
except ImportError:
    pass

_ft = ttnn.from_torch


def _from_torch(*a, **kw):
    if _in_t2t[0]:
        return _ft(*a, **kw)
    t = time.monotonic()
    try:
        return _ft(*a, **kw)
    finally:
        acc["other_ft_s"] += time.monotonic() - t
        acc["other_ft_n"] += 1


ttnn.from_torch = _from_torch
_t2t = T.Module.torch_to_tt


def torch_to_tt(self, key, transform=lambda x: x.t(), dtype=None):
    _in_t2t[0] += 1
    try:
        if T._weight_cache is not None:
            t = time.monotonic()
            r = _t2t(self, key, transform, dtype)
            acc["cached_t2t_s"] += time.monotonic() - t
            acc["cached_t2t_n"] += 1
            return r
        if dtype is None:
            dtype = T._dtype(ttnn.bfloat16)
        t = time.monotonic()
        host = _ft(transform(self.weights[key]), layout=ttnn.TILE_LAYOUT, dtype=dtype)
        t1 = time.monotonic()
        dev = ttnn.to_device(host, self.device)
        t2 = time.monotonic()
        acc["tile_s"] += t1 - t
        acc["upload_s"] += t2 - t1
        acc["t2t_n"] += 1
        return dev
    finally:
        _in_t2t[0] -= 1


T.Module.torch_to_tt = torch_to_tt
_depth = [0]


def _phase(ev, fn):
    def w(*a, **kw):
        _depth[0] += 1
        before = dict(acc)
        t = time.monotonic()
        try:
            return fn(*a, **kw)
        finally:
            _depth[0] -= 1
            if _depth[0] == 0:
                s = time.monotonic() - t
                d = {k: round(v - before.get(k, 0.0), 4) for k, v in acc.items() if v != before.get(k, 0.0)}
                parts = sum(d.get(k, 0.0) for k in ("ckpt_s", "tile_s", "upload_s", "other_ft_s", "cached_t2t_s"))
                rec(ev=ev, s=round(s, 4), rest_s=round(s - parts, 4), **d)
    return w


W._WorkerState.load_model = _phase("load", W._WorkerState.load_model)
W._WorkerState.predict_one = _phase("rep", W._WorkerState.predict_one)
import tt_bio.protenix as P  # noqa: E402
P.Protenix.load_from_checkpoint = classmethod(_phase("load", P.Protenix.load_from_checkpoint.__func__))

sys.argv = [str(BENCH)] + sys.argv[1:]
runpy.run_path(str(BENCH), run_name="__main__")
