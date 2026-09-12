#!/usr/bin/env python3
"""Every fused-activation op a 512 aa Boltz-2 fold actually executes, with shape, dtype and count.

The static grep finds call sites. It cannot tell you which of them a fold reaches, how many times,
or at what shape -- Boltz-2 chunks its Transition by rows, caches one projection across diffusion
steps, and half the `activation=` sites in tt_bio belong to Protenix or OpenFold3. So this wraps
the ttnn entry points for the duration of one real fold and records what went through them.

Two classes are recorded, because they are two different mechanisms and only one is the one the
row is about:

  A  matmul-fused    ttnn.linear/matmul with `activation=` or a program config carrying
                     `fused_activation` -- an SFPU pass run inside the matmul kernel
  B  eltwise-fused   ttnn.add/multiply (and their in-place forms) with
                     input_tensor_{a,b}_activations -- an SFPU pass folded into a binary op that
                     is already reading both operands

Splitting A costs an L1 round trip and may still win (the Transition silu does, 1.4549x).
Splitting B costs an entire extra op over the same bytes and cannot win; B is censused so the
total is honest about what is and is not addressable.

    census.py --out <json> [--size 512] [--steps 200] [--recycles 3]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--size", default="512")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--recycles", type=int, default=3)
a = ap.parse_args()

import torch                                                                   # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                    # noqa: E402
import ab_flag_levers as AB                                                    # noqa: E402
import tt_bio as _TB                                                           # noqa: E402
assert Path(_TB.__file__).resolve().is_relative_to(REPO), _TB.__file__
from tt_bio.tenstorrent import get_device                                      # noqa: E402
from tt_bio.worker import _WorkerState, _ensure_local_artifacts               # noqa: E402
from tt_bio import esmfold2 as _E                                              # noqa: E402
_E.set_progress(lambda *aa, **kk: None)

AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = a.steps, a.recycles

# site key -> record. The key is the CALL SITE plus the shape, because one site with two shapes is
# two different levers (the Transition chunk runs at two row counts).
REC: dict[tuple, dict] = {}
PHASE = ["load"]


def _site(depth: int) -> str:
    """The first frame above the ttnn wrapper that lives in tt_bio -- i.e. the model's own line."""
    f = sys._getframe(depth)
    while f is not None:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn:
            return f"{Path(fn).name}:{f.f_lineno}"
        f = f.f_back
    return "?"


def _shape(t):
    try:
        return list(t.shape)
    except Exception:
        return None


def _log(cls, act, ins, out, site):
    key = (cls, site, act, tuple(tuple(s or ()) for s in ins), tuple(_shape(out) or ()),
           str(out.dtype).split(".")[-1])
    r = REC.get(key)
    if r is None:
        r = REC[key] = {"class": cls, "site": site, "activation": act,
                        "in_shapes": [list(s or ()) for s in ins], "out_shape": _shape(out),
                        "out_dtype": str(out.dtype).split(".")[-1], "calls": 0,
                        "phases": collections.Counter()}
    r["calls"] += 1
    r["phases"][PHASE[0]] += 1


def _actname(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "+".join(_actname(x) for x in v)
    return str(v).split(".")[-1]


_orig = {}
for name in ("linear", "matmul"):
    _orig[name] = getattr(ttnn, name)


def _wrap_mm(name):
    orig = _orig[name]

    def w(*args, **kw):
        out = orig(*args, **kw)
        act = kw.get("activation")
        pc = kw.get("program_config")
        if act is None and pc is not None:
            act = _actname(getattr(pc, "fused_activation", None))
        if act is not None:
            ins = [_shape(t) for t in args[:2]]
            _log("matmul", _actname(act), ins, out, _site(2))
        return out
    return w


for name in ("linear", "matmul"):
    setattr(ttnn, name, _wrap_mm(name))

for name in ("add", "add_", "multiply", "multiply_", "subtract", "subtract_"):
    if not hasattr(ttnn, name):
        continue
    _orig[name] = getattr(ttnn, name)

    def _wrap_bin(nm):
        orig = _orig[nm]

        def w(*args, **kw):
            out = orig(*args, **kw)
            acts = [_actname(kw.get(k)) for k in
                    ("input_tensor_a_activations", "input_tensor_b_activations", "activations")]
            acts = [x for x in acts if x]
            if acts:
                ins = [_shape(t) for t in args[:2]]
                _log("eltwise", "+".join(acts), ins, out, _site(2))
            return out
        return w
    setattr(ttnn, name, _wrap_bin(name))

dev = get_device()
g = dev.compute_with_storage_grid_size()
work = Path(tempfile.mkdtemp(prefix="b2z2-census-"))
struct_dir = work / "out"; struct_dir.mkdir(parents=True)
msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
name = f"cdk2x2_{a.size}"
AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)
cfg = AB.build_cfg(msa_dir, struct_dir)
_ensure_local_artifacts(cfg)
state = _WorkerState("tenstorrent")
state.load_model(cfg)
state.bind_run("b2z2-census", cfg)

REC.clear()
PHASE[0] = "fold"
t0 = time.perf_counter()
state.predict_one(AB.FIX / f"{name}.yaml", cfg)
wall = time.perf_counter() - t0

rows = []
for r in REC.values():
    r = dict(r)
    r["phases"] = dict(r["phases"])
    o = r["out_shape"] or []
    r["out_elems"] = int(torch.tensor(o).prod()) if o else 0
    r["out_bytes"] = r["out_elems"] * (4 if "float32" in r["out_dtype"] else 2)
    rows.append(r)
rows.sort(key=lambda r: -r["calls"] * r["out_elems"])

out = {"doc": __doc__,
       "env": {"host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
               "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
               "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
               "fixture": name, "steps": a.steps, "recycles": a.recycles,
               "fold_wall_s": round(wall, 3),
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
       "sites": rows,
       "totals": {
           "matmul_sites": sum(1 for r in rows if r["class"] == "matmul"),
           "matmul_calls": sum(r["calls"] for r in rows if r["class"] == "matmul"),
           "matmul_out_MB": round(sum(r["calls"] * r["out_bytes"] for r in rows
                                      if r["class"] == "matmul") / 1e6, 1),
           "eltwise_sites": sum(1 for r in rows if r["class"] == "eltwise"),
           "eltwise_calls": sum(r["calls"] for r in rows if r["class"] == "eltwise"),
           "eltwise_out_MB": round(sum(r["calls"] * r["out_bytes"] for r in rows
                                       if r["class"] == "eltwise") / 1e6, 1)}}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=1))
print(json.dumps(out["totals"], indent=1))
for r in rows:
    print(f"{r['class']:8s} {r['site']:22s} {r['activation'] or '-':10s} "
          f"x{r['calls']:<6d} out={r['out_shape']} {r['out_dtype']}")
