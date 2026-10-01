#!/usr/bin/env python3
"""Grade a no_grad-forward lever (default `ops.NOGRAD_IS_INFERENCE`) against float64: `perf/bcx_afgrad` `stack --ckpt`, by arm.

The block VJP grade (vjp_grade.py) tapes each block directly, so it never runs the checkpointed
no_grad forward this lever changes. `stack --ckpt` does: the whole 4 + 48 stack checkpointed per
block under one tape from sequence logits, its loss and dL/dlogits against float64 autograd on
the same stack, a float64 finite difference along the device gradient, repeat, zero-seed and
permuted controls. Under `bindcraft2.fast_round()` (the round's program, stage 1 included), with:
  off  ops.NOGRAD_IS_INFERENCE False: the untaped forward runs the taped program
  on   True: it runs the inference program; the backward's recompute is taped either way
"""
import argparse, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--arms", default="off,on")
ap.add_argument("--lever", default="tt_bio.ops:NOGRAD_IS_INFERENCE", metavar="MODULE:ATTR")
ap.add_argument("--stats", default="", metavar="MODULE:ATTR", help="a reach counter printed per arm")
a = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import importlib
from tt_bio import bindcraft2, ops
mod_name, attr = a.lever.split(":")
mod = importlib.import_module(mod_name)
tag = "nograd" if attr == "NOGRAD_IS_INFERENCE" else attr.lower()
if a.stats:
    _sm, _sa = a.stats.split(":")
    stats = lambda: list(getattr(importlib.import_module(_sm), _sa))  # noqa: E731
else:
    stats = lambda: None  # noqa: E731
ns = argparse.Namespace(params=A.DEFAULT_PARAMS, card=0, n=a.n, extra=4, evo=48, seed=0,
                        controls_only=False, eps="1e-1,3e-2,1e-2,3e-3,1e-3", ckpt=True,
                        msa_mask=False, threads=8)
stem = f"stack_n{a.n}_e4_v48_ckpt"
with bindcraft2.fast_round():
    for arm in a.arms.split(","):
        setattr(mod, attr, arm == "on")
        print(f"== arm {arm} {a.lever}={getattr(mod, attr)} reach before {stats()}", flush=True)
        A.cmd_stack(ns)
        print(f"== arm {arm} reach after {stats()}", flush=True)
        for ext in (".json", ".pt"):
            src = A.OUT / (stem + ext)
            src.rename(A.OUT / f"{stem}_bcp_evo_{tag}_{arm}{ext}")
        d = json.loads((A.OUT / f"{stem}_bcp_evo_{tag}_{arm}.json").read_text())
        print(f"== arm {arm} device_vs_f64 {d['device_vs_f64']} loss dev {d['device']['loss']} "
              f"f64 {d['f64']['loss']} fd {d.get('fd_best', {}).get('ratio_fd_over_device')}",
              flush=True)
