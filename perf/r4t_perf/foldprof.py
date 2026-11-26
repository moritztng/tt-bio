#!/usr/bin/env python3
"""Split trunk_step's fold VJP (Folds.backward) into device->host readback and torch's backward through the folds.

    foldprof.py [--rm] -- <trunk_step.py args>

Wraps `Folds.backward` in-process (no production edit) and prints one JSON line per call after trunk_step's own
lines: n grads, elements, readback seconds (ttnn.to_torch), cast/reshape seconds, autograd seconds.
--rm: read each gradient back after an on-card `ttnn.to_layout(ROW_MAJOR)` (untilize is exact, so the master
digest must equal the default arm's; that equality is the arm's gate).
"""
import json, os, runpy, sys, time

rm = "--rm" in sys.argv
argv = sys.argv[sys.argv.index("--") + 1:]
step = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4t_step", "trunk_step.py")

import torch
import ttnn
from tt_bio.train import folds as F


def backward(self, grads):
    t0 = time.perf_counter(); srcs, gs, n_el, t_rb = [], [], 0, 0.0
    for h, g in grads:
        s = self.source(h)
        if s is None:
            raise KeyError(f"no fold source for a device weight of shape {tuple(h.shape)}")
        if not torch.is_tensor(g):
            a = time.perf_counter()
            if rm and g.layout != ttnn.ROW_MAJOR_LAYOUT:
                g = ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT)
            g = ttnn.to_torch(g)
            t_rb += time.perf_counter() - a
        n_el += g.numel()
        srcs.append(s); gs.append(g.to(s.dtype).reshape(s.shape))
    t1 = time.perf_counter()
    if srcs:
        torch.autograd.backward(srcs, gs, retain_graph=True)
    t2 = time.perf_counter()
    print(json.dumps(dict(foldprof=1, rm=rm, n=len(gs), elems=n_el, readback_s=round(t_rb, 4),
                          cast_s=round(t1 - t0 - t_rb, 4), autograd_s=round(t2 - t1, 4), total_s=round(t2 - t0, 4))), flush=True)


F.Folds.backward = backward
sys.argv = [step] + argv
runpy.run_path(step, run_name="__main__")
