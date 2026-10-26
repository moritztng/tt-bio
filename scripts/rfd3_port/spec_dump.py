#!/usr/bin/env python3
"""Dump one page design's per-step coordinates for the speculative atom index study.

    python scripts/rfd3_port/spec_dump.py OUT.npz [--fast] [--steps 200]

Runs the design_cell rfd3 fixture (rfd3_R4.json, seed 42, batch 1) once and records, per denoiser call: the
coordinates the atom index is built from (X_noisy_L), the output of every recycle (X_L after recycle 0, 1, ...) and
t_hat. The next step's X_noisy differs from what recycle 0 predicts only by step_scale * d_t / t_hat * (X_r1 - X_r0),
which is what decides whether a candidate set built during recycle 1 can be certified exact.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--steps", type=int, default=200)
a = ap.parse_args()

from tt_bio.main import ensure_rfd3_weights
from tt_bio.rfd3 import design as D
from tt_bio.rfd3 import model as M

rec = {"x_noisy": [], "t_hat": [], "x_rec": []}
call, fwr = M.RFD3DiffusionModule.__call__, M.RFD3DiffusionModule._forward_with_recycle


def call_rec(self, X_noisy_L, t, f, *x, **k):
    rec["x_noisy"].append(X_noisy_L.detach().float().cpu().numpy()[0])
    rec["t_hat"].append(float(t.reshape(-1)[0]))
    rec["x_rec"].append([])
    return call(self, X_noisy_L, t, f, *x, **k)


def process_rec(self, *x, **k):
    out = proc(self, *x, **k)
    rec["x_rec"][-1].append(out["X_L"].detach().float().cpu().numpy()[0])
    return out


proc = M.RFD3DiffusionModule._process_
M.RFD3DiffusionModule.__call__ = call_rec
M.RFD3DiffusionModule._process_ = process_rec
specs = json.loads((TREE / "perf/dsfix/fixtures/rfd3_R4.json").read_text())
ckpt = str(ensure_rfd3_weights(Path.home() / ".boltz"))
t0 = time.time()
with torch.no_grad():
    D.run_design(specs, Path(a.out).with_suffix(".d"), checkpoint_dir=ckpt, from_pdb=True, num_timesteps=a.steps,
                 seed=42, num_designs=1, batch_size=1, fast=a.fast, verbose=False)
n_rec = {len(r) for r in rec["x_rec"]}
assert len(n_rec) == 1, n_rec
np.savez_compressed(a.out, x_noisy=np.stack(rec["x_noisy"]), t_hat=np.array(rec["t_hat"]),
                    x_rec=np.stack([np.stack(r) for r in rec["x_rec"]]))
print(json.dumps({"calls": len(rec["t_hat"]), "recycles": n_rec.pop(), "atoms": rec["x_noisy"][0].shape[0],
                  "wall_s": round(time.time() - t0, 1)}))
