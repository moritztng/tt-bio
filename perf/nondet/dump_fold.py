#!/usr/bin/env python3
"""nondet_dump_fold.py <target.yaml> <out_dir> — one protenix-v2 solo fold with full
intermediate capture for run-to-run divergence localization.

Captures into <out_dir>/dump/:
  cond_*.pt        trunk conditioning dict passed to edm_sample (per key)
  noise.pt         the initial noise frame (edm step -1)
  x_step####.pt    per-step coordinate frames (200 + final)
  cif sha written to <out_dir>/dump/SHA.txt
"""
import os
import pathlib
import sys

import torch

import tt_bio.protenix as P

yaml_path, out_dir = sys.argv[1], sys.argv[2]
dump_dir = pathlib.Path(out_dir) / "dump"
dump_dir.mkdir(parents=True, exist_ok=True)

_orig_edm = P.edm_sample


def _to_host(v):
    import ttnn

    if isinstance(v, ttnn.Tensor):
        return ttnn.to_torch(v)
    return v


def wrapped_edm(diffusion, cond, n_atoms, **kw):
    for k, v in sorted(cond.items()):
        try:
            torch.save(_to_host(v), dump_dir / f"cond_{k}.pt")
        except Exception as e:  # non-tensor entries: record the type
            (dump_dir / f"cond_{k}.txt").write_text(f"{type(v)}: {e}")

    def dump_fn(step, x):
        name = "noise.pt" if step == -1 else f"x_step{step:04d}.pt"
        torch.save(x.detach().cpu().clone(), dump_dir / name)

    kw["dump_fn"] = dump_fn
    return _orig_edm(diffusion, cond, n_atoms, **kw)


P.edm_sample = wrapped_edm

sys.argv = [
    "tt-bio",
    "predict",
    yaml_path,
    "--model",
    "protenix-v2",
    "--single_sequence",
    "--seed",
    "0",
    "--out_dir",
    out_dir,
]
from tt_bio.main import cli

cli()
