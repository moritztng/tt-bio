#!/usr/bin/env python3
"""Capture the operands `OF3DiffusionModule` is actually called with, from a real fold.

The census needs the module's real inputs. `~/of3_ref_out.pkl` carries them on the host
that captured them, but the copy on qb2 is a **trunk-only** capture -- seven keys, none of
them `diffusion_module_xlout_real` -- and regenerating the reference chain needs the
upstream tree and a CPU venv that are not on this box. Inventing the operands is not an
option: twelve of the twenty-nine are mask-derived gather indices and block masks, and a
wrong one silently changes which kernel the forward picks.

So take them from the shipped fold, which computes exactly the set the census is a
statement about. The patch itself lives in `_capture_hook/sitecustomize.py` rather than
here, because `tt_bio.main predict` folds in a SPAWNED CHILD: patching the class in this
process records nothing and the fold finishes clean, which is the quiet version of the
failure and is how the first attempt at this went.

Durable by construction (K61): the bundle lands in the worktree, not /tmp.

    python3 perf/of3t_diffusion/capture_operands.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", default="perf/of3t_diffusion/fixtures/ubq.yaml")
    p.add_argument("--out", default="perf/of3t_diffusion/operands_ubq.pt")
    p.add_argument("--model", default="openfold3")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()

    out = os.path.abspath(a.out)
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ)
    env["OF3T_CAPTURE_OUT"] = out
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(HERE, "_capture_hook"), ROOT, env.get("PYTHONPATH", "")])
    cmd = [a.python, "-u", "-m", "tt_bio.main", "predict", a.fixture,
           "--model", a.model, "--single_sequence", "--sampling_steps", "1",
           "--diffusion_samples", "1", "--seed", "0",
           "--out_dir", "/tmp/of3t/of3t-diffusion/capture_out"]
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT, env=env)

    if not os.path.exists(out):
        print("NO CAPTURE (fold rc=%d)" % rc)
        return 1
    import torch
    rec = torch.load(out, weights_only=False)
    print("captured %d operands -> %s" % (len(rec), out))
    for n, r in rec.items():
        if r["kind"] == "tensor":
            print("  %-24s %-24s %-18s %s" % (n, tuple(r["t"].shape), r["dtype"], r["layout"]))
        else:
            print("  %-24s %r" % (n, r["v"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
