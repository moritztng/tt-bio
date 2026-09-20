#!/usr/bin/env python3
"""The fourth fix, verified on the SHIPPED default rather than on the function.

`of3t-traj20` settled the schedule family in closed form over the whole domain, but it did it
by calling `af3_lr` with `plateau_until` passed by hand. That proves the function carries both
families; it does not prove the recipe selects the right one. This reads the value out of
`train_loop`'s signature and runs the whole domain against upstream's own scheduler, executed.

The 20-step trajectory cannot see this and never could: both families are the same linear
warmup over 0..1000 and the first disagreement is at step 50,000.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch                                              # noqa: E402

from tt_bio.train.optim import af3_lr                     # noqa: E402
from tt_bio.train.recipes import train_loop               # noqa: E402

UP = "/home/ttuser/of3t_traj20/upstream/lr_schedulers.py"
OUT = "perf/of3t_traj20/schedule_family_shipped.json"
LR = 1.8e-3
N = 200_005
CFG = dict(base_lr=0.0, warmup_no_steps=1000, start_decay_after_n_steps=50000,
           decay_every_n_steps=50000, decay_factor=0.95)


def theirs(n):
    spec = importlib.util.spec_from_file_location("of3_lr_shipped", UP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros(1)
    opt = torch.optim.Adam([p], lr=LR)
    sch = mod.AlphaFoldLRScheduler(opt, last_epoch=-1, max_lr=LR, **CFG)
    out = []
    for _ in range(n):
        out.append(float(opt.param_groups[0]["lr"]))
        opt.step()
        sch.step()
    return out


def compare(ours, ref):
    bad = [(k, a, b) for k, (a, b) in enumerate(zip(ours, ref)) if a != b]
    worst_rel = max((abs(a - b) / max(abs(b), 1e-30) for _, a, b in bad), default=0.0)
    return {"points": len(ref), "exact": len(ref) - len(bad), "mismatches": len(bad),
            "first_mismatch": (None if not bad else
                               {"step": bad[0][0], "ours": bad[0][1], "theirs": bad[0][2]}),
            "worst_rel": worst_rel}


def main() -> int:
    sig = {k: v.default for k, v in inspect.signature(train_loop).parameters.items()}
    plateau, warm = sig["plateau_until"], CFG["warmup_no_steps"]
    ref = theirs(N)
    res = {
        "steps_compared": N, "lr": LR, "config": CFG,
        "train_loop_plateau_until_default": plateau,
        "as_the_shipped_recipe_now_builds_it": compare(
            [af3_lr(k, LR, warmup_steps=warm, plateau_until=plateau) for k in range(N)], ref),
        "with_plateau_until_None_the_pre_fix_call": compare(
            [af3_lr(k, LR, warmup_steps=warm) for k in range(N)], ref),
        "sha256_lr_schedulers": hashlib.sha256(open(UP, "rb").read()).hexdigest(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
