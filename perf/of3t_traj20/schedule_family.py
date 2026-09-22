#!/usr/bin/env python3
"""The fourth wiring gap, found by reading and settled in closed form: `recipes.py` picks the
wrong schedule FAMILY, and the 20-step window cannot see it.

`af3_lr` carries both AlphaFold-family schedules and `plateau_until` is the whole of the
difference between them (`tt_bio/train/optim.py:af3_lr`). OpenFold3 ships the AF2 form:
`AlphaFoldLRScheduler` holds the rate flat at `max_lr` until `start_decay_after_n_steps` and
counts the decay exponent from there. `recipes.py:119` calls
`af3_lr(s, lr, warmup_steps=warmup_steps)` and passes no `plateau_until`, which selects
Protenix's form instead: no plateau, exponent counted from step zero.

Inside 20 steps both are the same linear warmup, so the trajectory is blind to it by
construction. That is not a reason to leave it unreported -- it is a reason to settle it the
way §4 settles a pure function of the step index, over the whole domain.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from tt_bio.train.optim import af3_lr  # noqa: E402

UP = "/home/ttuser/of3t_traj20/upstream/lr_schedulers.py"
OUT = "perf/of3t_traj20/schedule_family.json"
LR = 1.8e-3
CFG = dict(base_lr=0.0, warmup_no_steps=1000, start_decay_after_n_steps=50000,
           decay_every_n_steps=50000, decay_factor=0.95)


def theirs(n):
    spec = importlib.util.spec_from_file_location("of3_lr", UP)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    p = torch.nn.Parameter(torch.zeros(1)); p.grad = torch.zeros(1)
    opt = torch.optim.Adam([p], lr=LR)
    sch = mod.AlphaFoldLRScheduler(opt, last_epoch=-1, max_lr=LR, **CFG)
    out = []
    for _ in range(n):
        out.append(float(opt.param_groups[0]["lr"]))
        opt.step(); sch.step()
    return out


def main():
    n = 200005
    t = theirs(n)
    shipped = [af3_lr(k, LR, warmup_steps=CFG["warmup_no_steps"],
                      decay_every_n_steps=CFG["decay_every_n_steps"],
                      decay_factor=CFG["decay_factor"], base_lr=CFG["base_lr"])
               for k in range(n)]
    fixed = [af3_lr(k, LR, warmup_steps=CFG["warmup_no_steps"],
                    decay_every_n_steps=CFG["decay_every_n_steps"],
                    decay_factor=CFG["decay_factor"], base_lr=CFG["base_lr"],
                    plateau_until=CFG["start_decay_after_n_steps"]) for k in range(n)]

    def cmp(ours):
        bad = [k for k in range(n) if ours[k] != t[k]]
        return {"points": n, "exact": n - len(bad), "mismatches": len(bad),
                "first_mismatch": (None if not bad else
                                   {"step": bad[0], "ours": ours[bad[0]], "theirs": t[bad[0]]}),
                "worst_abs": max((abs(ours[k] - t[k]) for k in range(n)), default=0.0),
                "worst_rel": max((abs(ours[k] - t[k]) / t[k] for k in range(n) if t[k] > 0),
                                 default=0.0)}

    res = {"steps_compared": n, "lr": LR, "config": CFG,
           "shipped_recipes_py_no_plateau_until": cmp(shipped),
           "with_plateau_until": cmp(fixed),
           "inside_the_20_step_window": {
               "exact_over_k_0_to_20": all(shipped[k] == t[k] for k in range(21)),
               "exact_over_the_whole_warmup_0_to_1000":
                   all(shipped[k] == t[k] for k in range(1001))},
           "sha256_lr_schedulers": hashlib.sha256(open(UP, "rb").read()).hexdigest()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
