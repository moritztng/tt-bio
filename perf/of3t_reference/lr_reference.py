#!/usr/bin/env python3
"""Emit the LR-schedule reference table: upstream's own AlphaFoldLRScheduler, in float64.

PROTOCOL §4 verifies the schedule exactly over its whole domain rather than along a trajectory,
because it is a pure function of the step index. This writes the values the comparison is made
against: every knee and both sides of it, plus uniformly spaced interior points.

The configured values come from their `train` preset, not from the scheduler's Python defaults.
That distinction matters: the class defaults to max_lr=1e-3, and `configure_optimizers` overrides
it with `optimizer_config.learning_rate`, which the preset sets to 1.8e-3.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-step", type=int, default=150000)
    ap.add_argument("--interior", type=int, default=1000)
    args = ap.parse_args()

    from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"]).settings
    opt_cfg, sched_cfg = cfg.optimizer, cfg.lr_scheduler
    kwargs = {
        "base_lr": sched_cfg.base_lr,
        "max_lr": opt_cfg.learning_rate,
        "warmup_no_steps": sched_cfg.warmup_no_steps,
        "start_decay_after_n_steps": sched_cfg.start_decay_after_n_steps,
        "decay_every_n_steps": sched_cfg.decay_every_n_steps,
        "decay_factor": sched_cfg.decay_factor,
    }

    knees = []
    for k in (kwargs["warmup_no_steps"], kwargs["start_decay_after_n_steps"]):
        knees += [k - 1, k, k + 1]
    for m in range(1, 1 + args.max_step // kwargs["decay_every_n_steps"]):
        k = kwargs["start_decay_after_n_steps"] + m * kwargs["decay_every_n_steps"]
        knees += [k - 1, k, k + 1]
    interior = [
        round(i * args.max_step / (args.interior - 1)) for i in range(args.interior)
    ]
    # 0..31 explicitly: the N-step trajectory lives entirely inside warmup, so the
    # comparison needs every one of its step indices present, not just the knees.
    steps = sorted({*range(0, 32), *knees, *interior, args.max_step})

    # The scheduler reports the LR for its current last_epoch, so drive one instance forward and
    # read it at every step rather than constructing one per step: constructing with last_epoch=k
    # is a different code path (it requires initial_lr on the group) and we want the path training
    # actually takes.
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.Adam([p], lr=opt_cfg.learning_rate)
    sched = AlphaFoldLRScheduler(opt, last_epoch=-1, **kwargs)

    table = {}
    want = set(steps)
    for k in range(max(steps) + 1):
        if k in want:
            table[k] = float(opt.param_groups[0]["lr"])
        sched.step()

    payload = {
        "source": "openfold3 core/utils/lr_schedulers.py AlphaFoldLRScheduler",
        "config": kwargs,
        "note": (
            "lr[k] is the learning rate the OPTIMIZER USES at step k, i.e. read before "
            "sched.step(). lr[0] is 0.0 because the scheduler is constructed with last_epoch=-1 "
            "and _LRScheduler.__init__ calls step() once; the first optimizer step therefore "
            "moves no weights at all."
        ),
        "n_points": len(table),
        "lr": {str(k): v for k, v in table.items()},
    }
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.out.write_text(body)
    print(f"{len(table)} points -> {args.out}")
    print(f"sha256 {hashlib.sha256(body.encode()).hexdigest()}")
    for k in (0, 1, 999, 1000, 1001, 49999, 50000, 50001, 99999, 100000, 100001):
        if k in table:
            print(f"  lr[{k}] = {table[k]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
