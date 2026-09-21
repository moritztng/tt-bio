#!/usr/bin/env python3
"""D107, measured: what our `AdamW.step` does with a parameter that got no gradient this step.

`tt_bio/train/optim.py:234` reads `if g is None: continue`, so a parameter absent from every
sample of a step is SKIPPED: its moments are not decayed and it takes no update. Upstream's
`_sync_and_average_grads` (`grad_manager.py`) assigns `param.grad` for every parameter it
manages, so `torch.optim.AdamW` still steps that parameter from its DECAYING MOMENTUM. Whether
the op runs, not what it computes.

D107 was filed from READING. It could not be measured in PROTOCOL S7's harness because that
harness has one fully-enabled sample in four, so participation is >= 1 at every rung (measured
spread [1,4]) and the divergence needs exactly 0. This makes it fire, on the smallest input
that can: one parameter, a gradient on some steps and none on one.

No card, no model. The question is arithmetic in the update rule, so the answer is arithmetic:
our shipped `AdamW` against `torch.optim.AdamW` on the same parameter, the same lr, betas, eps
and weight decay, over the same gradient sequence -- once with a zero-participation step and
once without, and the second is the A/A floor that makes the first readable.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402

from tt_bio.train import optim as optim_mod                                 # noqa: E402
from tt_bio import autograd as ag                                           # noqa: E402

LR, BETAS, EPS, WD = 1.8e-3, (0.9, 0.95), 1e-8, 0.0        # OF3's own; train_loop's wd is 0.0
STEPS = 6
SHAPE = (4, 4)


class _Handle:
    def __init__(self, arr):
        self.arr = np.ascontiguousarray(arr, dtype=np.float32)

    def device(self):
        return "host"

    @property
    def dtype(self):
        return "float32"

    @property
    def shape(self):
        return self.arr.shape


optim_mod.to_host = lambda t, dtype=None: (t.arr if isinstance(t, _Handle)
                                           else np.asarray(t)).astype(np.float32)
optim_mod.to_device = lambda arr, device, dtype=None, layout=None: _Handle(arr)


def run(skip_at):
    """`skip_at` is the step (1-based) where the parameter has NO gradient at all."""
    rng = np.random.default_rng(107)
    w0 = rng.standard_normal(SHAPE).astype(np.float32)
    grads = [rng.standard_normal(SHAPE).astype(np.float32) for _ in range(STEPS)]

    ag.forget_parameters()
    t = ag.parameter(_Handle(w0.copy()))
    ours = optim_mod.AdamW({"w": t}, lr=LR, betas=BETAS, eps=EPS,
                           weight_decay=WD, clip_norm=0.0)

    tw = torch.tensor(w0.copy(), dtype=torch.float64, requires_grad=True)
    theirs = torch.optim.AdamW([tw], lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)

    trace = []
    for k in range(1, STEPS + 1):
        g = grads[k - 1]
        # OURS: no gradient means `t.grad is None`, which is what a step with participation 0
        # produces -- `clip_and_accumulate` never wrote an entry for it.
        t.grad = None if k == skip_at else g
        ours.step()
        # THEIRS: upstream assigns a grad for every managed parameter, and a parameter with no
        # contribution gets ZEROS, not None. torch then steps it anyway.
        tw.grad = torch.zeros_like(tw) if k == skip_at else torch.tensor(g, dtype=torch.float64)
        theirs.step()
        a = ours.master["w"].astype(np.float64)
        b = tw.detach().numpy()
        num = float(np.linalg.norm(a - b))
        den = float(np.linalg.norm(b))
        trace.append({"k": k, "grad": None if k == skip_at else "present",
                      "rel": num / (den + 1e-30), "abs": num,
                      "ours_norm": float(np.linalg.norm(a)),
                      "theirs_norm": den})
    return trace


control = run(skip_at=None)          # A/A floor: participation >= 1 at every step
fires = run(skip_at=3)               # participation 0 at k = 3

out = {
    "defect": "D107",
    "site": "tt_bio/train/optim.py, AdamW.step: `if g is None: continue`",
    "lr": LR, "betas": list(BETAS), "eps": EPS, "weight_decay": WD,
    "steps": STEPS, "shape": list(SHAPE),
    "control_no_zero_participation_step": control,
    "zero_participation_at_k3": fires,
    "control_final_rel": control[-1]["rel"],
    "fires_final_rel": fires[-1]["rel"],
    "ratio_over_control": (fires[-1]["rel"] / control[-1]["rel"]
                           if control[-1]["rel"] > 0 else float("inf")),
}
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("wrote " + sys.argv[1])
