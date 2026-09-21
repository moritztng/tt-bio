#!/usr/bin/env python3
"""Our trajectory: ``tt_bio.train.optim.AdamW`` over the same gradient file.

Host only. ``to_host``/``to_device`` are replaced by a numpy handle so the optimizer runs
without a card -- everything the comparison reads is the fp32 master, which never leaves the
host anyway. The device write-back still happens through the handle, so ``check_displacement``
and the per-step kept ratio stay live rather than being stubbed out of the measurement.

    tt-bio/env/bin/python ours_arm.py grads.npz out.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from tt_bio import autograd as ag                            # noqa: E402
from tt_bio.train import optim as optim_mod                  # noqa: E402

CONF = ("conf.w", "conf.b")
LR, BETAS, EPS, WD = 1.8e-3, (0.9, 0.95), 1e-8, 0.0
CLIP_VAL = 10.0


class _Handle:
    """The smallest thing `optim.py` treats as a device tensor."""

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


def run(path):
    data = np.load(path)
    enabled = data["enabled"]
    steps, samples = enabled.shape
    names = sorted(k.split("/", 1)[1] for k in data.files if k.startswith("init/"))
    init = {n: data[f"init/{n}"] for n in names}
    grads = {n: data[f"grad/{n}"] for n in names}

    ag.forget_parameters()
    params = {n: ag.parameter(_Handle(init[n].copy())) for n in names}
    opt = optim_mod.AdamW(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD,
                          clip_norm=CLIP_VAL)

    trace = []
    for k in range(steps):
        for s in range(samples):
            disabled = set() if enabled[k, s] else set(CONF)
            for n, t in params.items():
                t.grad = None if n in disabled else _Handle(grads[n][k, s])
            opt.clip_and_accumulate(disabled=disabled)
            opt.zero_grad()
        counts = dict(opt.participation)
        opt.step()
        trace.append({
            "k": k + 1,
            "participation": {n: int(counts.get(n, 0)) for n in names},
            "theta": {n: opt.master[n].astype(np.float64).tolist() for n in names},
        })
    return {"optimizer": "tt_bio.train.optim.AdamW", "lr": LR, "betas": list(BETAS),
            "eps": EPS, "weight_decay": WD, "clip_val": CLIP_VAL,
            "source": optim_mod.__file__, "trace": trace}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grads")
    ap.add_argument("out")
    a = ap.parse_args()
    result = run(a.grads)
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps({"out": a.out, "source": result["source"],
                      "participation_by_step":
                          [t["participation"] for t in result["trace"]]}))
