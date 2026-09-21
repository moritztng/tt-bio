#!/usr/bin/env python3
"""The reference trajectory, produced by running OpenFold3's own files.

Not a reimplementation of their update rule: this imports
``openfold3.core.utils.grad_manager.PerSampleGradManager`` from the 0.4.3 reference checkout
and drives it exactly as ``runner.py:_training_step_manual_clip`` does -- per-sample
``clip_and_accumulate(disabled_params=...)``, then ``sync_and_average_grads()``, then
``torch.optim.Adam.step()`` (their ``configure_optimizers``, runner.py:845-850, plain Adam
with no weight decay), then ``reset_accumulator()``.

The only substitution is the distributed reduce. ``_sync_and_average_grads`` calls
``self._trainer.strategy.reduce(..., ReduceOp.SUM)``; at world size 1 that reduce is the
identity, so the stub returns its argument. Everything the defect is about -- who gets a
``.grad``, what a participation count of 0 does to it, and whether the optimizer steps it --
is their code, unmodified.

Run with the reference tree on PYTHONPATH:
    PYTHONPATH=<ref> of3-upstream-venv/bin/python upstream_arm.py grads.npz out.json [--dtype float32]
"""
import argparse
import json

import numpy as np
import torch

from openfold3.core.utils import grad_manager as gm_mod
from openfold3.core.utils.grad_manager import PerSampleGradManager

CONF = ("conf.w", "conf.b")
LR, BETAS, EPS = 1.8e-3, (0.9, 0.95), 1e-8   # OF3's own: model_config.py:143-146
CLIP_VAL = 10.0                              # model_config.py:157, per_sample_clipping True


class _Strategy:
    """World size 1. `reduce(x, SUM)` over one rank is `x`."""

    @staticmethod
    def reduce(tensor, reduce_op=None):
        return tensor


class _Trainer:
    strategy = _Strategy()
    global_step = 0


class _Model(torch.nn.Module):
    def __init__(self, init, dtype):
        super().__init__()
        # `named_parameters` has to yield the dotted names the accumulator keys on, and a
        # ParameterDict cannot hold a key with a dot in it. One submodule per prefix gives
        # `trunk.w` and `conf.w` as real names rather than names with the dot rewritten.
        self.trunk, self.conf = torch.nn.Module(), torch.nn.Module()
        for name, value in init.items():
            head, leaf = name.split(".")
            getattr(self, head).register_parameter(
                leaf, torch.nn.Parameter(torch.tensor(value, dtype=dtype)))


def run(path, dtype, skip_zero_step):
    data = np.load(path)
    enabled = data["enabled"]
    steps, samples = enabled.shape
    names = sorted(k.split("/", 1)[1] for k in data.files if k.startswith("init/"))
    init = {n: data[f"init/{n}"] for n in names}
    grads = {n: data[f"grad/{n}"] for n in names}

    model = _Model(init, dtype)
    trainer = _Trainer()
    manager = PerSampleGradManager(gradient_clip_val=CLIP_VAL, accumulate_grad_batches=1,
                                   log_grad_norm=False)
    manager.setup(model=model, trainer=trainer, logger=None)
    params = dict(model.named_parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)

    trace = []
    for k in range(steps):
        for s in range(samples):
            disabled = set() if enabled[k, s] else set(CONF)
            for name, p in params.items():
                p.grad = (None if name in disabled else
                          torch.tensor(grads[name][k, s], dtype=dtype))
            manager.clip_and_accumulate(logging_info=None, disabled_params=disabled)
            for p in params.values():
                p.grad = None
        counts = dict(manager.parameter_participation_counts)
        manager.sync_and_average_grads()
        if skip_zero_step:
            # The break control: upstream's own code, with OUR skip grafted onto it. A
            # parameter no sample activated keeps the value it had rather than stepping.
            held = {n: p.detach().clone() for n, p in params.items()
                    if counts.get(n, 0) == 0}
            opt.step()
            for n, v in held.items():
                with torch.no_grad():
                    params[n].copy_(v)
        else:
            opt.step()
        manager.reset_accumulator()
        trace.append({
            "k": k + 1,
            "participation": {n: int(counts.get(n, 0)) for n in names},
            "theta": {n: params[n].detach().double().numpy().tolist() for n in names},
        })
    return {"reference": gm_mod.__file__, "torch": torch.__version__,
            "dtype": str(dtype), "optimizer": "torch.optim.Adam",
            "lr": LR, "betas": list(BETAS), "eps": EPS, "clip_val": CLIP_VAL,
            "skip_zero_participation": skip_zero_step, "trace": trace}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grads")
    ap.add_argument("out")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--skip-zero-step", action="store_true",
                    help="break control: graft our skip onto their optimizer")
    a = ap.parse_args()
    result = run(a.grads, getattr(torch, a.dtype), a.skip_zero_step)
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps({"out": a.out, "reference": result["reference"],
                      "participation_by_step":
                          [t["participation"] for t in result["trace"]]}, indent=1))
