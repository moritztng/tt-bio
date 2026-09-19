#!/usr/bin/env python3
"""PROTOCOL SS4, first half: the OpenFold3 learning rate schedule, verified exactly.

The schedule is a pure function of the step index with no state beyond its configuration, so
it needs no trajectory and no card: it is evaluated at EVERY step of its domain on both
sides and required to agree bit for bit. That is a complete proof of the schedule, which is
strictly more than any N-step training run could show about it.

"Theirs" is upstream's own object, driven the way Lightning drives it -- construct
`AlphaFoldLRScheduler` over a real torch optimizer and call `.step()`, reading
`get_last_lr()` at each step index. Not a reimplementation of their formula: their formula,
running.

"Ours" is `tt_bio.train.af3_lr`.

Bar (PROTOCOL SS4): EXACT float64 equality. Both sides compute the same closed form and there
is no accumulation to excuse a difference, so anything but equality is a real disagreement.

Three checks, and the second and third are what make the first mean anything:
  1. agreement, over every step of every configuration;
  2. a negative control -- perturb ours at ONE step by 1 %, confirm the check fails at that
     step and only that step (PROTOCOL SS3e: a gate nobody has watched fail is not a gate);
  3. a regression guard -- the Protenix closed form this function carried before OF3 was
     added is recomputed inline and required to still be bit-identical, so extending the
     function cannot have silently moved the model that was already using it.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch

from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from tt_bio.train.optim import af3_lr

# (name, their kwargs, max_step). The first is upstream's shipped default. The rest compress
# the knees so that several decay periods, a non-zero base_lr and a warmup that ends exactly
# on the plateau boundary all land inside a domain small enough to enumerate exhaustively --
# corner cases a default-configured run reaches only after 100k steps, or never.
CONFIGS = [
    ("of3_defaults",
     dict(base_lr=0.0, max_lr=1e-3, warmup_no_steps=1000,
          start_decay_after_n_steps=50000, decay_every_n_steps=50000, decay_factor=0.95),
     100001),
    ("many_decay_periods",
     dict(base_lr=0.0, max_lr=1.8e-3, warmup_no_steps=100,
          start_decay_after_n_steps=300, decay_every_n_steps=200, decay_factor=0.9),
     4000),
    ("nonzero_base_lr",
     dict(base_lr=5e-5, max_lr=1e-3, warmup_no_steps=250,
          start_decay_after_n_steps=1000, decay_every_n_steps=137, decay_factor=0.87),
     3000),
    ("warmup_ends_on_plateau_boundary",
     dict(base_lr=0.0, max_lr=1e-3, warmup_no_steps=500,
          start_decay_after_n_steps=500, decay_every_n_steps=250, decay_factor=0.95),
     2000),
]

# The knees PROTOCOL SS4 names for the shipped configuration, and both sides of each.
DEFAULT_KNEES = [0, 1, 999, 1000, 1001, 49999, 50000, 50001, 99999, 100000, 100001]


def theirs(cfg: dict, max_step: int) -> list[float]:
    """Upstream's scheduler, stepped. Index k of the result is their lr at step k."""
    param = [torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))]
    opt = torch.optim.AdamW(param, lr=cfg["max_lr"])
    with warnings.catch_warnings():
        # Lightning calls optimizer.step() first; we never step the optimizer at all, so
        # torch's ordering warning fires on every call and says nothing about the schedule.
        warnings.simplefilter("ignore", UserWarning)
        sched = AlphaFoldLRScheduler(opt, **cfg)
        out = [sched.get_last_lr()[0]]
        for _ in range(max_step):
            sched.step()
            out.append(sched.get_last_lr()[0])
    return out


def ours(cfg: dict, max_step: int) -> list[float]:
    return [af3_lr(k, cfg["max_lr"],
                   warmup_steps=cfg["warmup_no_steps"],
                   decay_every_n_steps=cfg["decay_every_n_steps"],
                   decay_factor=cfg["decay_factor"],
                   base_lr=cfg["base_lr"],
                   plateau_until=cfg["start_decay_after_n_steps"])
            for k in range(max_step + 1)]


def protenix_reference(step: int, lr: float, warmup: int, every: int, factor: float) -> float:
    """The exact body `af3_lr` had before the OF3 form was added, recomputed here.

    Verbatim from tt_bio/train/optim.py at bd643929a. Its job is to fail if extending the
    function for OpenFold3 moved the schedule Protenix is already training on.
    """
    if step <= warmup:
        return step / warmup * lr
    return lr * (factor ** (step // every))


def compare(a: list[float], b: list[float]) -> list[int]:
    """Indices where the two disagree. Exact equality, no tolerance (PROTOCOL SS4)."""
    return [k for k, (x, y) in enumerate(zip(a, b)) if x != y]


def main() -> int:
    result: dict = {"instrument": "PROTOCOL SS4 -- LR schedule",
                    "bar": "exact float64 equality at every step",
                    "configs": {}, "negative_control": {}, "protenix_regression": {}}
    ok = True

    for name, cfg, max_step in CONFIGS:
        t, o = theirs(cfg, max_step), ours(cfg, max_step)
        assert len(t) == len(o) == max_step + 1
        bad = compare(t, o)
        entry = {"config": cfg, "steps_compared": len(t), "mismatches": len(bad),
                 "first_mismatches": bad[:8], "pass": not bad}
        if name == "of3_defaults":
            entry["knees"] = {str(k): {"theirs": t[k], "ours": o[k], "equal": t[k] == o[k]}
                              for k in DEFAULT_KNEES}
        result["configs"][name] = entry
        ok &= not bad
        print(f"[{'PASS' if not bad else 'FAIL'}] {name}: {len(t)} steps compared, "
              f"{len(bad)} mismatches")

    # --- negative control (PROTOCOL SS3e) --------------------------------------------
    name, cfg, max_step = CONFIGS[0]
    t, o = theirs(cfg, max_step), ours(cfg, max_step)
    victim = 37_000  # inside the plateau, where a 1 % error cannot hide in a warmup ramp
    o_bad = list(o)
    o_bad[victim] *= 1.01
    bad = compare(t, o_bad)
    nc_ok = bad == [victim]
    result["negative_control"] = {
        "perturbed_step": victim, "perturbation": "x1.01",
        "mismatching_steps": bad[:8], "n_mismatching": len(bad),
        "localised_to_the_perturbed_step_only": nc_ok, "pass": nc_ok}
    ok &= nc_ok
    print(f"[{'PASS' if nc_ok else 'FAIL'}] negative control: perturbing step {victim} by 1 % "
          f"flags {len(bad)} step(s), expected exactly [{victim}]")

    # --- regression guard on the Protenix caller --------------------------------------
    pcfg = dict(lr=1.8e-3, warmup=1000, every=50000, factor=0.95)
    before = [protenix_reference(k, pcfg["lr"], pcfg["warmup"], pcfg["every"], pcfg["factor"])
              for k in range(100002)]
    after = [af3_lr(k, pcfg["lr"], warmup_steps=pcfg["warmup"],
                    decay_every_n_steps=pcfg["every"], decay_factor=pcfg["factor"])
             for k in range(100002)]
    bad = compare(before, after)
    result["protenix_regression"] = {"steps_compared": len(before), "mismatches": len(bad),
                                     "pass": not bad}
    ok &= not bad
    print(f"[{'PASS' if not bad else 'FAIL'}] protenix regression: {len(before)} steps, "
          f"{len(bad)} mismatches")

    result["verdict"] = "PASS" if ok else "FAIL"
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nVERDICT: {result['verdict']}  ->  {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
