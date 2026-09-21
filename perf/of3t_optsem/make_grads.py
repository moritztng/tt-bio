#!/usr/bin/env python3
"""The gradient sequence both arms consume, written once so neither can draw its own.

Three parameters standing in for the shape D107 is about: a trunk weight that every sample
activates, and a two-tensor confidence head that some samples disable. ``enabled[k][s]`` says
whether sample ``s`` of step ``k`` activates the confidence head; a step whose column is all
False is the zero-participation step, which is the case upstream steps and we skip.
"""
import json
import sys

import numpy as np

STEPS, SAMPLES = 20, 4
SHAPES = {"trunk.w": (4, 4), "conf.w": (3, 5), "conf.b": (5,)}
CONF = ("conf.w", "conf.b")
# Step 3 is the zero-participation step in the `single` scenario. 20 steps is PROTOCOL 7's
# trajectory length, and it is what gives the reading dynamic range above the A16 zero
# baseline: over 6 steps at lr 1.8e-03 the weights have barely left their initial values, so
# a never-stepping arm and a skipping arm read the same number.
ZERO_AT = 3
# `never`   -- participation >= 1 everywhere. The A/A control; the no-zero arm PROTOCOL 5
#              requires to stay at its ~3e-08 level.
# `single`  -- one zero-participation step at k=3. Upstream's case: a global batch drawn
#              wholly from the four zero-confidence-weight datasets.
# `always`  -- the confidence head is disabled on EVERY sample of EVERY step. This is what
#              our own `train_loop` produces, because `af3_loss` skips a term whose weight is
#              zero and `LOSS_WEIGHTS["pretrain"]` sets `pae: 0.0` for the whole run.
SCENARIOS = ("never", "single", "always")


def build(scenario="single", seed=107, scale=3.0):
    rng = np.random.default_rng(seed)
    init = {n: rng.standard_normal(s).astype(np.float32) for n, s in SHAPES.items()}
    # `scale` puts some per-sample norms above the 10.0 clip and some below, so the clip
    # coefficient is exercised rather than sitting at 1.0 for the whole run.
    grads = {n: (rng.standard_normal((STEPS, SAMPLES) + s) * scale).astype(np.float32)
             for n, s in SHAPES.items()}
    enabled = np.ones((STEPS, SAMPLES), bool)
    if scenario == "single":
        enabled[ZERO_AT - 1, :] = False
    elif scenario == "always":
        enabled[:, :] = False
    elif scenario != "never":
        raise ValueError(f"{scenario!r} is not one of {SCENARIOS}")
    return init, grads, enabled


if __name__ == "__main__":
    scenario = sys.argv[1]
    out = sys.argv[2]
    init, grads, enabled = build(scenario)
    np.savez(out, enabled=enabled,
             **{f"init/{n}": v for n, v in init.items()},
             **{f"grad/{n}": v for n, v in grads.items()})
    print(json.dumps({"path": out, "scenario": scenario,
                      "steps": STEPS, "samples": SAMPLES,
                      "zero_participation_steps":
                          [int(k) + 1 for k in np.where(~enabled.any(1))[0]],
                      "confidence_params": list(CONF),
                      "shapes": {n: list(s) for n, s in SHAPES.items()}}, indent=1))
