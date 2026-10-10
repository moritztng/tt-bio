#!/usr/bin/env python3
"""One BindCraft 2 gradient step per (arm, design_dropout), so issue #17's fix can be graded.

The device arm folds with tt-bio's stacks on the card; the host arm is BindCraft 2's own
`AlphaFoldDesignModel`, untouched, on JAX. Both use the model's default key, so with dropout on
both draw the same masks, and the grade is whether the card's step follows the host's as closely
with dropout as without it -- and differs from its own dropout-off step, which it did not before.

    TT_VISIBLE_DEVICES=3 PYTHONPATH=.:/path/to/BindCraft2 python3 perf/bc2_dropout/probe.py \\
        --arm device --preset model_1_ptm --out runs/
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("device", "host"), required=True)
    ap.add_argument("--preset", default="model_1_ptm")
    ap.add_argument("--steps", type=int, default=2, help="steps per dropout setting; the first "
                    "compiles, so the time is taken from the last")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import test_bindcraft2_hw as hw
    from perf import clocksample
    from tt_bio import bindcraft2

    params, protein_states, losses = hw._pdl1_draw()

    def run(model, tag, calls=None):
        rows = {}
        for dropout in (False, True):
            model.dropout = dropout
            for step in range(a.steps):
                with clocksample.during(period=1.0) as clock:
                    t0 = time.time()
                    _, gradients, loss = model.sequence_gradients(protein_states, losses)
                    dt = time.time() - t0
            name, grad = next(iter(sorted(gradients.items())))
            grad = np.asarray(grad, dtype=np.float32)
            np.save(a.out / f"{tag}_{a.preset}_drop{int(dropout)}.npy", grad)
            rows[f"drop{int(dropout)}"] = {
                "loss": float(loss), "step_s": round(dt, 2), "grad_key": name,
                "finite": bool(np.isfinite(grad).all()), "clock": clock.line(),
                "calls": dict(calls()) if calls else None}
            print(tag, a.preset, dropout, rows[f"drop{int(dropout)}"], flush=True)
        (a.out / f"{tag}_{a.preset}.json").write_text(json.dumps(rows, indent=1))

    kwargs = dict(presets=a.preset, data_dir=str(params), max_cache_size=1, num_recycle=1,
                  length_bucket_size=32)
    if a.arm == "device":
        with bindcraft2.predictor(trunk="device", checkpoints=params) as build:
            model = build(**kwargs)
            run(model, "device", lambda: {"evo": build.evoformer.calls,
                                          "extra": getattr(build.extra_msa, "calls", None),
                                          "template": getattr(build.template, "calls", None)})
    else:
        from bindcraft.af2 import AlphaFoldDesignModel
        run(AlphaFoldDesignModel(**kwargs), "host")


if __name__ == "__main__":
    main()
