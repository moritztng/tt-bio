#!/usr/bin/env python3
"""of3t-fullstep64: our full training step, exact on or off, replaying the float64 run's draws.

    devstep.py --exact on|off --draws draws.pt --grad-out G.pt [--weights-out W.pt] --out F.json

`perf/of3t_stackship/stepcost.py` with three additions and nothing else changed: the rollout's
draws come from the float64 reference's `draws.pt` (via `draws.Draws`, the same recorder the
reference sampled with), every parameter's gradient is saved (`trainfwd_run --grad-out`), and
optionally the walked device weights are saved for the bijection. `--exact off` is the public off
switch, `autograd.exact_training(False)`.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    get = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
    exact, out = get("--exact"), Path(get("--out"))
    draws_path, grad_out, weights_out = get("--draws"), get("--grad-out"), get("--weights-out")
    assert exact in ("on", "off"), exact
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[0:0] = [here, os.path.join(os.getcwd(), "perf/of3t_trainfwd")]
    import torch
    from draws import Draws
    from tt_bio import autograd as ag
    from tt_bio import openfold3_fold
    from tt_bio.train.openfold3 import OpenFold3Forward
    import trainfwd_run

    replay = torch.load(draws_path, weights_only=False)["torch_randn"]
    log = {"calls": 0, "n_draws": 0, "mismatch": []}
    orig = openfold3_fold.OpenFold3._gen_rollout

    def gen_rollout(self, *a, **k):
        with Draws(replay) as rec:
            got = orig(self, *a, **k)
        log["calls"] += 1
        log["n_draws"] += len(rec.recorded)
        log["mismatch"] += rec.mismatch
        return got

    openfold3_fold.OpenFold3._gen_rollout = gen_rollout
    walked = {}
    orig_params = OpenFold3Forward.parameters

    def parameters(self):
        got = orig_params(self)
        walked.update(got)
        return got

    OpenFold3Forward.parameters = parameters

    sys.argv = ["trainfwd_run.py", "--arm", "full", "--out", str(out), "--grad-out", grad_out]
    with (ag.exact_training(False) if exact == "off" else contextlib.nullcontext()):
        ops = list(ag.exact_training_ops())
        rc = trainfwd_run.main()
    rec = json.loads(out.read_text())
    rec["fullstep64"] = {
        "exact": exact, "exact_training_ops": ops,
        "counters": {"softmax": dict(ag.EXACT_SOFTMAX_STATS),
                     "layer_norm": dict(ag.EXACT_LAYER_NORM_STATS)},
        "draws": {"file": draws_path, "n_recorded": len(replay), "rollout_calls": log["calls"],
                  "n_consumed": log["n_draws"], "mismatch_count": len(log["mismatch"]),
                  "mismatch": log["mismatch"][:20]}}
    if weights_out:
        import ttnn
        torch.save({k: ttnn.to_torch(t).to(torch.float32) for k, t in walked.items()},
                   weights_out)
        rec["fullstep64"]["weights_dump"] = {"file": weights_out, "n": len(walked)}
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("DEVSTEP " + json.dumps({"exact": exact, "loss": rec.get("loss"),
                                   "sqnorm": rec.get("squared_gradient_norm"),
                                   "draws": rec["fullstep64"]["draws"],
                                   "aiclk": rec.get("aiclk_line")}), flush=True)
    return rc if rc else (1 if log["mismatch"] or log["calls"] != 1 else 0)


if __name__ == "__main__":
    raise SystemExit(main())
