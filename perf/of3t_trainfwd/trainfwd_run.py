#!/usr/bin/env python3
"""One arm of the OF3 training forward, through the registered adapter, on one card.

PROTOCOL SS6's rule, unchanged: *a path is covered when a parameter gradient MOVES against an
arm with the path off, not when a config key is set.* So every arm here is a real forward and a
real backward, and what is reported is the gradient, not that the code ran.

Arms
----

``full``        the adapter's forward, the registered ``af3`` objective's seeds, one backward.
                This is ``model_forward``: the eight terms are keyed off the outputs the
                forward produced, and the gradient is carried into the model's own parameters.

``zeroseed``    the same forward, every seed replaced by zeros. The control for ``full``. A
                parameter that moves here moves without a loss, which would mean the count in
                ``full`` is not measuring what it says.

``norollout``   the same forward with the confidence heads fed the GROUND TRUTH coordinates
                instead of the rolled-out ones. This is ``diffusion_rollout`` OFF, and it is
                the right control precisely because upstream runs the rollout under
                ``no_grad`` (``of3pkg043/.../model.py:381``): the rollout's whole reach into a
                parameter gradient is the structure it hands the confidence heads, so removing
                the rollout and keeping everything else is what isolates it.

Nothing upstream is modified. A zero difference is the finding.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from perf.clocksample import during  # noqa: E402


def git_sha() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def grad_snapshot(params) -> dict:
    """``{name: float64 flat numpy}`` for every parameter carrying a gradient."""
    import ttnn
    from tt_bio import autograd as ag
    out = {}
    for name, raw in params.items():
        p = ag._PARAMS.get(id(raw))
        g = getattr(p, "grad", None)
        if g is None:
            continue
        out[name] = ttnn.to_torch(g).to(torch.float64).reshape(-1).numpy()
    return out


def score(a: dict, b: dict) -> dict:
    """``a`` against ``b``, per section and overall, in float64."""
    tot_a = tot_d = 0.0
    sec: dict[str, dict] = {}
    moved = 0
    for name, ga in a.items():
        gb = b.get(name)
        d = ga if gb is None else ga - gb
        sa, sd = float((ga ** 2).sum()), float((d ** 2).sum())
        tot_a += sa
        tot_d += sd
        if sd > 0.0:
            moved += 1
        s = name.split(".")[0]
        e = sec.setdefault(s, {"n": 0, "sq": 0.0, "delta_sq": 0.0})
        e["n"] += 1
        e["sq"] += sa
        e["delta_sq"] += sd
    for e in sec.values():
        e["share_of_own"] = e["delta_sq"] / e["sq"] if e["sq"] else None
    return {"squared_norm_arm_on": tot_a, "squared_norm_of_difference": tot_d,
            "share_of_squared_gradient_norm": tot_d / tot_a if tot_a else None,
            "n_params_moved": moved, "n_params_with_gradient_on": len(a),
            "by_section": sec}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("full", "zeroseed", "norollout"))
    ap.add_argument("--batch", type=Path,
                    default=Path("/home/ttuser/of3t/bundle_min/batch_step003.pt"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt")))
    ap.add_argument("--rollout", type=int, default=20)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--grad-out", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.perf_counter()

    from tt_bio import autograd as ag
    from tt_bio.train import objectives
    from tt_bio.train.losses import of3_loss_weights
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward

    rec = {
        "instrument": "PROTOCOL SS6: the OF3 training forward through tt_bio.train.catalogue",
        "arm": a.arm, "batch": str(a.batch), "checkpoint": str(a.checkpoint),
        "rollout": a.rollout, "cycles": a.cycles, "seed": a.seed,
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": git_sha(),
    }

    ds = OpenFold3Dataset(a.batch)
    fwd = OpenFold3Forward(a.checkpoint, rollout=a.rollout, num_cycles=a.cycles, seed=a.seed)
    batch = ds.batch([0])
    rec["padded_width"] = int(batch["features"]["token_mask"].shape[0])
    rec["real_tokens"] = int(batch["coord_mask"].shape[0] and
                             batch["features"]["token_mask"].sum())
    rec["pdb_id"] = batch["pdb_id"]

    # `norollout` hands the confidence heads the ground truth instead of a rollout. The edit
    # is on the ADAPTER's own switch, not a second forward.
    if a.arm == "norollout":
        fwd.repr_coords_in = batch["true_xyz"]

    with during() as clk:
        try:
            t_f = time.perf_counter()
            outputs = fwd(batch)
            fwd_s = time.perf_counter() - t_f
            params = fwd.parameters()
            rec["params_reachable"] = len(params)
            print(f"[{time.perf_counter()-t0:.0f}s] forward done in {fwd_s:.1f}s, "
                  f"{len(params)} device weights reachable", flush=True)

            host = {k: np.asarray(ag_to_host(v), np.float64) for k, v in outputs.items()}
            weights = of3_loss_weights("initial_training", "weighted-pdb")
            loss, breakdown, seeds = objectives.objective("af3")(batch, host, weights=weights)
            rec["loss"] = float(loss)
            rec["weights"] = weights
            rec["breakdown"] = {k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                                    for kk, vv in v.items() if kk != "derived"}
                                for k, v in breakdown.items()}
            rec["seeded_outputs"] = sorted(seeds)
            print(f"[{time.perf_counter()-t0:.0f}s] loss {float(loss):.9f}; seeds "
                  f"{sorted(seeds)}", flush=True)

            if a.arm == "zeroseed":
                seeds = {k: np.zeros_like(v) for k, v in seeds.items()}

            t_b = time.perf_counter()
            ag.backward([outputs[k] for k in seeds],
                        [to_dev(seeds[k], outputs[k]) for k in seeds])
            bwd_s = time.perf_counter() - t_b
            rec["forward_s"], rec["backward_s"] = fwd_s, bwd_s
            rec["step_s"] = fwd_s + bwd_s

            g = grad_snapshot(params)
            rec["params_with_grad"] = len(g)
            rec["params_nonzero_grad"] = sum(1 for v in g.values() if float((v ** 2).sum()) > 0)
            rec["squared_gradient_norm"] = sum(float((v ** 2).sum()) for v in g.values())
            by: dict[str, dict] = {}
            for name, v in g.items():
                e = by.setdefault(name.split(".")[0], {"n": 0, "sq": 0.0})
                e["n"] += 1
                e["sq"] += float((v ** 2).sum())
            rec["by_section"] = by
            if a.grad_out:
                a.grad_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({k: torch.from_numpy(v) for k, v in g.items()}, a.grad_out)
                rec["grad_dump"] = str(a.grad_out)
            rec["ok"] = True
        except Exception:
            import traceback
            rec["ok"] = False
            rec["error"] = traceback.format_exc()[-8000:]
    rec["aiclk_during"] = clk.summary()
    rec["aiclk_line"] = clk.line(0)
    rec["total_s"] = time.perf_counter() - t0

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(rec["aiclk_line"], flush=True)
    if not rec["ok"]:
        print(rec["error"][-2500:], flush=True)
    else:
        print(f"{a.arm}: {rec['params_nonzero_grad']} of {rec['params_reachable']} tensors "
              f"carry a non-zero gradient; step {rec['step_s']:.1f} s "
              f"(fwd {rec['forward_s']:.1f} + bwd {rec['backward_s']:.1f})", flush=True)
    print("->", a.out, flush=True)
    return 0 if rec["ok"] else 1


def ag_to_host(t):
    import ttnn
    return ttnn.to_torch(t.value).to(torch.float32).numpy()


def to_dev(arr, like):
    import ttnn
    return ttnn.from_torch(torch.from_numpy(np.asarray(arr, np.float32)),
                           dtype=like.value.dtype, layout=like.value.layout,
                           device=like.value.device())


if __name__ == "__main__":
    raise SystemExit(main())
