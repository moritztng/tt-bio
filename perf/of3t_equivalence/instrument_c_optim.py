#!/usr/bin/env python3
"""PROTOCOL SS5: the optimizer recurrence, driven by injected gradients, no model, no card.

The point of the instrument (PROTOCOL SS5) is that the optimizer is a closed-form recurrence,
so its equivalence can be established without running a model at all: feed both sides the
IDENTICAL synthetic gradient sequence and compare the parameter state at every step. That
isolates optimizer-state divergence from gradient divergence, which a trajectory run
structurally cannot.

Bar (PROTOCOL SS5): relative 1e-06 per parameter per step, in fp32 master precision. Not a
bf16 argument -- nothing here is bf16.

"Theirs" is `torch.optim.Adam` constructed exactly as `configure_optimizers` constructs it
(`projects/of3_all_atom/runner.py:854`): lr 1.8e-3, betas (0.9, 0.95), eps 1e-8, and NO
weight decay, because they use Adam and not AdamW.

"Ours" is `tt_bio.train.optim.AdamW`. Its `step()` ends by casting the fp32 master down and
writing it to the device, which needs ttnn; the recurrence under test here is the master
update, which is pure host arithmetic. So the two host<->device helpers are replaced by
host-side stand-ins for the duration, leaving every line of the update itself running as
shipped. What this instrument therefore does NOT cover is the bf16 write-back -- that is a
device-side property and belongs with the gradient instrument, not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from tt_bio.train import optim as tt_optim
from tt_bio.train.optim import AdamW, af3_lr

# Their configuration, read off projects/of3_all_atom/config/model_config.py:143-153.
OF3 = dict(learning_rate=1.8e-3, beta1=0.9, beta2=0.95, eps=1e-8)
SCHED = dict(base_lr=0.0, warmup_no_steps=1000,
             start_decay_after_n_steps=50000, decay_every_n_steps=50000, decay_factor=0.95)
STEPS = 200
# Shapes chosen to look like real parameter tensors rather than scalars: a bias, a square
# projection, a fat linear and a 3-d stack.
SHAPES = {"bias": (128,), "proj": (64, 64), "linear": (256, 96), "stack": (4, 32, 32)}


class _HostValue:
    """Stands in for a ttnn tensor handle: answers `.device()` and `.dtype`, holds an array."""

    def __init__(self, arr: np.ndarray):
        self.arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.dtype = "float32"

    def device(self):
        return None


class _Param:
    """Stands in for `autograd.Tensor`: a value and a gradient, which is all `step()` reads."""

    __slots__ = ("value", "grad")

    def __init__(self, arr):
        self.value = _HostValue(arr)
        self.grad = None


def _install_host_stubs():
    """Swap the two PCIe helpers for host-side equivalents, returning the originals."""
    orig = (tt_optim.to_host, tt_optim.to_device)
    tt_optim.to_host = lambda t, dtype=None: (
        t.arr if isinstance(t, _HostValue) else orig[0](t, dtype=dtype))
    tt_optim.to_device = lambda arr, device, dtype=None, layout=None: _HostValue(arr)
    return orig


def gradients(seed: int) -> dict:
    """A fixed synthetic gradient sequence: STEPS draws per parameter, same for both sides.

    Scales differ per tensor by three orders of magnitude on purpose, so a bug that is
    invisible at one scale is not invisible everywhere.
    """
    rng = np.random.default_rng(seed)
    scale = {"bias": 1e-1, "proj": 1e-2, "linear": 1e-3, "stack": 1e-4}
    return {n: [(rng.standard_normal(s) * scale[n]).astype(np.float32)
                for _ in range(STEPS)] for n, s in SHAPES.items()}


def init_weights(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {n: (rng.standard_normal(s) * 0.02).astype(np.float32) for n, s in SHAPES.items()}


def run_theirs(w0: dict, grads: dict, lrs: list[float] | None) -> list[dict]:
    """torch.optim.Adam as `configure_optimizers` builds it. Returns weights after each step."""
    params = {n: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
              for n, v in w0.items()}
    opt = torch.optim.Adam(list(params.values()), lr=OF3["learning_rate"],
                           betas=(OF3["beta1"], OF3["beta2"]), eps=OF3["eps"])
    out = []
    for k in range(STEPS):
        if lrs is not None:
            for g in opt.param_groups:
                g["lr"] = lrs[k]
        opt.zero_grad()
        for n, p in params.items():
            p.grad = torch.tensor(grads[n][k], dtype=torch.float32)
        opt.step()
        out.append({n: p.detach().numpy().copy() for n, p in params.items()})
    return out


def run_ours(w0: dict, grads: dict, schedule) -> list[dict]:
    """Our AdamW, configured to BE Adam: weight_decay 0 and clipping off."""
    params = {n: _Param(v) for n, v in w0.items()}
    opt = AdamW(params, lr=OF3["learning_rate"], betas=(OF3["beta1"], OF3["beta2"]),
                eps=OF3["eps"], weight_decay=0.0, clip_norm=0.0, schedule=schedule)
    out = []
    for k in range(STEPS):
        for n, p in params.items():
            p.grad = grads[n][k]
        opt.step()
        out.append({n: opt.master[n].copy() for n in params})
    return out


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def compare(ours: list[dict], theirs: list[dict]) -> dict:
    worst = {"rel": -1.0, "step": None, "tensor": None}
    per_step = []
    for k, (o, t) in enumerate(zip(ours, theirs), start=1):
        step_worst = max(((rel(o[n], t[n]), n) for n in o), key=lambda x: x[0])
        per_step.append({"step": k, "worst_rel": step_worst[0], "worst_tensor": step_worst[1]})
        if step_worst[0] > worst["rel"]:
            worst = {"rel": step_worst[0], "step": k, "tensor": step_worst[1]}
    return {"worst": worst, "final_step_rel": per_step[-1]["worst_rel"],
            "per_step_worst": per_step}


BAR = 1e-6


def main() -> int:
    _install_host_stubs()
    result = {"instrument": "PROTOCOL SS5 -- optimizer under injected gradients",
              "bar": f"relative {BAR:.0e} per parameter per step, fp32 master",
              "steps": STEPS, "their_config": OF3, "arms": {}, "negative_control": {}}
    ok = True

    w0, grads = init_weights(7), gradients(11)

    # --- arm 1: constant lr. The pure Adam recurrence, nothing else moving. -------------
    a = compare(run_ours(w0, grads, None), run_theirs(w0, grads, None))
    a["pass"] = a["worst"]["rel"] <= BAR
    result["arms"]["constant_lr"] = a
    ok &= a["pass"]
    print(f"[{'PASS' if a['pass'] else 'FAIL'}] constant lr: worst relative "
          f"{a['worst']['rel']:.3e} at step {a['worst']['step']} on {a['worst']['tensor']!r}")

    # --- arm 2: the OF3 schedule on both sides -----------------------------------------
    # Ours computes lr(self.steps) AFTER incrementing, so its step k uses lr(k). Lightning
    # calls optimizer.step() and only then scheduler.step(), so their step k uses lr(k-1).
    # The arm is run both ways and the difference between them is reported rather than
    # assumed away: if the two alignments differ, the off-by-one is real and is a finding.
    sched = lambda s: af3_lr(s, OF3["learning_rate"], warmup_steps=SCHED["warmup_no_steps"],
                             decay_every_n_steps=SCHED["decay_every_n_steps"],
                             decay_factor=SCHED["decay_factor"], base_lr=SCHED["base_lr"],
                             plateau_until=SCHED["start_decay_after_n_steps"])
    ours_sched = run_ours(w0, grads, sched)
    aligned = compare(ours_sched, run_theirs(w0, grads, [sched(k) for k in range(1, STEPS + 1)]))
    lagged = compare(ours_sched, run_theirs(w0, grads, [sched(k) for k in range(0, STEPS)]))
    aligned["pass"] = aligned["worst"]["rel"] <= BAR
    result["arms"]["of3_schedule_aligned"] = aligned
    result["arms"]["of3_schedule_lightning_lag"] = {
        "worst": lagged["worst"], "final_step_rel": lagged["final_step_rel"],
        "note": "lr(k-1) against our lr(k); reported to size the off-by-one, not as a pass"}
    ok &= aligned["pass"]
    print(f"[{'PASS' if aligned['pass'] else 'FAIL'}] OF3 schedule, aligned: worst relative "
          f"{aligned['worst']['rel']:.3e} at step {aligned['worst']['step']} on "
          f"{aligned['worst']['tensor']!r}")
    print(f"[info] same arm with Lightning's lr(k-1) alignment: worst relative "
          f"{lagged['worst']['rel']:.3e} -- the size of a one-step schedule offset")

    # --- negative control (PROTOCOL SS3e) ----------------------------------------------
    # The first control written here FAILED, and the failure is a result rather than a bug.
    # Scaling ONE tensor's gradient by 1.01 at EVERY step moves the weight trajectory by
    # 2.5e-07 relative -- three orders under the bar, invisible. That is not a weak check,
    # it is Adam: the update is mhat / (sqrt(vhat) + eps), and a constant factor on g
    # cancels between numerator and denominator except through eps. So a uniform gradient
    # scaling error is STRUCTURALLY undetectable in a weight trajectory, which is the
    # sharpest available argument for why PROTOCOL SS3's per-parameter gradient check is
    # load-bearing and cannot be replaced by a longer trajectory run. It is recorded below
    # as `invariance` rather than deleted.
    #
    # The control that does break what this check reads perturbs the gradient of one tensor
    # at ONE step, which changes the m/v ratio instead of scaling it away.
    victim = "proj"
    theirs_ref = run_theirs(w0, grads, None)

    uniform = {n: ([g * 1.01 for g in v] if n == victim else v) for n, v in grads.items()}
    inv_final = run_ours(w0, uniform, None)[-1]
    inv_rel = {n: rel(inv_final[n], theirs_ref[-1][n]) for n in inv_final}
    result["negative_control"]["invariance"] = {
        "perturbation": f"{victim} gradient x1.01 at EVERY step",
        "per_tensor_final_rel": inv_rel,
        "tensors_over_bar": sorted(n for n, r in inv_rel.items() if r > BAR),
        "finding": "Adam is invariant to a uniform per-tensor gradient scaling; a weight "
                   "trajectory cannot detect one, so it cannot substitute for a direct "
                   "per-parameter gradient comparison"}
    print(f"[finding] uniform x1.01 on {victim!r} at every step moves the trajectory "
          f"{inv_rel[victim]:.3e} -- under the {BAR:.0e} bar. Adam's scale invariance.")

    kick_step = 50
    kicked = {n: [g * (1.01 if (n == victim and i == kick_step) else 1.0)
                  for i, g in enumerate(v)] for n, v in grads.items()}
    kick_final = run_ours(w0, kicked, None)[-1]
    kick_rel = {n: rel(kick_final[n], theirs_ref[-1][n]) for n in kick_final}
    over = sorted(n for n, r in kick_rel.items() if r > BAR)
    nc_ok = over == [victim]
    result["negative_control"]["single_step_kick"] = {
        "perturbed_tensor": victim, "perturbed_step": kick_step,
        "perturbation": "gradient x1.01 at that one step only",
        "per_tensor_final_rel": kick_rel, "tensors_over_bar": over,
        "localised": nc_ok, "pass": nc_ok}
    ok &= nc_ok
    print(f"[{'PASS' if nc_ok else 'FAIL'}] negative control: 1 % on {victim!r} at step "
          f"{kick_step} only -> over-bar tensors {over}, expected exactly ['{victim}'] "
          f"(rel {kick_rel[victim]:.3e})")

    # Which check fails if our model is replaced by zeros? Not this one -- this instrument
    # never runs a model. Its analogue is a zeroed GRADIENT stream, so that is asserted.
    zero_grads = {n: [np.zeros_like(g) for g in v] for n, v in grads.items()}
    zero_final = run_ours(w0, zero_grads, None)[-1]
    zero_rel = {n: rel(zero_final[n], theirs_ref[-1][n]) for n in zero_final}
    zero_caught = all(r > BAR for r in zero_rel.values())
    result["negative_control"]["zeroed_gradients"] = {
        "question": "which check fails if the thing under test is replaced by zeros?",
        "per_tensor_final_rel": zero_rel, "all_tensors_over_bar": zero_caught,
        "pass": zero_caught}
    ok &= zero_caught
    print(f"[{'PASS' if zero_caught else 'FAIL'}] zeroed-gradient control: all "
          f"{len(zero_rel)} tensors over bar = {zero_caught}")

    result["verdict"] = "PASS" if ok else "FAIL"
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nVERDICT: {result['verdict']}  ->  {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
