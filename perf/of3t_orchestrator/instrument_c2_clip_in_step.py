#!/usr/bin/env python3
"""PROTOCOL A12: the seam between SS4's clipping half and SS5's optimizer.

SS4 proves the clip COEFFICIENT against their real `compute_global_norm`, standalone. SS5
proves the optimizer TRAJECTORY -- and does it with `clip_norm=0.0`, clipping switched off, so
that the arm measures Adam and nothing else. Both are right about their own half, and neither
asks the question the update rule actually turns on: **does `step()` apply that coefficient, to
those gradients, before the moments?** A12(ii) says a factorisation is itself a claim. This is
that claim, for the one seam where both neighbours already pass.

Four ways it could be wrong and still pass SS4 and SS5, all of them checked here:
  - the coefficient is computed and not applied (or applied to a copy);
  - it is applied AFTER the moment update, which scales the step but not the moment history,
    so the two trajectories only separate on the second step and never converge again;
  - it is applied when it should not be, so a run that never exceeds the threshold is not
    bit-identical to a run with clipping off;
  - a disabled parameter enters the norm, which is D5's defect in the place it reaches the
    weights rather than in the coefficient.

Their side is executed, not transcribed: `compute_global_norm` from
`openfold3.core.utils.grad_manager`, the coefficient at `grad_manager.py:185-187`, then a real
`torch.optim.Adam`. Ours is the shipped `AdamW.step`, unmodified, driven through its own API.

BARS, fixed before any number here existed:
  - clip ACTIVE: relative 1e-06 per parameter per step. Same quantity and same dtype as SS5's
    optimizer bar, so it inherits SS5's justification rather than inventing a looser one.
  - clip INACTIVE: EXACT equality against a `clip_norm=0.0` run. Nothing is approximated when
    the coefficient is 1.0, so anything but bit-identity is a real disagreement.
  - the negative control must read OVER the active bar, and the binding fraction must be
    1.00 -- an arm whose clip never binds proves nothing, and that failure is silent (K54).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from openfold3.core.utils.grad_manager import compute_global_norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.train.optim import AdamW                      # noqa: E402
from tt_bio.train import optim as tt_optim                # noqa: E402

OF3 = dict(learning_rate=1.8e-3, beta1=0.9, beta2=0.95, eps=1e-8)
CLIP = 10.0
STEPS = 50
SHAPES = {"bias": (128,), "proj": (64, 64), "linear": (256, 96)}
BAR = 1e-6


class _HostValue:
    def __init__(self, arr):
        self.arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.dtype = "float32"

    def device(self):
        return None


class _Param:
    __slots__ = ("value", "grad")

    def __init__(self, arr):
        self.value = _HostValue(arr)
        self.grad = None


def _install_host_stubs():
    orig = (tt_optim.to_host, tt_optim.to_device)
    tt_optim.to_host = lambda t, dtype=None: (
        t.arr if isinstance(t, _HostValue) else orig[0](t, dtype=dtype))
    tt_optim.to_device = lambda arr, device, dtype=None, layout=None: _HostValue(arr)
    return orig


def init_weights(seed=7):
    rng = np.random.default_rng(seed)
    return {n: (rng.standard_normal(s) * 0.02).astype(np.float32) for n, s in SHAPES.items()}


def gradients(seed, scale):
    """Same sequence for both sides. `scale` decides whether the clip binds."""
    rng = np.random.default_rng(seed)
    return {n: [(rng.standard_normal(s) * scale).astype(np.float32) for _ in range(STEPS)]
            for n, s in SHAPES.items()}


def their_coef(gstep: dict, disabled=frozenset()):
    """Their global norm and clip coefficient, from their code."""
    ps = []
    for k, v in gstep.items():
        if k in disabled:
            continue
        p = torch.nn.Parameter(torch.zeros(v.shape))
        p.grad = torch.tensor(v, dtype=torch.float32)
        ps.append(p)
    gnorm, with_grad = compute_global_norm(ps)
    if not with_grad:
        return 0.0, 1.0
    mx = torch.tensor(float(CLIP))
    return float(gnorm), float(mx / torch.maximum(gnorm, mx))   # grad_manager.py:185-187


def run_theirs(w0, grads, disabled=frozenset(), clip_after=False):
    """Clip with their coefficient, then Adam. `clip_after` is the negative control."""
    params = {n: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
              for n, v in w0.items() if n not in disabled}
    opt = torch.optim.Adam(list(params.values()), lr=OF3["learning_rate"],
                           betas=(OF3["beta1"], OF3["beta2"]), eps=OF3["eps"])
    out, coefs = [], []
    for k in range(STEPS):
        gstep = {n: grads[n][k] for n in grads}
        _, coef = their_coef(gstep, disabled)
        coefs.append(coef)
        before = {n: p.detach().numpy().copy() for n, p in params.items()}
        opt.zero_grad()
        for n, p in params.items():
            g = gstep[n] if clip_after else gstep[n] * coef
            p.grad = torch.tensor(g, dtype=torch.float32)
        opt.step()
        if clip_after:
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(torch.tensor(before[n] + coef * (p.detach().numpy() - before[n])))
        out.append({n: p.detach().numpy().copy() for n, p in params.items()})
    return out, coefs


def run_ours(w0, grads, clip_norm=CLIP, disabled=(), want_master=False):
    params = {n: _Param(v) for n, v in w0.items()}
    opt = AdamW(params, lr=OF3["learning_rate"], betas=(OF3["beta1"], OF3["beta2"]),
                eps=OF3["eps"], weight_decay=0.0, clip_norm=clip_norm)
    out = []
    for k in range(STEPS):
        for n, p in params.items():
            p.grad = grads[n][k]
        opt.step(disabled=disabled)
        out.append({n: opt.master[n].copy() for n in params if n not in disabled})
    # The disabled tensors are deliberately absent from `out`, so a "did it move?" check
    # written against `out` reads True because the key is missing. That is the shape of K54 --
    # a control that passes because it does not apply -- so the masters are returned whole and
    # the check below indexes the disabled name directly, where a typo raises KeyError.
    return (out, {n: opt.master[n].copy() for n in params}) if want_master else out


def rel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def compare(ours, theirs):
    worst = {"rel": -1.0, "step": None, "tensor": None}
    for k, (o, t) in enumerate(zip(ours, theirs), start=1):
        for n in t:
            r = rel(o[n], t[n])
            if r > worst["rel"]:
                worst = {"rel": r, "step": k, "tensor": n}
    return worst


def main() -> int:
    _install_host_stubs()
    res = {"instrument": "PROTOCOL A12 -- the clip/optimizer seam",
           "bar_active": BAR, "bar_inactive": "exact", "steps": STEPS,
           "clip_norm": CLIP, "arms": {}}
    ok = True
    w0 = init_weights()

    # --- arm 1: the clip BINDS at every step ------------------------------------------
    g_big = gradients(11, 1.0)
    ours = run_ours(w0, g_big)
    theirs, coefs = run_theirs(w0, g_big)
    bound = sum(1 for c in coefs if c < 1.0) / len(coefs)
    w = compare(ours, theirs)
    arm_ok = w["rel"] <= BAR and bound == 1.0
    res["arms"]["clip_binds"] = {"worst": w, "binding_fraction": bound,
                                 "min_coef": min(coefs), "max_coef": max(coefs),
                                 "pass": arm_ok}
    ok &= arm_ok
    print(f"[{'PASS' if arm_ok else 'FAIL'}] clip active: worst relative {w['rel']:.3e} at "
          f"step {w['step']} on {w['tensor']!r}; the clip bound on "
          f"{bound*100:.0f} % of steps, coef {min(coefs):.6f}..{max(coefs):.6f}")
    if bound != 1.0:
        print("       the arm did not bind on every step -- it is measuring Adam, not clipping")

    # --- arm 2: the clip NEVER binds -> bit-identical to clipping off ------------------
    g_small = gradients(12, 1e-4)
    _, coefs_small = run_theirs(w0, g_small)
    off = run_ours(w0, g_small, clip_norm=0.0)
    on = run_ours(w0, g_small, clip_norm=CLIP)
    ident = all(np.array_equal(a[n], b[n]) for a, b in zip(off, on) for n in a)
    arm2 = ident and max(coefs_small) == 1.0
    res["arms"]["clip_inactive_is_identity"] = {
        "bit_identical": ident, "their_max_coef": max(coefs_small), "pass": arm2}
    ok &= arm2
    print(f"[{'PASS' if arm2 else 'FAIL'}] clip inactive: clip_norm={CLIP} is bit-identical to "
          f"clip_norm=0 over {STEPS} steps ({ident}), and their coefficient stayed "
          f"{max(coefs_small):.1f}")

    # --- negative control (PROTOCOL SS3e): clip applied AFTER the moments -------------
    bad, _ = run_theirs(w0, g_big, clip_after=True)
    wbad = compare(ours, bad)
    ctrl = wbad["rel"] > BAR
    res["arms"]["negative_control_clip_after_moments"] = {
        "worst": wbad, "rejected": ctrl,
        "note": "same coefficient, applied to the UPDATE instead of the gradient"}
    ok &= ctrl
    print(f"[{'PASS' if ctrl else 'FAIL'}] negative control: clipping the update instead of "
          f"the gradient reads {wbad['rel']:.3e}, {wbad['rel']/BAR:.0f}x the bar -- rejected")

    # --- arm 3: a disabled parameter is out of the norm AND out of the update ---------
    dis = ("linear",)
    ours_d = run_ours(w0, g_big, disabled=dis)
    theirs_d, coefs_d = run_theirs(w0, g_big, disabled=frozenset(dis))
    wd = compare(ours_d, theirs_d)
    # and the disabled weight itself must not have moved at all
    _, master = run_ours(w0, g_big, disabled=dis, want_master=True)
    still = bool(np.array_equal(master[dis[0]], w0[dis[0]]))
    arm3 = wd["rel"] <= BAR and still
    res["arms"]["disabled_param"] = {"worst": wd, "disabled": list(dis),
                                     "coef_range": [min(coefs_d), max(coefs_d)],
                                     "untouched": bool(still), "pass": arm3}
    ok &= arm3
    print(f"[{'PASS' if arm3 else 'FAIL'}] disabled param {dis[0]!r}: worst relative "
          f"{wd['rel']:.3e} with their coefficient recomputed over the enabled set "
          f"({min(coefs_d):.6f}..{max(coefs_d):.6f}), and the disabled weight is untouched "
          f"({still})")

    res["verdict"] = "PASS" if ok else "FAIL"
    out = Path(__file__).resolve().parent / "instrument_c2_clip_in_step.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nVERDICT: {res['verdict']}  ->  {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
