#!/usr/bin/env python3
"""D11: our LR schedule was wired one step ahead of theirs. No model, no card.

PROTOCOL SS4 says the schedule is a pure function of the step index, so it can be verified
exactly over its whole domain. That was already done and it PASSED -- the closed forms agree
bit for bit. The defect is one level up, in the WIRING: `AdamW.step` incremented its counter
and then read the schedule, so our k-th update ran at `lr(k)` while upstream's runs at
`lr(k-1)`.

So this instrument deliberately does not hand either side a list of learning rates. It
constructs upstream's real `AlphaFoldLRScheduler` on a real `torch.optim.Adam` and steps them
in the order their runner steps them (`opt.step()` then `self.lr_schedulers().step()`,
`projects/of3_all_atom/runner.py:460-461`), and it reads our rate out of `AdamW.last_lr`
after each of our steps. Anything that aligns the two by hand cannot see this defect, and
PROTOCOL SS5's instrument C did align them by hand -- that is why it passed while D11 was
open.

Four arms:

  A. the closed form over its whole domain, at every knee and 1000 interior points -- bar is
     exact float64 equality (SS4), re-run here because a wiring fix must not have touched it;
  B. the wiring: the rate each side's k-th optimizer step actually applies, k = 1..2005, from
     both sides' real objects -- bar is exact float64 equality;
  C. `d_k = w_k - w_0` over k = 1..20 on an injected gradient, shipped warmup and a scaled
     warmup (SS7a). `d_1` must be exactly 0 on BOTH sides; discrimination is k = 2..20;
  D. the negative control (SS3e): put the off-by-one back, one line, and confirm B and C
     fail -- and that they fail on the rate and on `d_1`, which is what they read.

`AlphaFoldLRScheduler` is loaded from upstream's own file by path, with its sha256 recorded.
The file is byte-identical in 0.4.3 and 0.5.0, so the version question does not arise here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from tt_bio.train import optim as tt_optim  # noqa: E402
from tt_bio.train.optim import AdamW, af3_lr  # noqa: E402

OUT = "perf/of3t_updaterule"
UPSTREAM = "/home/moritz/.coworker/scratch/of3t-reference/upstream050"
LRS = f"{UPSTREAM}/openfold3/core/utils/lr_schedulers.py"
# Their `configure_optimizers` (runner.py:852-886) and the OF3 stage config it reads.
OF3 = dict(learning_rate=1.8e-3, beta1=0.9, beta2=0.95, eps=1e-8)
SCHED = dict(base_lr=0.0, warmup_no_steps=1000,
             start_decay_after_n_steps=50000, decay_every_n_steps=50000, decay_factor=0.95)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def their_scheduler_class():
    spec = importlib.util.spec_from_file_location("of3_lr_schedulers", LRS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AlphaFoldLRScheduler


# ---------------------------------------------------------------- host stand-ins for ttnn

class _HostValue:
    """A ttnn tensor handle's whole surface as `AdamW.step` uses it."""

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


def install_host_stubs():
    orig = (tt_optim.to_host, tt_optim.to_device)
    tt_optim.to_host = lambda t, dtype=None: (
        t.arr if isinstance(t, _HostValue) else orig[0](t, dtype=dtype))
    tt_optim.to_device = lambda arr, device, dtype=None, layout=None: _HostValue(arr)
    return orig


# ---------------------------------------------------------------------------- the four arms

def our_schedule(cfg):
    return lambda s: af3_lr(s, OF3["learning_rate"], warmup_steps=cfg["warmup_no_steps"],
                            decay_every_n_steps=cfg["decay_every_n_steps"],
                            decay_factor=cfg["decay_factor"], base_lr=cfg["base_lr"],
                            plateau_until=cfg["start_decay_after_n_steps"])


def arm_a_closed_form(Sched, cfg, n_interior=1000):
    """SS4: ours against theirs as functions of the step index, over the whole domain.

    Theirs is evaluated by driving the real scheduler object to each index rather than by
    transcribing `get_lr`, so a transcription error cannot be the thing that agrees.
    """
    knees = cfg["warmup_no_steps"], cfg["start_decay_after_n_steps"], cfg["decay_every_n_steps"]
    top = max(knees) * 2 + 2
    shown = (0, 1, 2, cfg["warmup_no_steps"], cfg["warmup_no_steps"] + 1,
             cfg["start_decay_after_n_steps"], cfg["start_decay_after_n_steps"] + 1)
    points = sorted({0, 1, top, *shown} | {k + d for k in (0, *knees, sum(knees))
                                           for d in (-1, 0, 1) if 0 <= k + d <= top}
                    | set(np.linspace(0, top, n_interior, dtype=np.int64).tolist()))
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=OF3["learning_rate"])
    sch = Sched(opt, last_epoch=-1, max_lr=OF3["learning_rate"],
                base_lr=cfg["base_lr"], warmup_no_steps=cfg["warmup_no_steps"],
                start_decay_after_n_steps=cfg["start_decay_after_n_steps"],
                decay_every_n_steps=cfg["decay_every_n_steps"],
                decay_factor=cfg["decay_factor"])
    ours_f = our_schedule(cfg)
    theirs, at = {}, 0
    for k in points:
        while at < k:
            sch.step()
            at += 1
        theirs[k] = float(opt.param_groups[0]["lr"])
    diffs = [(k, ours_f(k), theirs[k]) for k in points]
    bad = [(k, o, t) for k, o, t in diffs if o != t]
    return {"points": len(points), "domain": [0, top], "knees": list(knees),
            "exact_matches": len(points) - len(bad), "mismatches": len(bad),
            "worst": max((abs(o - t) for _, o, t in diffs), default=0.0),
            "first_mismatch": bad[0] if bad else None,
            "samples": {str(k): {"ours": ours_f(k), "theirs": theirs[k]} for k in shown},
            "pass": not bad}


def their_applied_rates(Sched, cfg, steps):
    """The lr their k-th `opt.step()` applies, taken off the optimizer, in their order."""
    p = torch.nn.Parameter(torch.zeros(1))
    p.grad = torch.zeros(1)
    opt = torch.optim.Adam([p], lr=OF3["learning_rate"],
                           betas=(OF3["beta1"], OF3["beta2"]), eps=OF3["eps"])
    sch = Sched(opt, last_epoch=-1, max_lr=OF3["learning_rate"],
                base_lr=cfg["base_lr"], warmup_no_steps=cfg["warmup_no_steps"],
                start_decay_after_n_steps=cfg["start_decay_after_n_steps"],
                decay_every_n_steps=cfg["decay_every_n_steps"],
                decay_factor=cfg["decay_factor"])
    out = []
    for _ in range(steps):
        out.append(float(opt.param_groups[0]["lr"]))   # what THIS opt.step() will apply
        opt.step()
        sch.step()                                      # runner.py:460-461, in that order
    return out


def our_applied_rates(cfg, steps):
    """The lr our k-th `AdamW.step()` applied, read back off the optimizer."""
    params = {"w": _Param(np.zeros(4, np.float32))}
    opt = AdamW(params, lr=OF3["learning_rate"], betas=(OF3["beta1"], OF3["beta2"]),
                eps=OF3["eps"], weight_decay=0.0, clip_norm=0.0, schedule=our_schedule(cfg))
    out = []
    for _ in range(steps):
        params["w"].grad = np.zeros(4, np.float32)
        opt.step()
        out.append(float(opt.last_lr))
    return out


def arm_b_wiring(Sched, cfg, steps=2005):
    ours = our_applied_rates(cfg, steps)
    theirs = their_applied_rates(Sched, cfg, steps)
    bad = [(k + 1, o, t) for k, (o, t) in enumerate(zip(ours, theirs)) if o != t]
    return {"steps": steps, "exact_matches": steps - len(bad), "mismatches": len(bad),
            "first_mismatch": bad[0] if bad else None,
            "worst_abs": max((abs(o - t) for o, t in zip(ours, theirs)), default=0.0),
            "first_five": [{"k": k + 1, "ours": ours[k], "theirs": theirs[k]}
                           for k in range(5)],
            "at_warmup_knee": [{"k": k, "ours": ours[k - 1], "theirs": theirs[k - 1]}
                               for k in (cfg["warmup_no_steps"], cfg["warmup_no_steps"] + 1)
                               if k <= steps],
            "pass": not bad}


def their_dk(Sched, cfg, steps, grad, w0):
    p = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32))
    opt = torch.optim.Adam([p], lr=OF3["learning_rate"],
                           betas=(OF3["beta1"], OF3["beta2"]), eps=OF3["eps"])
    sch = Sched(opt, last_epoch=-1, max_lr=OF3["learning_rate"],
                base_lr=cfg["base_lr"], warmup_no_steps=cfg["warmup_no_steps"],
                start_decay_after_n_steps=cfg["start_decay_after_n_steps"],
                decay_every_n_steps=cfg["decay_every_n_steps"],
                decay_factor=cfg["decay_factor"])
    base, out = np.asarray(w0, np.float64), []
    for k in range(steps):
        opt.zero_grad()
        p.grad = torch.tensor(grad[k], dtype=torch.float32)
        opt.step()
        sch.step()
        out.append(p.detach().numpy().astype(np.float64) - base)
    return out


def our_dk(cfg, steps, grad, w0):
    params = {"w": _Param(np.asarray(w0, np.float32))}
    opt = AdamW(params, lr=OF3["learning_rate"], betas=(OF3["beta1"], OF3["beta2"]),
                eps=OF3["eps"], weight_decay=0.0, clip_norm=0.0, schedule=our_schedule(cfg))
    base, out = np.asarray(w0, np.float64), []
    for k in range(steps):
        params["w"].grad = grad[k]
        opt.step()
        out.append(opt.master["w"].astype(np.float64) - base)
    return out


def arm_c_trajectory(Sched, cfg, steps=20, seed=4242, gate_growth=False):
    """SS7a: the compared quantity is the update, and `d_1` must be exactly 0 on both.

    `d_k = w_k - w_0` is a DIFFERENCE of two fp32 vectors that are nearly equal -- during
    warmup the weights move by ~1e-4 relative -- so forming it amplifies each side's own
    fp32 rounding by `||w_k|| / ||d_k||`. That amplification is a property of the dtype and
    the step size, not of either implementation, and it is derived here from the fp32 unit
    roundoff 2**-24 = 5.96e-08 and the measured norms alone, exactly as SS9/A7 derived the
    clipping floor. Two independently rounded vectors differenced carry sqrt(2) of it.

    So each rung is reported three ways: the relative L2 on `d_k`, the same floor the dtype
    predicts for it, and the relative L2 on `w_k` itself -- which is what shows the
    divergence lives in the differencing rather than in the trajectory.
    """
    u32 = 2.0 ** -24
    rng = np.random.default_rng(seed)
    w0 = (rng.standard_normal(64) * 0.02).astype(np.float32)
    grad = [(rng.standard_normal(64) * 1e-2).astype(np.float32) for _ in range(steps)]
    ours, theirs = our_dk(cfg, steps, grad, w0), their_dk(Sched, cfg, steps, grad, w0)
    base = np.asarray(w0, np.float64)
    rows = []
    for k, (o, t) in enumerate(zip(ours, theirs), start=1):
        nt, nw = float(np.linalg.norm(t)), float(np.linalg.norm(base + t))
        rel = float(np.linalg.norm(o - t) / (nt + 1e-30))
        floor = float(np.sqrt(2.0) * u32 * nw / nt) if nt > 0 else 0.0
        rows.append({"k": k, "d_ours_norm": float(np.linalg.norm(o)), "d_theirs_norm": nt,
                     "rel_d": rel, "fp32_differencing_floor": floor,
                     "rel_over_floor": (rel / floor) if floor > 0 else 0.0,
                     "rel_w": float(np.linalg.norm((base + o) - (base + t)) / nw)})
    disc = [r for r in rows if r["k"] >= 2]
    # SS7b: the bar is the GROWTH LAW, fitted over k = 2..20 (SS9/A9) on the divergence.
    # Rungs where the two stacks agree BIT FOR BIT are dropped from the fit rather than
    # floored to some epsilon: log(0) is not a small number, it is undefined, and clamping
    # it invents a divergence at exactly the rungs that had none. They are counted instead,
    # because "7 of 19 rungs are bit-identical" is the stronger statement anyway.
    live = [r for r in disc if r["rel_d"] > 0.0]
    slope = (float(np.polyfit(np.log([r["k"] for r in live]),
                              np.log([r["rel_d"] for r in live]), 1)[0])
             if len(live) >= 2 else 0.0)
    d1_zero = rows[0]["d_ours_norm"] == 0.0 == rows[0]["d_theirs_norm"]
    inside = all(r["rel_d"] <= r["fp32_differencing_floor"] for r in disc)
    worst = max(disc, key=lambda r: r["rel_d"])
    return {"steps": steps, "warmup_no_steps": cfg["warmup_no_steps"],
            "d1_ours": rows[0]["d_ours_norm"], "d1_theirs": rows[0]["d_theirs_norm"],
            "d1_zero_both_sides": d1_zero,
            "worst_rel_d_k2_20": worst["rel_d"], "worst_k": worst["k"],
            "worst_floor": worst["fp32_differencing_floor"],
            "worst_rel_over_floor": worst["rel_over_floor"],
            "worst_rel_w": max(r["rel_w"] for r in disc),
            "growth_exponent_k2_20": slope,
            "rungs_k2_20": len(disc), "rungs_bit_identical": len(disc) - len(live),
            "growth_fitted_over_k": [r["k"] for r in live],
            "all_rungs_inside_fp32_floor": inside,
            "growth_gated": gate_growth,
            "growth_note": ("SS7a runs the trajectory twice and says why: at the shipped "
                            "warmup the 20 updates are ~1e-4 relative, so the divergence "
                            "series sits entirely under the fp32 differencing floor and a "
                            "power-law fit over it describes rounding staircases rather "
                            "than a divergence. The growth law is therefore read off the "
                            "SCALED-warmup run, which is the run SS7a added for exactly "
                            "that reason; the shipped run is gated on d_1 and the floor and "
                            "its exponent is reported, not used."),
            "per_step": rows,
            "pass": bool(d1_zero and inside and (slope <= 1.0 or not gate_growth))}


class _ShippedOrder:
    """Put the defect back: read the schedule at `steps` AFTER the increment, one line.

    The fixed `step` reads `schedule(self.steps)` with `self.steps` still holding the count
    of COMPLETED steps, so shifting the callable by one restores exactly what shipped and
    nothing else. The two checks that must break are B (the rate the step applies) and C
    (`d_1`), because those are what they read. Arm A must NOT break -- the closed form was
    never the defect, and a control that took it down too would mean the arms are not
    measuring separate things.
    """

    def __enter__(self):
        orig = self.orig = AdamW.step

        def step(self_, **kw):
            sched = self_.schedule
            if sched is None:
                return orig(self_, **kw)
            self_.schedule = lambda s: sched(s + 1)
            try:
                return orig(self_, **kw)
            finally:
                self_.schedule = sched

        AdamW.step = step
        return self

    def __exit__(self, *exc):
        AdamW.step = self.orig
        return False


def main() -> int:
    install_host_stubs()
    Sched = their_scheduler_class()
    rep = {"defect": "D11 -- AdamW read its schedule after incrementing its step counter",
           "fix": "tt_bio/train/optim.py: read the schedule BEFORE the increment",
           "their_call_order": "opt.step(); self.lr_schedulers().step() "
                               "(projects/of3_all_atom/runner.py:460-461)",
           "their_scheduler": {"file": LRS, "sha256": sha256(LRS),
                               "identical_in_0.4.3_and_0.5.0": True},
           "their_optimizer_config": OF3, "their_schedule_config": SCHED,
           "torch": torch.__version__}
    ok = True

    rep["arm_a_closed_form"] = a = arm_a_closed_form(Sched, SCHED)
    ok &= a["pass"]
    print(f"[{'PASS' if a['pass'] else 'FAIL'}] A closed form: {a['exact_matches']}/"
          f"{a['points']} exact over steps {a['domain']}, worst abs {a['worst']:.3e}")

    rep["arm_b_wiring"] = b = arm_b_wiring(Sched, SCHED)
    ok &= b["pass"]
    print(f"[{'PASS' if b['pass'] else 'FAIL'}] B wiring: {b['exact_matches']}/{b['steps']} "
          f"applied rates exact; k=1 ours {b['first_five'][0]['ours']:.6e} theirs "
          f"{b['first_five'][0]['theirs']:.6e}")

    rep["arm_c_trajectory"] = {}
    for tag, warm, gate in (("shipped_warmup_1000", 1000, False),
                            ("scaled_warmup_20", 20, True)):
        cfg = dict(SCHED, warmup_no_steps=warm)
        c = arm_c_trajectory(Sched, cfg, gate_growth=gate)
        rep["arm_c_trajectory"][tag] = c
        ok &= c["pass"]
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] C {tag}: d_1 ours {c['d1_ours']:.6e} "
              f"theirs {c['d1_theirs']:.6e}; worst rel on d_k {c['worst_rel_d_k2_20']:.3e} "
              f"at k={c['worst_k']} against an fp32 differencing floor of "
              f"{c['worst_floor']:.3e} ({c['worst_rel_over_floor']:.2f}x); rel on w_k "
              f"{c['worst_rel_w']:.3e}; growth exponent {c['growth_exponent_k2_20']:+.3f}")

    # --- D. the negative control ---------------------------------------------------------
    with _ShippedOrder():
        ctl = {"arm_a_closed_form": arm_a_closed_form(Sched, SCHED, n_interior=200),
               "arm_b_wiring": arm_b_wiring(Sched, SCHED, steps=64),
               "arm_c_trajectory": arm_c_trajectory(Sched, SCHED)}
    broke = sorted(k for k, v in ctl.items() if not v["pass"])
    survived = sorted(k for k, v in ctl.items() if v["pass"])
    nc_ok = broke == ["arm_b_wiring", "arm_c_trajectory"] and survived == ["arm_a_closed_form"]
    rep["negative_control"] = {
        "perturbation": "the shipped order restored: schedule read at `steps` AFTER the "
                        "increment, so step k applies lr(k)",
        "broke": broke, "survived": survived,
        "expected_broken": ["arm_b_wiring", "arm_c_trajectory"],
        "expected_survived": ["arm_a_closed_form"],
        "b_first_mismatch": ctl["arm_b_wiring"]["first_mismatch"],
        "c_d1_ours": ctl["arm_c_trajectory"]["d1_ours"],
        "c_d1_theirs": ctl["arm_c_trajectory"]["d1_theirs"],
        "c_worst_rel_d_k2_20": ctl["arm_c_trajectory"]["worst_rel_d_k2_20"],
        "answers": "which check fails if the wiring is wrong? the applied rate at every k "
                   "and d_1, which reads 0 only when the first step runs at lr(0). The "
                   "closed form is untouched, which is why it is a separate arm.",
        "pass": nc_ok}
    ok &= nc_ok
    print(f"[{'PASS' if nc_ok else 'FAIL'}] D control: broke {broke}, survived {survived}; "
          f"control d_1 ours {ctl['arm_c_trajectory']['d1_ours']:.6e} against theirs "
          f"{ctl['arm_c_trajectory']['d1_theirs']:.6e}")

    rep["pass"] = bool(ok)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/lr_wiring.json", "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(f"\n{'PASS' if ok else 'FAIL'} -- wrote {OUT}/lr_wiring.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
