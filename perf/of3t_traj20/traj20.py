#!/usr/bin/env python3
"""PROTOCOL §7: the assembled 20-step trajectory, at OpenFold3's real parameter scope.

§4 verified the schedule, §4 the clipping, §5 the optimizer and the §4/§5 seam, each in
isolation under controlled drive. None of that can catch WIRING: a component applied in the
wrong order, called at the wrong time, fed stale state, or not called at all. That is this
instrument's one job.

Both sides run their OWN assembled machinery over the same 4,147-tensor OpenFold3 parameter
set, from the same `w_0`, on the same sequence of samples:

  ours   `tt_bio.train.optim.AdamW`, constructed exactly as `tt_bio/train/recipes.py:118`
         constructs it and stepped exactly as `train_loop` steps it. Nothing is passed that
         the shipped caller does not pass; a hand-tuned construction would be a different
         program from the one that ships.
  theirs `torch.optim.Adam` + upstream's real `AlphaFoldLRScheduler` +
         upstream's real `PerSampleGradManager`, all three loaded from upstream's own files
         by path with their sha256 recorded, stepped in `runner.py:449-470`'s order.

THE COMPARED QUANTITY IS THE UPDATE (§7a). `d_k = w_k - w_0` per tensor, scored as
||d_k_ours - d_k_ref||_2 / (||d_k_ref||_2 + 1e-30). Comparing `w_k` is vacuous here: 20 warmup
steps move the weights ~1e-4 relative, so a relative L2 on `w_k` is dominated by a `w_0` that
is identical on both stacks by construction, and reads ~0 for a correct implementation and for
one computing garbage alike.

THE DRIVE IS CLOSED-LOOP, and that is what lets this see stale state. Sample s of step k has

    g_p = a[k][s] * G_ref_p  +  rho * b_p * (w_p - w0_p)

with `G_ref` the published float64 OpenFold3 reference gradient (real magnitudes, real
per-tensor structure, so the 10.0 clip threshold sits where it really sits), `a[k][s]` the
per-sample scalars that stand in for the data order, and the second term the feedback: each
side's step-k gradient is read off ITS OWN current weights. A trajectory driven by a fixed
gradient list cannot distinguish a stack that updates its weights from one that does not.
The feedback's share of the drive norm is MEASURED at every rung and reported; a share near
zero would mean the closed loop is decorative, so it is a control on the instrument itself.

What this deliberately does NOT do is compare the two stacks' gradients. That is instrument A
(§3), it is reported by `of3t-gradients` and `of3t-trajectory`, and mixing it in here would
bury a wiring signal under a gradient disagreement measured at 7.43. §7a-bis is the other half
of the same point: Adam cancels a uniform per-tensor gradient scaling, so no trajectory of any
length is the right instrument for gradient correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types

# Pin the thread count BEFORE numpy or torch is imported. Five of these arms sharing one
# 16-core host with each library's default (one thread per core) put the load average at 53
# and cut the per-rung rate by roughly an order of magnitude -- the work is memory-bound at
# 571 M elements, so oversubscription buys nothing and costs the cache.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np

sys.path.insert(0, os.getcwd())

from tt_bio.train import optim as tt_optim          # noqa: E402
from tt_bio.train.optim import AdamW, af3_lr        # noqa: E402

# The post-fix shipped construction, READ from `train_loop`'s signature rather than
# transcribed. `of3t-wirefix` closed four wiring divergences in the source and the arm that
# measures them has to be the source's own values, or the arm and the thing it claims to
# measure can drift apart without either changing.
def _shipped_defaults():
    import inspect
    from tt_bio.train.recipes import train_loop
    return {k: v.default for k, v in inspect.signature(train_loop).parameters.items()}

BUNDLE = "/home/ttuser/of3t/bundle_min"
UPSTREAM = "/home/ttuser/of3t_traj20/upstream"
OUT = "perf/of3t_traj20"

# `projects/of3_all_atom/config/model_config.py:143-163`, the OF3 stage settings verbatim.
OPT = dict(learning_rate=1.8e-3, beta1=0.9, beta2=0.95, eps=1e-8)
SCHED = dict(base_lr=0.0, warmup_no_steps=1000, start_decay_after_n_steps=50000,
             decay_every_n_steps=50000, decay_factor=0.95)
CLIP_VAL = 10.0
STEPS = 20                                          # §7, fixed at pass 1, not a knob


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stub_upstream_deps() -> None:
    """Let upstream's own `grad_manager.py` execute without Lightning or torchmetrics.

    Only the logging and the distributed reduce need them. The arithmetic this row reads --
    `compute_global_norm`, `_clip_grads`, `clip_and_accumulate`, `_sync_and_average_grads` --
    is torch, and it runs as written. Transcribing it instead would put a second
    implementation in the comparison, which is the thing this campaign refuses everywhere
    else.
    """
    import torch

    class _Metric:
        def to(self, *a, **k): return self
        def update(self, *a, **k): pass
        def compute(self): return torch.tensor(0.0)
        def reset(self): pass

    tm = types.ModuleType("torchmetrics"); tm.MaxMetric = tm.MeanMetric = _Metric
    pl = types.ModuleType("pytorch_lightning")
    pl.Trainer = object; pl.loggers = types.SimpleNamespace(Logger=object)
    sys.modules.setdefault("torchmetrics", tm)
    sys.modules.setdefault("pytorch_lightning", pl)
    for m in ("openfold3", "openfold3.core", "openfold3.core.utils"):
        sys.modules.setdefault(m, types.ModuleType(m))
    tu = types.ModuleType("openfold3.core.utils.tensor_utils")
    tu.tensor_tree_map = lambda fn, tree: tree
    sys.modules.setdefault("openfold3.core.utils.tensor_utils", tu)


# ------------------------------------------------------------------ the shared scope + drive

def load_scope(limit: int | None):
    """The real OpenFold3 parameter set: names from the published presence census, values
    from the published `w_0`, gradient magnitudes from the published float64 reference."""
    import torch
    presence = f"{BUNDLE}/grad_presence_recycles0.json"
    names = sorted(json.load(open(presence)))
    w0 = torch.load(f"{BUNDLE}/w0_r0_rebuild.pt", map_location="cpu", weights_only=False)
    gref = torch.load(f"{BUNDLE}/grads_f64_r0.pt", map_location="cpu", weights_only=False)
    if isinstance(gref, dict) and "grads" in gref:
        gref = gref["grads"]
    have = [n for n in names if n in w0 and n in gref and gref[n] is not None]
    dropped = [n for n in names if n not in have]
    if limit:
        have = have[::max(1, len(have) // limit)][:limit]
    W0 = {n: w0[n].to(torch.float32).numpy().copy() for n in have}
    G = {n: gref[n].to(torch.float32).numpy().copy() for n in have}
    del w0, gref
    return have, W0, G, len(names), dropped


class Drive:
    """The gradient oracle. One function, called by both sides, at each side's own weights."""

    def __init__(self, names, W0, G, *, steps, per_step, rho, seed=20260920):
        rng = np.random.default_rng(seed)
        self.names, self.W0, self.G, self.rho = names, W0, G, rho
        # The data order: one scalar per (step, sample), the same sequence on both sides.
        self.a = np.exp(rng.normal(0.0, 0.6, size=(steps + 1, per_step))).astype(np.float32)
        # A per-tensor coupling for the feedback term. Fixed once, shared by both sides.
        self.b = {n: np.float32(rng.normal(0.0, 1.0)) for n in names}

    def grad(self, k, s, weights):
        a = self.a[k, s]
        out = {}
        for n in self.names:
            g = self.G[n] * a
            if self.rho:
                g = g + (self.rho * self.b[n]) * (weights(n) - self.W0[n])
            out[n] = g
        return out

    def feedback_share(self, k, s, weights):
        """||feedback|| / ||drive||, measured. A share near 0 means the loop is not closed."""
        fb = tot = 0.0
        a = self.a[k, s]
        for n in self.names:
            f = (self.rho * self.b[n]) * (weights(n) - self.W0[n])
            g = self.G[n] * a + f
            fb += float(f.ravel() @ f.ravel()); tot += float(g.ravel() @ g.ravel())
        return math.sqrt(fb) / (math.sqrt(tot) + 1e-30)


def disabled_sets(names, steps, per_step, enabled_every):
    """Upstream disables the confidence head on any sample whose confidence weight is zero
    (`runner.py:_get_sample_disabled_param_names`), which `initial_training.yml` does on 4 of
    its 5 datasets. That is what makes the per-parameter participation counts differ, and the
    per-parameter average is the thing our `step()` never applies."""
    conf = sorted(n for n in names
                  if n.startswith("aux_heads.") and not n.startswith("aux_heads.distogram"))
    out = {}
    for k in range(1, steps + 1):
        for s in range(per_step):
            out[(k, s)] = set() if ((k + s) % enabled_every == 0) else set(conf)
    return conf, out


# --------------------------------------------------------------------------------- our side

def run_ours(names, W0, drive, dis, *, steps, per_step, warmup, variant, log, ulp=0.0,
             ulp_seed=20260921):
    """Our assembled step, constructed and driven as `tt_bio/train/recipes.py` does it.

    A generator: it yields the weight the forward reads after each step. The 20 snapshots of
    `d_k` a list would hold are 91 GB at this scope, so both sides are advanced step-locked
    and each rung is scored as it is produced.
    """
    lw = load_by_path("lr_wiring", "perf/of3t_updaterule/lr_wiring.py")
    lw.install_host_stubs()
    params = {n: lw._Param(W0[n]) for n in names}
    lr = OPT["learning_rate"]
    # `recipes.py:118` verbatim: no weight_decay, no clip_norm, no betas, no plateau_until.
    # Whatever those defaults are IS the shipped update rule, and this row is about what
    # ships. `variant` names the one deviation each arm makes, and nothing else moves.
    sched = (lambda s: af3_lr(s, lr, warmup_steps=warmup))
    kw = dict(lr=lr, schedule=sched)
    if variant in ("fixed", "fixed5", "fixed5mis"):
        # `recipes.py` AFTER of3t-wirefix, again verbatim: the arguments it now names and the
        # schedule family they select. Nothing is hand-tuned -- these come out of the shipped
        # signature, and `shipped_defaults` is recorded in the result file so the arm carries
        # the identity of the source it claims to be.
        d = _shipped_defaults()
        kw["weight_decay"] = d["weight_decay"]
        kw["schedule"] = (lambda s: af3_lr(s, lr, warmup_steps=warmup,
                                           plateau_until=d["plateau_until"]))
        # `fixed` is the FOUR divergences of3t-traj20 found, and it pins `AdamW`'s own class
        # default rather than leaving it implicit, so the arm still means what it meant after
        # the fifth fix moved what `recipes.py` passes. `fixed5` adds the fifth: upstream runs
        # beta2 = 0.95 (`model_config.py:143-146`) and Adam's library default is 0.999, which
        # is what shipped because `recipes.py` never passed betas. The pair is what attributes
        # it -- one arm cannot.
        kw["betas"] = (0.9, 0.999) if variant == "fixed" else d["betas"]
    if variant in ("wired", "wd0", "avg", "avgstale"):
        # The attribution arms. `wired` closes both gaps at once -- upstream's Adam carries no
        # weight decay and upstream clips per sample -- and `wd0` closes only the first, so the
        # pair says which of the two the divergence is.
        kw["weight_decay"] = 0.0
    opt = AdamW(params, **kw)
    if variant in ("miswire", "fixed5mis"):
        _restore_d11(opt)
    urng = np.random.default_rng(ulp_seed)
    read = lambda n: params[n].value.arr            # noqa: E731  the weight the forward reads
    for k in range(1, steps + 1):
        opt.zero_grad()
        if variant in ("wired", "avg", "avgstale", "fixed", "fixed5", "fixed5mis"):
            # Upstream's shape: clip each sample, accumulate, then step.
            # `avgstale` is the break control run on the HEALTHY arm: `stale` on the shipped
            # arm is saturated by that arm's own wiring error, so it cannot show a move.
            src = (k - 1) if (variant == "avgstale" and k > 1) else k
            coefs = []
            for s in range(per_step):
                g = drive.grad(src, s, read)
                for n in names:
                    params[n].grad = None if n in dis[(src, s)] else g[n]
                coefs.append(opt.clip_and_accumulate(disabled=dis[(src, s)])["clip"])
                opt.zero_grad()
            if variant in ("avg", "avgstale"):
                # The third gap closed IN THE HARNESS, which is what `of3t-traj20` had to do
                # because `step()` did not do it. `fixed` deliberately does NOT come through
                # here: after of3t-wirefix `step()` divides by the participation count itself,
                # so `fixed` reproducing `avg` is the check that the source fix is the arm.
                for n, c in opt.participation.items():
                    if c:
                        opt.accum[n] = opt.accum[n] / np.float32(c)
        else:
            # The shipped loop: ONE backward over the batch, so the optimizer sees the sum
            # over the samples the batch activated. `clip_and_accumulate` has no caller
            # anywhere in `tt_bio/`.
            src = (k - 1) if (variant == "stale" and k > 1) else k
            acc = {}
            for s in range(per_step):
                g = drive.grad(src, s, read)
                for n in names:
                    if n in dis[(src, s)]:
                        continue
                    acc[n] = g[n] if n not in acc else acc[n] + g[n]
            for n in names:
                params[n].grad = acc.get(n)
            coefs = None
        if ulp:
            # The rounding floor of the closed loop. Perturb the gradient the step is about
            # to consume by a relative `ulp` per element, formed in float64 and rounded back
            # to fp32 -- which is exactly the form a differently-associated fp32 expression
            # takes, landing on the same value about half the time and one ulp away the rest.
            # Done in fp32 it would inject nothing: `2**-24` relative is half an ulp and
            # rounds straight back.
            src_acc = opt.accum if opt.accum else {n: params[n].grad for n in names
                                                   if params[n].grad is not None}
            for n in src_acc:
                a = src_acc[n].astype(np.float64)
                a *= (1.0 + ulp * urng.standard_normal(a.shape))
                src_acc[n] = a.astype(np.float32)
            if not opt.accum:
                for n in src_acc:
                    params[n].grad = src_acc[n]
        opt.step()
        log.append({"k": k, "lr": opt.last_lr, "grad_norm": opt.last_grad_norm,
                    "clip": opt.last_clip, "per_sample": opt.last_per_sample,
                    "per_sample_clip_coefs": coefs})
        yield {n: params[n].value.arr.copy() for n in names}


def _restore_d11(opt):
    """The instrument-can-fail arm: put D11's off-by-one back, one line, on our side only.

    `step()` reads `self.schedule(self.steps)` BEFORE the increment, which is upstream's
    order. Shifting the callable by one restores exactly what shipped before `of3t-updaterule`
    closed D11 and nothing else, so what the arm measures is the wiring and not a second
    change riding along with it.
    """
    inner = opt.schedule
    opt.schedule = lambda s: inner(s + 1)


# ------------------------------------------------------------------------------- their side

def run_theirs(names, W0, drive, dis, *, steps, per_step, warmup, log, clip_val=CLIP_VAL):
    """Upstream's assembled step: their GradManager, their Adam, their scheduler, their order."""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    stub_upstream_deps()
    gm_mod = load_by_path("of3_grad_manager", f"{UPSTREAM}/grad_manager.py")
    lr_mod = load_by_path("of3_lr_schedulers", f"{UPSTREAM}/lr_schedulers.py")

    params = {n: torch.nn.Parameter(torch.from_numpy(W0[n].copy())) for n in names}

    class _Model(torch.nn.Module):
        def named_parameters(self, *a, **k):
            return iter(params.items())

    class _Strategy:
        def reduce(self, t, reduce_op=None):
            return t

    trainer = types.SimpleNamespace(strategy=_Strategy(), world_size=1, global_step=0,
                                    is_last_batch=False, accumulate_grad_batches=per_step)
    gm = gm_mod.PerSampleGradManager(gradient_clip_val=clip_val,
                                     accumulate_grad_batches=per_step)
    gm.setup(model=_Model(), trainer=trainer, logger=None)

    opt = torch.optim.Adam(list(params.values()), lr=OPT["learning_rate"],
                           betas=(OPT["beta1"], OPT["beta2"]), eps=OPT["eps"])
    sch = lr_mod.AlphaFoldLRScheduler(
        opt, last_epoch=-1, max_lr=OPT["learning_rate"], base_lr=SCHED["base_lr"],
        warmup_no_steps=warmup,
        start_decay_after_n_steps=SCHED["start_decay_after_n_steps"],
        decay_every_n_steps=SCHED["decay_every_n_steps"],
        decay_factor=SCHED["decay_factor"])

    read = lambda n: params[n].detach().numpy()     # noqa: E731
    for k in range(1, steps + 1):
        coefs = []
        for s in range(per_step):
            opt.zero_grad()                                       # runner.py:430
            g = drive.grad(k, s, read)
            with torch.no_grad():
                for n in names:
                    params[n].grad = (None if n in dis[(k, s)]
                                      else torch.from_numpy(g[n].copy()))
            gn, _ = gm_mod.compute_global_norm(
                [p for n, p in params.items() if n not in dis[(k, s)]])
            gn = float(gn)
            coefs.append(min(1.0, clip_val / gn) if gn > 0 else 1.0)
            gm.clip_and_accumulate(disabled_params=dis[(k, s)])   # runner.py:452
        lr_now = float(opt.param_groups[0]["lr"])
        gm.sync_and_average_grads()                               # runner.py:458
        # Read BEFORE `reset_accumulator` clears it. The spread is the whole point: a set
        # with one entry means the per-parameter average is a uniform scaling and Adam
        # cancels it, and more than one means it is not.
        spread = sorted(set(gm.parameter_participation_counts.values()))
        opt.step()                                                # runner.py:462
        sch.step()                                                # runner.py:463
        gm.reset_accumulator()                                    # runner.py:466
        log.append({"k": k, "lr": lr_now, "per_sample_clip_coefs": coefs,
                    "participation_spread": spread})
        yield {n: params[n].detach().numpy().copy() for n in names}


# ---------------------------------------------------------------------------------- scoring

def score_step(names, k, wo, wt, W0, nw0):
    """§7a, one rung. Per tensor and at model scope, on `d_k` -- and on `w_k` beside it,
    which is what shows why `w_k` alone would have been vacuous."""
    u32 = 2.0 ** -24
    num = den = onorm = 0.0
    per = []
    for n in names:
        b = W0[n].astype(np.float64)
        do = wo[n].astype(np.float64) - b
        dt = wt[n].astype(np.float64) - b
        d = do - dt
        dd = float(d.ravel() @ d.ravel())
        tt = float(dt.ravel() @ dt.ravel())
        num += dd; den += tt; onorm += float(do.ravel() @ do.ravel())
        per.append((math.sqrt(dd) / (math.sqrt(tt) + 1e-30), n, tt))
    num, den, onorm = math.sqrt(num), math.sqrt(den), math.sqrt(onorm)
    live = [x for x in per if x[2] > 0.0]
    vals = sorted(x[0] for x in live)
    worst = max(live) if live else (0.0, None, 0.0)
    # The fp32 differencing floor `of3t-updaterule` derived from the dtype and the norms
    # alone: `d_k` differences two nearly equal fp32 vectors, so forming it amplifies each
    # side's own rounding by ||w_k|| / ||d_k||, and two independently rounded vectors carry
    # sqrt(2) of it. It is a property of the dtype and the step size, not of either stack.
    floor = (math.sqrt(2.0) * u32 * (nw0 + den) / den) if den > 0 else 0.0
    rel = num / (den + 1e-30)
    return {"k": k, "rel_d": rel, "d_ours_norm": onorm, "d_theirs_norm": den,
            "rel_w": num / nw0, "fp32_differencing_floor": floor,
            "rel_over_floor": (rel / floor) if floor > 0 else 0.0,
            "median_per_tensor": (vals[len(vals) // 2] if vals else 0.0),
            "p90_per_tensor": (vals[int(0.9 * (len(vals) - 1))] if vals else 0.0),
            "worst_per_tensor": worst[0], "worst_tensor": worst[1],
            "tensors_scored": len(live), "tensors_zero_ref": len(per) - len(live),
            "tensors_bit_identical": sum(1 for x in per if x[0] == 0.0),
            # The 74 tensors whose REFERENCE update is exactly zero are a categorical bar, not
            # an approximate one: `lr*wd*theta` moves every one of them and nothing else does,
            # so with weight decay on, 0 of 4,147 were bit-identical. They are excluded from
            # `worst_per_tensor` (it maxes over the live set), so counting them needs its own
            # field -- inferring it from the totals is what `of3t-traj20` could not do.
            "tensors_zero_ref_bit_identical": sum(1 for x in per
                                                  if x[2] == 0.0 and x[0] == 0.0),
            "tensors_live_bit_identical": sum(1 for x in per if x[2] > 0.0 and x[0] == 0.0)}


def growth(rows, lo=2):
    live = [r for r in rows if r["k"] >= lo and r["rel_d"] > 0.0]
    if len(live) < 2:
        return {"exponent": None, "fitted_over_k": [], "note": "fewer than two live rungs"}
    x = np.log([r["k"] for r in live]); y = np.log([r["rel_d"] for r in live])
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    return {"exponent": float(slope), "intercept": float(icpt),
            "r2": float(1.0 - resid.var() / y.var()) if y.var() > 0 else 1.0,
            "fitted_over_k": [r["k"] for r in live],
            "rungs_dropped_bit_identical": sum(1 for r in rows
                                               if r["k"] >= lo and r["rel_d"] == 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["shipped", "scaled", "wired", "wd0", "avg", "avgstale", "miswire",
                             "stale", "fixed", "fixed5", "miswire5", "aa", "ulp"])
    ap.add_argument("--ulp", type=float, default=2.0 ** -24,
                    help="the ulp arm's per-element relative gradient perturbation")
    ap.add_argument("--limit", type=int, default=0, help="smoke only; 0 = the full scope")
    ap.add_argument("--per-step", type=int, default=0)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    warmup = SCHED["warmup_no_steps"] if a.arm == "shipped" else STEPS
    per_step = a.per_step or (1 if a.arm == "shipped" else 4)
    variant = {"shipped": "shipped", "scaled": "shipped", "wired": "wired",
               "wd0": "wd0", "avg": "avg", "avgstale": "avgstale",
               "miswire": "miswire",
               "stale": "stale",
               # `fixed` is the result: the post-fix shipped path. `ulp` runs it on both
               # sides and perturbs one, so the only thing it can read is the loop's own
               # response to rounding. `aa` runs UPSTREAM on both sides, which is the path
               # this row did not touch, and it must read exactly zero at every rung.
               "fixed": "fixed", "fixed5": "fixed5", "ulp": "fixed5",
               # The instrument-can-fail arm, moved onto the POST-FIX path: an instrument
               # that stops being able to fail once you fix things has stopped measuring.
               "miswire5": "fixed5mis",
               "aa": "shipped"}[a.arm]
    # The feedback coupling is set per arm so the closed loop carries a comparable share of
    # the drive in both, because the shipped warmup moves the weights ~1e-4 relative over 20
    # steps and the scaled one moves them ~1e-2. The share is MEASURED at three rungs and
    # published either way; a share near zero means the loop is decorative.
    rho = a.rho if a.rho is not None else (5.0 if a.arm == "shipped" else 0.005)
    # `wd0` is a `scaled`-family arm: same warmup, same accumulation, one deviation.

    t0 = time.time()
    names, W0, G, n_declared, dropped = load_scope(a.limit or None)
    elems = int(sum(v.size for v in W0.values()))
    drive = Drive(names, W0, G, steps=STEPS, per_step=per_step, rho=rho)
    conf, dis = disabled_sets(names, STEPS, per_step, enabled_every=4)
    nw0 = math.sqrt(sum(float(W0[n].astype(np.float64).ravel()
                              @ W0[n].astype(np.float64).ravel()) for n in names))
    load_s = time.time() - t0

    t1 = time.time()
    ours_log, theirs_log, rows, shares = [], [], [], {}
    if a.arm == "aa":
        # A/A: upstream against upstream, two independent instances of the same code over the
        # same drive. Anything but an exact zero at every rung means the instrument is not
        # deterministic and every magnitude this file reports is unreadable.
        gt = run_theirs(names, W0, drive, dis, steps=STEPS, per_step=per_step, warmup=warmup,
                        log=theirs_log)
        go = run_theirs(names, W0, drive, dis, steps=STEPS, per_step=per_step, warmup=warmup,
                        log=ours_log)
    elif a.arm == "ulp":
        gt = run_ours(names, W0, drive, dis, steps=STEPS, per_step=per_step, warmup=warmup,
                      variant=variant, log=theirs_log)
        go = run_ours(names, W0, drive, dis, steps=STEPS, per_step=per_step, warmup=warmup,
                      variant=variant, log=ours_log, ulp=a.ulp)
    else:
        gt = run_theirs(names, W0, drive, dis, steps=STEPS, per_step=per_step, warmup=warmup,
                        log=theirs_log)
        go = run_ours(names, W0, drive, dis, steps=STEPS, per_step=per_step, warmup=warmup,
                      variant=variant, log=ours_log)
    for k, (wt, wo) in enumerate(zip(gt, go), start=1):
        rows.append(score_step(names, k, wo, wt, W0, nw0))
        if k in (2, 10, STEPS):
            shares[k] = drive.feedback_share(k, 0, lambda n: wt[n])
        print(f"k={k:2d} rel_d={rows[-1]['rel_d']:.6e} "
              f"floor={rows[-1]['fp32_differencing_floor']:.3e} "
              f"med={rows[-1]['median_per_tensor']:.3e} "
              f"[{time.time() - t1:.0f}s]", flush=True)
    run_s = time.time() - t1

    res = {
        "arm": a.arm, "variant": variant, "warmup_no_steps": warmup,
        "accumulate_grad_batches": per_step, "rho": rho, "steps": STEPS,
        "ulp": (a.ulp if a.arm == "ulp" else 0.0),
        "shipped_defaults": {k: v for k, v in _shipped_defaults().items()
                             if k in ("betas", "weight_decay", "plateau_until", "lr",
                                      "warmup_steps")},
        "scope": {"tensors": len(names), "tensors_declared": n_declared,
                  "tensors_dropped": len(dropped), "elements": elems,
                  "confidence_head_tensors_disabled_on_3_of_4_samples": len(conf)},
        "d1": {"ours_norm": rows[0]["d_ours_norm"], "theirs_norm": rows[0]["d_theirs_norm"],
               "rel_d": rows[0]["rel_d"],
               "zero_both_sides": rows[0]["d_ours_norm"] == 0.0 == rows[0]["d_theirs_norm"]},
        "our_step_log": ours_log, "their_step_log": theirs_log,
        "feedback_share_of_drive_norm": shares,
        "growth_k2_20": growth(rows),
        "per_step": rows,
        "timing_s": {"load": load_s, "trajectory": run_s},
        "sha256": {
            "w0": sha256(f"{BUNDLE}/w0_r0_rebuild.pt"),
            "grads_f64_r0": sha256(f"{BUNDLE}/grads_f64_r0.pt"),
            "grad_manager.py": sha256(f"{UPSTREAM}/grad_manager.py"),
            "lr_schedulers.py": sha256(f"{UPSTREAM}/lr_schedulers.py"),
        },
    }
    out = a.out or f"{OUT}/traj20_{a.arm}{a.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_step", "our_step_log", "their_step_log")},
                     indent=1)[:3000])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
