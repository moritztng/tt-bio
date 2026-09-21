#!/usr/bin/env python3
"""PROTOCOL S7 with the MODEL in the loop: 20 optimizer steps driven by the gradient the model
itself produces, not by a gradient handed to the optimizer by the harness.

`of3t-traj20` drove 4,147 parameters for 20 steps from `a[k][s] * G_ref + rho * b_p * (w_p -
w0_p)`. That is stronger than a trajectory for the four state-free factors and it is blind to
exactly one thing: the wiring BETWEEN the model and the update rule. A drive that hands the
optimizer a tensor by hand cannot say whether the gradient the model produced reached the
optimizer under the right name, whether the accumulator summed the right samples, whether
clipping saw the assembled gradient or a per-sample one, or whether step k+1's forward read the
weights step k wrote. This file closes that half.

SCOPE. `diffusion_module.diffusion_conditioning`: 26 tensors, 36.9462 % of OpenFold3's squared
gradient norm (`perf/of3t_orchestrator/SECTION_MASS_MEASURED.json`, model total
10.279642678524981). It is the largest scope in this model with all four of what a
model-in-the-loop trajectory needs: a taped device forward AND backward, a captured 0.4.3
boundary whose cotangent was VERIFIED complete against all 26 float64 reference gradients, an
upstream module runnable standalone at arbitrary weights, and a step cost that fits 20 rungs.
What stopped the larger scopes is in SCOPE_LADDER.md, measured rather than asserted.

THE LOOP IS `recipes.train_loop`'s LOOP, not a transcription of it. Our side calls the shipped
discovery (`lora.weights_for(model=...)` -> `walked_weights`), the shipped optimizer
(`train.optim.AdamW`) constructed as `recipes.py:135` constructs it, and the shipped per-sample
order:

    for each sample s:  forward -> backward(their cotangent) -> clip_and_accumulate -> zero_grad
    step() -> params.rebind()

The one thing not taken from `train_loop` is the LOSS. `tt_bio/train/losses.py` does now carry
`of3_loss_weights` and `objectives.af3_loss` exists, but the cotangent seeded here is upstream's
OWN, captured at this boundary in float64. Substituting our loss would put our loss inside the
reference and the comparison would stop being against upstream.

THE COMPARED QUANTITY IS `d_k = w_k - w_0` (S7a), scored
`||d_k_ours - d_k_ref||_2 / (||d_k_ref||_2 + 1e-30)` per tensor and over the concatenation.
`w_k` is vacuous at this step size: 20 warmup steps move the weights ~1e-4 relative, so a
relative L2 on `w_k` is dominated by a `w_0` identical on both sides by construction.

THE SAMPLE AXIS IS THE MODEL'S OWN. The reference gradient sums the single branch over 48 noise
levels and runs the pair branch once. One step here is that same total work, split into
`--per-step` accumulation samples over a disjoint partition of the 48, with the pair branch on
sample 0 alone. So the accumulation cycle partitions the model's own sample axis, which is what
gradient accumulation is, and the pair parameters land a participation count of 1 against the
single branch's `per_step` -- a spread both sides have to divide by correctly, and neither side
is told about.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_tape"))

import numpy as np                                                       # noqa: E402

CAP = "/home/ttuser/of3t_cond_cap/cond_boundary.pt"
DIFFCAP = "/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
OF3PKG = "/home/ttuser/of3t_refprec/of3pkg043"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
PREFIX = "diffusion_module.diffusion_conditioning."

MODEL_SQ_NORM = 10.279642678524981
SECTION_PCT_OF_MODEL = 36.94617946669957

# `projects/of3_all_atom/config/model_config.py:143-163`, OF3's own stage settings.
OPT = dict(learning_rate=1.8e-3, beta1=0.9, beta2=0.95, eps=1e-8)
SCHED = dict(base_lr=0.0, warmup_no_steps=1000, start_decay_after_n_steps=50000,
             decay_every_n_steps=50000, decay_factor=0.95)
CLIP_VAL = 10.0
STEPS = 20                                        # S7, fixed. Not a knob.


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load_by_path(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def partition(n_sample, per_step, steps, seed=20260921):
    """The data order: step k's disjoint partition of the 48 noise levels into `per_step`
    accumulation samples. The same sequence on every side."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(1, steps + 1):
        perm = rng.permutation(n_sample)
        out[k] = [sorted(int(x) for x in b) for b in np.array_split(perm, per_step)]
    return out


def shipped_defaults():
    """`train_loop`'s own signature, READ rather than transcribed. `of3t-wirefix` closed four
    divergences in the source; an arm that hardcodes the post-fix values can drift from the
    source without either changing."""
    import inspect
    from tt_bio.train.recipes import train_loop
    return {k: v.default for k, v in inspect.signature(train_loop).parameters.items()}


# ------------------------------------------------------------------------------ scoring (S7a)

def score_step(names, k, wo, wt, W0, nw0):
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
        num += dd
        den += tt
        onorm += float(do.ravel() @ do.ravel())
        per.append((math.sqrt(dd) / (math.sqrt(tt) + 1e-30), n, tt))
    num, den, onorm = math.sqrt(num), math.sqrt(den), math.sqrt(onorm)
    live = [x for x in per if x[2] > 0.0]
    vals = sorted(x[0] for x in live)
    worst = max(live) if live else (0.0, None, 0.0)
    # The fp32 differencing floor `of3t-updaterule` derived from the dtype and the norms alone:
    # `d_k` differences two nearly equal vectors, so forming it amplifies each side's rounding
    # by ||w_k|| / ||d_k||, and two independently rounded vectors carry sqrt(2) of it.
    floor = (math.sqrt(2.0) * u32 * (nw0 + den) / den) if den > 0 else 0.0
    rel = num / (den + 1e-30)
    return {"k": k, "rel_d": rel, "d_ours_norm": onorm, "d_theirs_norm": den,
            "rel_w": num / nw0, "fp32_differencing_floor": floor,
            "rel_over_floor": (rel / floor) if floor > 0 else 0.0,
            "median_per_tensor": (vals[len(vals) // 2] if vals else 0.0),
            "worst_per_tensor": worst[0], "worst_tensor": worst[1],
            "tensors_scored": len(live), "tensors_zero_ref": len(per) - len(live),
            "tensors_bit_identical": sum(1 for x in per if x[0] == 0.0)}


def growth(rows, lo=2):
    """S7b: the SHAPE over k = 2..20 in log-log. Linear or sub-linear passes; super-linear
    fails at any magnitude, including when every per-step reading sits inside its bar."""
    live = [r for r in rows if r["k"] >= lo and r["rel_d"] > 0.0]
    if len(live) < 2:
        return {"exponent": None, "note": "fewer than two live rungs"}
    x = np.log([r["k"] for r in live])
    y = np.log([r["rel_d"] for r in live])
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    return {"exponent": float(slope), "intercept": float(icpt),
            "r2": float(1.0 - resid.var() / y.var()) if y.var() > 0 else 1.0,
            "fitted_over_k": [r["k"] for r in live],
            "shape": ("super-linear" if slope > 1.0
                      else "linear" if slope >= 0.9 else "sub-linear"),
            "rungs_dropped_bit_identical": sum(1 for r in rows
                                               if r["k"] >= lo and r["rel_d"] == 0.0)}


# --------------------------------------------------------------------------------- their side

def build_theirs(dtype):
    """Upstream 0.4.3's own `DiffusionConditioning`, standalone, at the boundary
    `capture_cond_boundary.py` verified: it reproduces `cond_out` and, seeded with
    `cond_out_cot`, reproduces all 26 `diffusion_conditioning.*` float64 gradients.
    Constructed here exactly as that script constructs it -- their config, their entry point,
    their checkpoint -- so nothing of ours sits upstream of the reference."""
    import torch
    from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
    dc = DiffusionConditioning(**dict(cfg.architecture.diffusion_module.diffusion_conditioning))
    dc = dc.to(dtype=dtype)
    dc.train()

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    own = {k[len(PREFIX):]: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
           for k, v in sd.items() if k.startswith(PREFIX)}
    inc = dc.load_state_dict(own, strict=False)
    del ck, sd
    return dc, own, inc


def run_theirs(dtype, blocks, cap, kwargs, *, steps, warmup, log, autocast=False):
    """Upstream's assembled step: their module, their `PerSampleGradManager`, their
    `torch.optim.Adam`, their `AlphaFoldLRScheduler`, stepped in `runner.py:449-470`'s order.
    A generator yielding `{checkpoint name: fp64 numpy}` after each step."""
    import torch
    import types
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    # Their module is built FIRST, before any stub is installed: `project_entry` imports the
    # REAL `pytorch_lightning.strategies`, and a stub inserted ahead of it shadows the package
    # rather than standing in for it.
    dc, _own, _inc = build_theirs(dtype)
    _stub_upstream_deps()
    gm_mod = load_by_path("of3_grad_manager", "/home/ttuser/of3t_traj20/upstream/grad_manager.py")
    lr_mod = load_by_path("of3_lr_schedulers",
                          "/home/ttuser/of3t_traj20/upstream/lr_schedulers.py")
    params = dict(dc.named_parameters())
    per_step = len(blocks[1])

    class _Model(torch.nn.Module):
        def named_parameters(self, *a, **k):
            return iter(params.items())

    class _Strategy:
        def reduce(self, t, reduce_op=None):
            return t

    trainer = types.SimpleNamespace(strategy=_Strategy(), world_size=1, global_step=0,
                                    is_last_batch=False, accumulate_grad_batches=per_step)
    gm = gm_mod.PerSampleGradManager(gradient_clip_val=CLIP_VAL,
                                     accumulate_grad_batches=per_step)
    gm.setup(model=_Model(), trainer=trainer, logger=None)
    opt = torch.optim.Adam(list(params.values()), lr=OPT["learning_rate"],
                           betas=(OPT["beta1"], OPT["beta2"]), eps=OPT["eps"])
    sch = lr_mod.AlphaFoldLRScheduler(
        opt, last_epoch=-1, max_lr=OPT["learning_rate"], base_lr=SCHED["base_lr"],
        warmup_no_steps=warmup, start_decay_after_n_steps=SCHED["start_decay_after_n_steps"],
        decay_every_n_steps=SCHED["decay_every_n_steps"], decay_factor=SCHED["decay_factor"])

    si_cot, zij_cot = cap["si_cot"], cap["zij_cot"]
    t_all = kwargs["t"]
    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if autocast
           else torch.autocast("cpu", enabled=False))

    for k in range(1, steps + 1):
        coefs, spreads = [], None
        for s, idx in enumerate(blocks[k]):
            opt.zero_grad(set_to_none=True)
            call = dict(kwargs)
            call["t"] = t_all[:, idx].to(dtype)
            with ctx:
                si, zij = dc(**call)
            roots = [si.to(torch.float64)]
            seeds = [si_cot[:, idx].to(torch.float64)]
            if s == 0:
                # The pair branch runs ONCE per forward in the model, so it enters the
                # accumulation cycle once. That is what puts a real participation spread in
                # front of both sides' averaging.
                roots.append(zij.to(torch.float64))
                seeds.append(zij_cot.to(torch.float64))
            torch.autograd.backward(roots, seeds)
            disabled = set() if s == 0 else {n for n, p in params.items() if p.grad is None}
            gn, _ = gm_mod.compute_global_norm(
                [p for n, p in params.items() if n not in disabled])
            gn = float(gn)
            coefs.append(min(1.0, CLIP_VAL / gn) if gn > 0 else 1.0)
            gm.clip_and_accumulate(disabled_params=disabled)          # runner.py:452
        lr_now = float(opt.param_groups[0]["lr"])
        gm.sync_and_average_grads()                                   # runner.py:458
        spreads = sorted(set(gm.parameter_participation_counts.values()))
        opt.step()                                                    # runner.py:462
        sch.step()                                                    # runner.py:463
        gm.reset_accumulator()                                        # runner.py:466
        log.append({"k": k, "lr": lr_now, "per_sample_clip_coefs": coefs,
                    "participation_spread": spreads})
        yield {n: p.detach().to(torch.float64).numpy().copy() for n, p in params.items()}


def _stub_upstream_deps():
    """Let upstream's own `grad_manager.py` execute without Lightning or torchmetrics. Only the
    logging and the distributed reduce need them; the arithmetic this row reads is torch and it
    runs as written."""
    import types
    import torch

    class _Metric:
        def to(self, *a, **k):
            return self

        def update(self, *a, **k):
            pass

        def compute(self):
            return torch.tensor(0.0)

        def reset(self):
            pass

    tm = types.ModuleType("torchmetrics")
    tm.MaxMetric = tm.MeanMetric = _Metric
    pl = types.ModuleType("pytorch_lightning")
    pl.Trainer = object
    pl.loggers = types.SimpleNamespace(Logger=object)
    sys.modules.setdefault("torchmetrics", tm)
    sys.modules.setdefault("pytorch_lightning", pl)


# ----------------------------------------------------------------------------------- our side

def run_ours(blocks, cap, *, steps, warmup, log, brk="none", zero_grad_model=False,
             repin=False):
    """Our assembled step: the shipped discovery, the shipped optimizer, `train_loop`'s order.

    `brk` is the break control and it perturbs the thing THIS instrument claims to see:
      `norebind`  -- skip `params.rebind()`, so step k+1's forward reads the weights discovery
                     saw instead of the ones the optimizer wrote. An injected drive cannot see
                     this at all: it never asks the model for a gradient, so it never notices
                     that the model is standing still.
      `permute`   -- hand the optimizer the gradients under a rotated name map, so parameter p
                     is stepped by parameter q's gradient.
    """
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion import OF3DiffusionConditioning
    from tt_bio.openfold3_weights import _sub
    from tt_bio.train.lora import weights_for
    from tt_bio.train.optim import AdamW, af3_lr

    n_sample = int(cap["si_ref"].shape[1])
    n_token = int(cap["si_ref"].shape[-2])
    c_s, c_z = int(cap["si_ref"].shape[-1]), int(cap["zij_ref"].shape[-1])
    use_cond = bool(cap["use_conditioning"])

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dcsd = _sub(_sub(sd, "diffusion_module"), "diffusion_conditioning")
    shape_by_name = {k: tuple(v.shape) for k, v in dcsd.items() if torch.is_tensor(v)}

    # The bijection by transpose-invariant fingerprint, `device_cond_gradient.py`'s: `_w_tt`
    # hands `from_torch` a fresh transposed copy, so an id map over the checkpoint catches
    # nothing, and recording every `from_torch` for the duration of construction cannot miss a
    # load path by construction.
    def fingerprint(x):
        x = x.double()
        return (tuple(sorted(x.shape)), round(float(x.sum()), 9),
                round(float(x.abs().max()), 9), x.numel())

    fp_name, fp_clash = {}, set()
    for kk, v in dcsd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        f = fingerprint(v)
        if f in fp_name:
            fp_clash.add(f)
        fp_name[f] = kk

    reg = {}
    orig = ttnn.from_torch

    def recording(tensor, *a, **kw):
        v = orig(tensor, *a, **kw)
        try:
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                f = fingerprint(tensor)
                if f in fp_name and f not in fp_clash:
                    reg[id(v)] = fp_name[f]
        except Exception:
            pass
        return v

    dev = get_device()
    act = ttnn.float32
    ttnn.from_torch = recording
    try:
        with device_dtype_override(act):
            dc = OF3DiffusionConditioning(dcsd, None)
    finally:
        ttnn.from_torch = orig

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)

    z_trunk = cap["zij_trunk"].reshape(1, n_token, n_token, c_z).float()
    s_trunk = cap["si_trunk"].reshape(1, n_token, c_s).float()
    if not use_cond:
        # The reference drew `use_conditioning=False` at this step, so their own forward zeroes
        # both trunk inputs before the concat. Ours takes zeros for the same reason: it is the
        # function the captured cotangent and the published gradient were taken at.
        z_trunk = torch.zeros_like(z_trunk)
        s_trunk = torch.zeros_like(s_trunk)
    relpos_d = ft(cap["relpos"].reshape(1, n_token, n_token, -1))
    z_trunk_d, s_trunk_d = ft(z_trunk), ft(s_trunk)
    s_input_d = ft(cap["si_input"].reshape(1, n_token, -1))
    tokm = cap["token_mask"].reshape(n_token).double().float()
    tok_d = ft(tokm.reshape(1, n_token, 1))
    pair_d = ft((tokm[:, None] * tokm[None, :]).reshape(1, n_token, n_token, 1))
    n_emb = cap["n_emb"].reshape(n_sample, -1).float()
    nemb_d = [ft(n_emb[i].reshape(1, 1, -1)) for i in range(n_sample)]
    si_cot, zij_cot = cap["si_cot"], cap["zij_cot"]

    def fwd():
        with device_dtype_override(act):
            dc.pair(z_trunk_d, relpos_d, pair_d)
            dc.single(s_trunk_d, s_input_d, nemb_d[0], tok_d)

    # THE SHIPPED DISCOVERY, not a hand-built parameter dict. `weights_for(model=...)` runs one
    # forward under the grad hook and then WALKS the built model, which is what reaches a weight
    # a module fused in its own `__init__`. It also fills `params.slots`, and `slots` is the
    # whole of what makes `rebind()` able to put the optimizer's new tensor back.
    params = weights_for(fwd, None, model=dc)
    named = {path: reg.get(id(t.value)) for path, t in params.items()}
    unnamed = sorted(p for p, n in named.items() if n is None)

    # `recipes.py:135` verbatim: the arguments `train_loop` passes and no others, with the
    # values OF3's config sets. A hand-tuned construction would be a different program from
    # the one that ships.
    d = shipped_defaults()
    lr = OPT["learning_rate"]
    opt = AdamW(params, lr=lr, betas=(OPT["beta1"], OPT["beta2"]),
                weight_decay=d["weight_decay"],
                schedule=lambda s: af3_lr(s, lr, warmup_steps=warmup,
                                          plateau_until=d["plateau_until"]))
    if brk == "permute":
        order = sorted(params)
        rot = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
        opt.master = {n: opt.master[rot[n]] for n in order}

    def master_in_checkpoint_orientation():
        out = {}
        for path, nm in named.items():
            if nm is None:
                continue
            m = opt.master[path]
            want = shape_by_name[nm]
            a = m.reshape(m.shape[-len(want):]) if m.ndim > len(want) else m
            if tuple(a.shape) != want and tuple(a.shape)[::-1] == want:
                a = a.T.copy()
            out[nm] = a.astype(np.float32).copy()
        return out

    for k in range(1, steps + 1):
        coefs = []
        for s, idx in enumerate(blocks[k]):
            # One accumulation sample: the model's own forward at THIS step's weights, its own
            # backward seeded with upstream's cotangent, summed over this sample's block of the
            # noise axis by `backward` accumulating into the same leaves across tapes.
            with device_dtype_override(act), ag.tape():
                roots, seeds = [], []
                if s == 0:
                    z = dc.pair(ag.Tensor(z_trunk_d), ag.Tensor(relpos_d), ag.Tensor(pair_d))
                    roots.append(z)
                    seeds.append(ft(zij_cot[0, 0]))
                for j in idx:
                    si = dc.single(ag.Tensor(s_trunk_d), ag.Tensor(s_input_d),
                                   ag.Tensor(nemb_d[j]), ag.Tensor(tok_d))
                    roots.append(si)
                    seeds.append(ft(si_cot[0, j]))
                ag.backward(roots, seeds)
            if zero_grad_model:
                # A16: a model that computes nothing, run through the same optimizer and the
                # same scorer rather than asserted.
                for t in params.values():
                    if t.grad is not None:
                        t.grad = ttnn.multiply(t.grad, 0.0)
            coefs.append(opt.clip_and_accumulate()["clip"])
            opt.zero_grad()
        spread = sorted(set(opt.participation.values())) if opt.participation else None
        opt.step()
        moved = 0 if brk == "norebind" else params.rebind()
        # WEIGHT FRESHNESS, measured at every rung and not assumed. `AdamW.step` replaces each
        # leaf`s `value` with a fresh device tensor and `rebind()` writes that tensor back into
        # the model, but the tape resolves a parameter by the IDENTITY OF THE HANDLE
        # (`autograd._PARAMS` is keyed on `id(raw)` and `parameter_for` also checks
        # `t.value is raw`). So after a step the model holds a handle the tape has never seen.
        # This counts how many of the walked weights the tape can still resolve; anything below
        # the full set means the next forward`s backward cannot reach them.
        if repin:
            # The one-line repair, kept behind a flag because this row does not move a default:
            # hand the LEAF back to `autograd.parameter`, which re-keys the registry onto the
            # handle the optimizer just produced. `parameter()`s own docstring names this as
            # the thing a caller owes after a step; no caller in `tt_bio/` does it.
            for t in params.values():
                ag.parameter(t)
        resolved = 0
        for nm, (owner, key) in params.slots.items():
            raw = owner[key] if isinstance(owner, (dict, list)) else getattr(owner, key)
            if ag.parameter_for(raw) is not None:
                resolved += 1
        log.append({"k": k, "lr": opt.last_lr, "grad_norm": opt.last_grad_norm,
                    "clip": opt.last_clip, "per_sample": opt.last_per_sample,
                    "per_sample_clip_coefs": coefs, "rebound": moved,
                    "tape_resolves_after_step": resolved, "of_walked": len(params.slots),
                    "participation_spread": spread})
        yield master_in_checkpoint_orientation()

    # Reported out of the loop for the caller's evidence block.
    run_ours.last = {"n_params": len(params), "n_named": sum(1 for v in named.values() if v),
                     "unnamed": unnamed, "n_slots": len(params.slots),
                     "fingerprint_clashes": len(fp_clash),
                     "device_weights_walked": len(params)}


# --------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="shipped",
                    choices=["shipped", "renorm", "aa", "zero", "norebind", "permute",
                             "repin", "theirs-f32-bar"])
    ap.add_argument("--per-step", type=int, default=4)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--warmup", type=int, default=SCHED["warmup_no_steps"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import torch
    t0 = time.time()
    sys.path.insert(0, OF3PKG)

    cap = torch.load(CAP, map_location="cpu", weights_only=False)
    n_sample = int(cap["si_ref"].shape[1])
    blocks = partition(n_sample, a.per_step, a.steps)
    print(f"[{time.time()-t0:.0f}s] boundary {CAP}: {n_sample} noise levels, "
          f"{a.per_step} accumulation samples/step, blocks of "
          f"{[len(b) for b in blocks[1]]}", flush=True)

    # Their kwargs come from the diffusion boundary, exactly as `capture_cond_boundary.py`
    # takes them, because `cond_boundary.pt` holds the device port's inputs and not the
    # `batch` upstream's own module needs.
    D = torch.load(DIFFCAP, map_location="cpu", weights_only=False)
    kw = D["kwargs"]
    base_kwargs = dict(batch=kw["batch"], si_input=kw["si_input"], si_trunk=kw["si_trunk"],
                       zij_trunk=kw["zij_trunk"], use_conditioning=bool(kw["use_conditioning"]),
                       chunk_size=kw.get("chunk_size"), t=kw["t"])
    del D
    print(f"[{time.time()-t0:.0f}s] their kwargs from {DIFFCAP}, "
          f"use_conditioning={base_kwargs['use_conditioning']}", flush=True)

    ref_grad = cap["grad_f64"]
    names = sorted(ref_grad)
    W0 = None

    theirs_log, ours_log = [], []

    def f64_kwargs():
        kwv = dict(base_kwargs)
        for key in ("si_input", "si_trunk", "zij_trunk"):
            kwv[key] = kwv[key].to(torch.float64)
        return kwv

    # The reference trajectory: upstream's module and upstream's optimizer in float64.
    gt = run_theirs(torch.float64, blocks, cap, f64_kwargs(), steps=a.steps,
                    warmup=a.warmup, log=theirs_log)

    if a.arm == "theirs-f32-bar":
        # A26's reachable bar, done as a trajectory: upstream's OWN program at float32 against
        # upstream's own program at float64. Any independent implementation at fp32 reads at
        # least this, and a ratio against it is what says whether ours is as close to the ideal
        # as an implementation of equal precision would be.
        kwv = dict(base_kwargs)
        for key in ("si_input", "si_trunk", "zij_trunk"):
            kwv[key] = kwv[key].to(torch.float32)
        go = run_theirs(torch.float32, blocks, cap, kwv, steps=a.steps, warmup=a.warmup,
                        log=ours_log)
    elif a.arm == "aa":
        # A/A: the reference against itself, two independent instances of the same code over
        # the same drive. Anything but an exact zero at every rung means the instrument is not
        # deterministic and every magnitude below is unreadable.
        go = run_theirs(torch.float64, blocks, cap, f64_kwargs(), steps=a.steps,
                        warmup=a.warmup, log=ours_log)
    else:
        go = run_ours(blocks, cap, steps=a.steps, warmup=a.warmup, log=ours_log,
                      brk=("norebind" if a.arm == "norebind"
                           else "permute" if a.arm == "permute" else "none"),
                      zero_grad_model=(a.arm == "zero"),
                      repin=(a.arm == "repin"))

    rows = []
    t1 = time.time()
    for k, (wt, wo) in enumerate(zip(gt, go), start=1):
        if W0 is None:
            pass
        if k == 1:
            # `w_0` is the checkpoint, identical on both sides by construction, and it is read
            # back from THEIR step-1 state rather than assumed: lr(1) is exactly 0 under the
            # AF3 warmup, so a correct implementation has w_1 == w_0 on both sides.
            pass
        rows.append(None)
        rows[-1] = (wo, wt)
        print(f"k={k:2d} [{time.time()-t1:.0f}s]", flush=True)

    # `w_0` from the checkpoint, in checkpoint orientation, for `d_k = w_k - w_0`.
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    W0 = {n: sd[PREFIX + n].to(torch.float64).numpy().astype(np.float32).copy()
          for n in names if PREFIX + n in sd}
    names = [n for n in names if n in W0 and n in rows[0][0] and n in rows[0][1]]
    nw0 = math.sqrt(sum(float(W0[n].astype(np.float64).ravel()
                              @ W0[n].astype(np.float64).ravel()) for n in names))

    scored = [score_step(names, k, wo, wt, W0, nw0) for k, (wo, wt) in enumerate(rows, start=1)]
    for r in scored:
        print(f"k={r['k']:2d} rel_d={r['rel_d']:.6e} floor={r['fp32_differencing_floor']:.3e} "
              f"worst={r['worst_per_tensor']:.3e} ({r['worst_tensor']})", flush=True)

    ev = {}
    tp = sys.modules.get("tt_bio.taped_ttnn")
    if tp is not None:
        ev["_SOFTMAX_BW_RENORM"] = bool(tp._SOFTMAX_BW_RENORM)
    tt = sys.modules.get("tt_bio.tenstorrent")
    if tt is not None:
        ev["HOST_F64_SOFTMAX_STATS"] = dict(tt.HOST_F64_SOFTMAX_STATS)

    res = {
        "arm": a.arm,
        "scope": {"section": "diffusion_module.diffusion_conditioning",
                  "tensors_compared": len(names), "tensors_in_reference": len(ref_grad),
                  "pct_of_model_sq_grad_norm": SECTION_PCT_OF_MODEL,
                  "model_sq_norm": MODEL_SQ_NORM,
                  "elements": int(sum(W0[n].size for n in names))},
        "steps": a.steps, "accumulate_grad_batches": a.per_step,
        "warmup_no_steps": a.warmup,
        "noise_level_partition_step1": blocks[1],
        "shipped_defaults": {k: v for k, v in shipped_defaults().items()
                             if k in ("betas", "weight_decay", "plateau_until", "lr",
                                      "warmup_steps")},
        "d1": {"ours_norm": scored[0]["d_ours_norm"], "theirs_norm": scored[0]["d_theirs_norm"],
               "rel_d": scored[0]["rel_d"],
               "zero_both_sides": scored[0]["d_ours_norm"] == 0.0 == scored[0]["d_theirs_norm"]},
        "growth_k2_20": growth(scored),
        "per_step": scored,
        "our_step_log": ours_log, "their_step_log": theirs_log,
        "discovery": getattr(run_ours, "last", None),
        "flag_reach": ev,
        "env": {k: os.environ.get(k) for k in
                ("TT_BIO_SOFTMAX_BW_RENORM", "TT_BIO_HOST_F64_SOFTMAX_AB",
                 "TT_VISIBLE_DEVICES")},
        "timing_s": {"total": time.time() - t0},
        "sha256": {"cond_boundary": sha256(CAP), "checkpoint": sha256(CKPT),
                   "grad_manager.py": sha256("/home/ttuser/of3t_traj20/upstream/grad_manager.py"),
                   "lr_schedulers.py":
                       sha256("/home/ttuser/of3t_traj20/upstream/lr_schedulers.py")},
    }
    out = a.out or f"perf/of3t_modeltraj/traj_{a.arm}{a.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_step", "our_step_log", "their_step_log")},
                     indent=1, default=str)[:4000])
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
