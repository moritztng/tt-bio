#!/usr/bin/env python3
"""PROTOCOL S7 with the model in the loop, one rung up: the WHOLE `diffusion_module`.

`of3t-modeltraj` ran the 20-step trajectory at `diffusion_module.diffusion_conditioning`,
26 tensors, 36.9462 % of the model squared gradient norm, and priced the rung above it at
5.51 h of reference-side compute. That price was taken at one noise level and four threads.
Re-measured here at the granularity the trajectory actually runs -- twelve of the 48 noise
levels per accumulation sample, twelve threads -- the same reference side reads 5.03 s per
noise level, so the rung is 1.34 h and it is affordable. Nothing else moved: N is still 20,
`d_k = w_k - w_0` is still the compared quantity, the growth law is still S7bs over k = 2..20.

THE TWO SIDES ARE DECOUPLED, and that is the only structural change from `modeltraj.py`.
There the reference generator and the device generator were zipped, so a stall on either
cost the whole run and the card was held for the reference sides hours. Here each side
runs alone, writes `w_k` for every step it finishes, and `--score` reads the two sets off
disk. A side that dies at step 14 leaves fourteen scored rungs instead of nothing, and the
card is held for the device sides ~25 min rather than the reference sides 1.34 h.

SCOPE. Their `diffusion_module` is, on our side, `OF3DiffusionConditioning` feeding
`OF3DiffusionModule` -- two classes, one scope. The composition is what
`openfold3_sample_diffusion.OF3SampleDiffusion` assembles and what `fold()` runs, so the
parameter set is the shipped ones, discovered by the shipped walk.

THE SHIPPED DEFAULT IS WHAT RUNS. `of3t-modeltraj` had to carry its repair as a `repin`
flag because `AdamW.step` replaced a leafs value and left the tapes identity-keyed
registry behind. `of3t-rebind` moved that re-keying into `autograd.Tensor.value`s setter
(965c24f52), so the default and the repaired program are now the same program and there is
no flag to set. `tape_resolves_after_step` is still counted at every step, because a fix
that is asserted rather than measured at THIS scope is not a fix at this scope.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)
import refpath                                                            # noqa: E402

CAP = "/home/ttuser/of3t_cond_cap/cond_boundary.pt"
DIFFCAP = "/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
OF3PKG = refpath.OF3PKG
REFDEPS = refpath.REFDEPS
REF_TREE = None
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
PREFIX = "diffusion_module."
SCRATCH = "/tmp/of3t/trajwide"
# The RECORD does not live in /tmp. qb2 rebooted on 2026-09-21 at 12:48Z and took a 20-step
# reference run, every steplog and every done-marker with it, because /tmp is wiped on boot on
# this host. The w_k dumps are too large for git (one rung is ~190 MB), so they go on the root
# filesystem, which survives a reboot, and the steplogs are mirrored into the branch.
RUNS = "/home/ttuser/of3t_runs/trajwide"

MODEL_SQ_NORM = 10.279642678524981          # `of3t-wholemodel`, grads_f64_043.pt

# `projects/of3_all_atom/config/model_config.py:143-163`, OF3s own stage settings.
OPT = dict(learning_rate=1.8e-3, beta1=0.9, beta2=0.95, eps=1e-8)
SCHED = dict(base_lr=0.0, warmup_no_steps=1000, start_decay_after_n_steps=50000,
             decay_every_n_steps=50000, decay_factor=0.95)
CLIP_VAL = 10.0
STEPS = 20                                        # S7, fixed. Not a knob.

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_tape"))

import numpy as np                                                       # noqa: E402


import resume as _resume


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def partition(n_sample, per_step, steps, seed=20260921):
    """`modeltraj.partition` verbatim, same seed: step ks disjoint partition of the 48 noise
    levels into `per_step` accumulation samples. The same sequence on both sides."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(1, steps + 1):
        perm = rng.permutation(n_sample)
        out[k] = [sorted(int(x) for x in b) for b in np.array_split(perm, per_step)]
    return out


def shipped_defaults():
    import inspect
    from tt_bio.train.recipes import train_loop
    return {k: v.default for k, v in inspect.signature(train_loop).parameters.items()}


def wdir(out_dir, side_arm):
    d = os.path.join(out_dir, "w", side_arm)
    os.makedirs(d, exist_ok=True)
    return d


STEPLOG_SINK = None      # set by main(): writes the steplog after every rung, not at the end


def save_step(d, k, w):
    """Atomic: `np.savez` appends `.npz`, so the part file has to end in it too."""
    tmp = os.path.join(d, f"k{k:02d}.part.npz")
    np.savez(tmp, **w)
    os.replace(tmp, os.path.join(d, f"k{k:02d}.npz"))
    if STEPLOG_SINK is not None:
        STEPLOG_SINK()


def have_steps(d):
    return sorted(int(os.path.basename(p)[1:3]) for p in glob.glob(os.path.join(d, "k??.npz")))


# ------------------------------------------------------------------------------ scoring (S7a)

def score_step(names, k, wo, wt, W0, nw0, W0o=None):
    """`modeltraj.score_step` verbatim, with one addition the wider scope forced.

    `W0o` is OUR side's own starting weights. It defaults to None, which reproduces the
    inherited single-baseline form exactly: both sides differenced against the float64
    checkpoint. That is only correct while our w_0 IS the checkpoint. At conditioning
    scope it was, since all 26 tensors there are fp32-resident. At `diffusion_module`
    scope 264 of 549 are bf16-resident, so our w_0 is bf16(checkpoint) and differencing
    us against the float64 checkpoint injects a CONSTANT 4.082229 load-quantisation
    offset into every d_k. `d_k = w_k - w_0` means each side's own w_0; pass `W0o`.
    """
    u32 = 2.0 ** -24
    num = den = onorm = 0.0
    per = []
    Bo = W0 if W0o is None else W0o
    for n in names:
        b = W0[n].astype(np.float64)
        do = wo[n].astype(np.float64) - Bo[n].astype(np.float64)
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
    """`modeltraj.growth` verbatim, S7b."""
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

def load_by_path(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_upstream_deps():
    """`modeltraj._stub_upstream_deps` verbatim: let upstream own `grad_manager.py` execute
    without Lightning or torchmetrics. Only the logging and the distributed reduce need them;
    the arithmetic this row reads is torch and it runs as written."""
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


def align_layer_norm_z(m, own, dtype):
    """Put the pair LayerNorm where THIS checkpoint keeps it, before the state dict loads.

    ON THE TREE WE ACTUALLY BUILD FROM THIS IS A NO-OP, and the `theirs` arm records that:
    `layer_norm_z_realigned = []`, 0 missing, 0 unexpected, 761 parameters. The paths below are
    the ones `refpath.assert_resolved()` returns, `<tree>/openfold3/core/model/layers/...`, with
    the line numbers read off each tree separately rather than shared between them.

    of3pkg043, upstream 0.4.3, THE TREE UNDER TEST:
      `AttentionPairBias` (attention_pair_bias.py:34) owns its own `layer_norm_z` at :107 and
      applies it at :156, which is exactly where `of3-p2-155k.pt` keeps its 24 per-block
      tensors, so the checkpoint loads TOTALLY with no rewiring. `DiffusionTransformer`
      (diffusion_transformer.py:190) does build a shared `layer_norm_z` at :254 applied at :313,
      but only `if self.use_cross_attention`, i.e. when `n_query is not None` -- the atom
      attention enc/dec stacks, which is what the comment above :312 says. The token-level
      `diffusion_transformer` passes `n_query = None`, so it has no shared norm and the loop
      below never reaches it. The atom transformers DO reach the guard, and are skipped one line
      later by the shared-variant test, because the checkpoint carries
      `atom_attn_enc/dec.atom_transformer.layer_norm_z.weight` (16 wide) for them. So `moved` is
      empty for two different reasons, not one.

    pylibs, upstream 0.5.0, NOT under test and present only as a dep tree:
      the norm is hoisted unconditionally (`diffusion_transformer.py:246`, applied at :303) and
      a separate `DiffusionAttentionPairBias` (`attention_pair_bias.py:212`) has none. Loading
      preview-2 weights there with `strict=False` leaves the shared norm at its all-ones init
      and silently DISCARDS 24 trained per-block tensors sitting 0.2982 to 0.7355 relative away
      from ones -- a different model, not a different precision. That is the 1 missing / 24
      unexpected this row measured and reported before D149, and it happened because three
      `sys.path.insert(1, ...)` calls put `pylibs` ahead of `of3pkg043`. The mechanism was the
      PATH, not an architecture gap: same checkpoint, same code here, 0.4.3 takes it whole.
      This function stays because it is what proves that, and the total-load assertion below is
      what caught it.

    The rule here is the one our own port already applies
    (`tt_bio/openfold3_diffusion_transformer.py:270-275`): the state dict picks the variant, per
    transformer instance. The atom transformers DO ship a shared norm in this checkpoint
    (`atom_attn_enc/dec.atom_transformer.layer_norm_z.weight`, 16 wide) and are left untouched;
    only the main `diffusion_transformer` is rewired. No math of upstream's is replaced: their
    `LayerNorm`, `Linear` and `Attention` still run, the norm just sits where the weights say.
    """
    import torch.nn as nn
    from openfold3.core.model.primitives.normalization import LayerNorm
    moved = []
    for p, mod in m.named_modules():
        if not (hasattr(mod, "layer_norm_z") and hasattr(mod, "blocks")):
            continue
        q = f"{p}." if p else ""
        if f"{q}layer_norm_z.weight" in own:
            continue                                  # shared variant, as upstream expects
        if f"{q}blocks.0.attention_pair_bias.layer_norm_z.weight" not in own:
            continue                                  # neither variant present, leave alone
        mod.layer_norm_z = nn.Identity()              # stop the hoisted pre-stack norm
        for b in mod.blocks:
            apb = b.attention_pair_bias
            apb.layer_norm_z = LayerNorm(apb.c_z, create_offset=False).to(dtype=dtype)
            def _prep(a, z, mask, _apb=apb, _orig=apb._prep_bias):
                return _orig(a=a, z=_apb.layer_norm_z(z), mask=mask)
            apb._prep_bias = _prep
        moved.append((p, len(mod.blocks)))
    return moved


def build_theirs(dtype):
    """Upstream 0.4.3 own `DiffusionModule`, standalone, at the captured boundary.
    `scope_ladder.build_diff` verbatim, which is `of3_all_atom/model.py:102` construction:
    the config OBJECT, not a splat, because splatting raises on `atom_attn_dec`."""
    global REF_TREE
    import torch
    from openfold3.core.model.structure.diffusion_module import DiffusionModule
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    REF_TREE = refpath.assert_resolved()

    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
    m = DiffusionModule(config=cfg.architecture.diffusion_module).to(dtype=dtype)
    m.train()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    own = {k[len(PREFIX):]: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
           for k, v in sd.items() if k.startswith(PREFIX)}
    moved = align_layer_norm_z(m, own, dtype)
    inc = m.load_state_dict(own, strict=False)
    # A reference built by loading a checkpoint into a DIFFERENT generation's module must prove
    # the load was TOTAL before any number taken against it means anything. This was a printed
    # count for a whole pass while the reference ran an untrained all-ones LayerNorm in 24 blocks.
    if inc.missing_keys or inc.unexpected_keys:
        raise SystemExit(
            f"reference load is not total: {len(inc.missing_keys)} missing, "
            f"{len(inc.unexpected_keys)} unexpected. missing={list(inc.missing_keys)[:8]} "
            f"unexpected={list(inc.unexpected_keys)[:8]}. The module and the checkpoint are "
            f"different architecture generations; fix align_layer_norm_z rather than relaxing "
            f"this check.")
    del ck, sd
    return m, own, inc, moved


def dropout_census(m):
    """A trajectory over a module in `train()` mode is reproducible only if nothing in it
    draws. Counted rather than assumed: a non-zero rate makes every rung below a sample of a
    distribution, and the A/A arm is what would catch it."""
    import torch
    rates = {}
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.modules.dropout._DropoutNd):
            rates[n] = float(mod.p)
    return {"modules": len(rates), "nonzero": sorted(n for n, v in rates.items() if v > 0.0),
            "max_rate": (max(rates.values()) if rates else 0.0)}


def run_theirs(dtype, blocks, cot, kwargs, *, steps, warmup, log, d_out, also_second=False):
    """Upstream assembled step: their module, their `PerSampleGradManager`, their
    `torch.optim.Adam`, their `AlphaFoldLRScheduler`, stepped in `runner.py:449-470` order.

    One accumulation sample is one call of the module over that sample block of the noise
    axis, seeded with their own captured cotangent over the same block. Splitting the 48
    levels across four calls recomputes the pair branch four times instead of once and changes
    nothing about the gradient: a pair parameter gradient is the sum over the levels whose
    cotangent reaches it, however the levels are grouped.

    `also_second` builds a SECOND independent instance of the same code over the same drive
    and steps it beside the first, which is the A/A in one process: anything but an exact zero
    at every rung means the instrument is not deterministic and every magnitude is unreadable.
    """
    import torch
    import types
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))

    def assemble():
        m, _own, inc, zn_moved = build_theirs(dtype)
        _stub_upstream_deps()
        gm_mod = load_by_path("of3_grad_manager",
                              "/home/ttuser/of3t_traj20/upstream/grad_manager.py")
        lr_mod = load_by_path("of3_lr_schedulers",
                              "/home/ttuser/of3t_traj20/upstream/lr_schedulers.py")
        params = dict(m.named_parameters())
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
        return dict(m=m, inc=inc, zn_moved=zn_moved, gm=gm, gm_mod=gm_mod, opt=opt,
                    sch=sch, params=params)

    A = assemble()
    census = dropout_census(A["m"])
    print(f"dropout census: {json.dumps(census)}", flush=True)
    print(f"layer_norm_z realigned to the checkpoint's variant: {A['zn_moved']}", flush=True)
    print(f"load_state_dict: {len(A['inc'].missing_keys)} missing, "
          f"{len(A['inc'].unexpected_keys)} unexpected", flush=True)
    B = assemble() if also_second else None

    t_all, xl_all = kwargs["t"], kwargs["xl_noisy"]
    t0 = time.time()
    aa_rows = []

    def one_step(E, k):
        params, gm, gm_mod = E["params"], E["gm"], E["gm_mod"]
        coefs = []
        for s, idx in enumerate(blocks[k]):
            E["opt"].zero_grad(set_to_none=True)
            call = dict(kwargs)
            call["t"] = t_all[:, idx].to(dtype)
            call["xl_noisy"] = xl_all[:, idx].to(dtype)
            out = E["m"](**call)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.autograd.backward([out], [cot[:, idx].to(dtype).reshape(out.shape)])
            disabled = {n for n, p in params.items() if p.grad is None}
            gn, _ = gm_mod.compute_global_norm(
                [p for n, p in params.items() if n not in disabled])
            gn = float(gn)
            coefs.append(min(1.0, CLIP_VAL / gn) if gn > 0 else 1.0)
            gm.clip_and_accumulate(disabled_params=disabled)          # runner.py:452
        lr_now = float(E["opt"].param_groups[0]["lr"])
        gm.sync_and_average_grads()                                   # runner.py:458
        spreads = sorted(set(gm.parameter_participation_counts.values()))
        E["opt"].step()                                               # runner.py:462
        E["sch"].step()                                               # runner.py:463
        gm.reset_accumulator()                                        # runner.py:466
        return dict(lr=lr_now, coefs=coefs, grad_norm_last_sample=gn,
                    n_disabled_last_sample=len(disabled), participation_spread=spreads)

    k0 = _resume.load_theirs(d_out, A, B, log, aa_rows)
    if k0:
        t0 -= log[-1]["wall_s"]          # wall_s stays cumulative across a resumed arm
        print(f"theirs: resuming at k={k0 + 1}", flush=True)
    for k in range(k0 + 1, steps + 1):
        row = one_step(A, k)
        wa = {n: p.detach().to(torch.float64).numpy().astype(np.float32)
              for n, p in A["params"].items()}
        if B is not None:
            one_step(B, k)
            wb = {n: p.detach().to(torch.float64).numpy().astype(np.float32)
                  for n, p in B["params"].items()}
            same = sum(1 for n in wa if np.array_equal(wa[n], wb[n]))
            aa_rows.append({"k": k, "tensors": len(wa), "bit_identical": same,
                            "max_abs_diff": max(float(np.abs(wa[n] - wb[n]).max())
                                                for n in wa)})
            print(f"  A/A k={k}: {same} of {len(wa)} bit-identical, "
                  f"max|d|={aa_rows[-1]['max_abs_diff']:.3e}", flush=True)
        row.update(k=k, wall_s=time.time() - t0)
        log.append(row)
        save_step(d_out, k, wa)
        _resume.save_theirs(d_out, k, A, B, log, aa_rows)
        print(f"[{time.time()-t0:.0f}s] their k={k:2d} lr={row['lr']:.6e} "
              f"clip={row['coefs']} spread={row['participation_spread']}", flush=True)

    run_theirs.last = {
        "ref_tree": REF_TREE,
        "n_parameters": len(A["params"]),
        "n_elements": int(sum(p.numel() for p in A["params"].values())),
        "dropout": census, "layer_norm_z_realigned": A["zn_moved"],
        "missing_keys": len(A["inc"].missing_keys),
        "unexpected_keys": len(A["inc"].unexpected_keys), "aa_in_process": aa_rows}


# ----------------------------------------------------------------------------------- our side

def build_ours(cap, kw, act, brk):
    """Our `diffusion_module`: `OF3DiffusionConditioning` feeding `OF3DiffusionModule`, which
    is what `openfold3_sample_diffusion.OF3SampleDiffusion` assembles and what `fold()` runs.

    The bijection device tensor -> checkpoint name is `device_gradient.py` verbatim: record
    every `ttnn.from_torch` for the duration of construction, match the tensor it was handed
    back to the checkpoint by a transpose-invariant fingerprint, and record the ORIENTATION
    bit-exactly at the load rather than inferring it from a shape at comparison time. D83: the
    shape test cannot fire on a square weight and 87 of the compared tensors are square.
    """
    import math as _m
    import torch
    import torch.nn.functional as F
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion import OF3DiffusionConditioning
    from tt_bio.openfold3_diffusion_module import OF3DiffusionModule
    from tt_bio.openfold3_weights import _sub
    from tt_bio.openfold3_fold import build_dm_device_aux
    from tt_bio import openfold3_host_prep as HP
    from tt_bio._vendor.openfold3.core.utils.atom_attention_block_utils import (
        get_block_indices, get_pair_atom_block_mask, get_query_block_padding)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dmsd = _sub(sd, "diffusion_module")
    shape_by_name = {k: tuple(v.shape) for k, v in dmsd.items() if torch.is_tensor(v)}

    def fingerprint(x):
        x = x.double()
        return (tuple(sorted(x.shape)), round(float(x.sum()), 9),
                round(float(x.abs().max()), 9), x.numel())

    fp_name, fp_clash = {}, set()
    for k, v in dmsd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        f = fingerprint(v)
        if f in fp_name:
            fp_clash.add(f)
        fp_name[f] = k

    reg, orient = {}, {}
    orig = ttnn.from_torch

    def recording(tensor, *args, **kwargs):
        v = orig(tensor, *args, **kwargs)
        try:
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                f = fingerprint(tensor)
                if f in fp_name and f not in fp_clash:
                    nm = fp_name[f]
                    reg[id(v)] = nm
                    ck = dmsd[nm]
                    same = tensor.shape == ck.shape and bool(torch.equal(tensor, ck))
                    flip = (tensor.dim() == 2 and ck.dim() == 2
                            and tensor.shape == ck.t().shape
                            and bool(torch.equal(tensor, ck.t().contiguous())))
                    orient[id(v)] = "N" if same else ("T" if flip else "?")
        except Exception:
            pass
        return v

    batch = kw["batch"]
    sq = lambda x: (x.reshape(x.shape[2:]) if x.dim() > 2 and x.shape[0] == 1 and x.shape[1] == 1
                    else x.squeeze(0))
    n_atom = int(kw["atom_mask"].shape[-1])
    n_token = int(kw["token_mask"].shape[-1])
    n_struct = int(kw["xl_noisy"].shape[1])
    sigma_data = float(dmsd.get("diffusion_module.sigma_data", torch.tensor(16.0)))

    atom_mask = sq(batch["atom_mask"]).float()
    a2t = sq(batch["atom_to_token_index"]).long()
    token_mask = sq(batch["token_mask"]).float()
    N_QUERY, N_KEY = 32, 128
    nb = _m.ceil(n_atom / N_QUERY)
    NP = nb * N_QUERY
    n_tok_pad = _m.ceil(n_token / 32) * 32
    pad_right = get_query_block_padding(n_atom, N_QUERY)
    key_block_idxs, invalid_mask = get_block_indices(
        atom_mask=atom_mask, n_query=N_QUERY, n_key=N_KEY, device=torch.device("cpu"))
    mask_trunked = get_pair_atom_block_mask(
        atom_mask=atom_mask, num_blocks=nb, n_query=N_QUERY, n_key=N_KEY,
        pad_len_right_q=pad_right, key_block_idxs=key_block_idxs, invalid_mask=invalid_mask)
    npe_q = F.pad(a2t, (0, pad_right), value=0).reshape(nb, N_QUERY).long()
    npe_k = torch.gather(a2t.unsqueeze(0).expand(nb, n_atom), 1, key_block_idxs.long())
    zij_mask = ((~invalid_mask).float())[:, None, :].expand(nb, N_QUERY, N_KEY) * mask_trunked
    a2t_mean = torch.zeros(n_token, n_atom)
    a2t_mean[a2t, torch.arange(n_atom)] = atom_mask
    a2t_mean = a2t_mean / a2t_mean.sum(-1, keepdim=True).clamp_min(1.0)
    feats = {"ref_pos": sq(batch["ref_pos"]).float(), "atom_mask": atom_mask,
             "ref_charge": sq(batch["ref_charge"]).float(),
             "ref_mask": sq(batch["ref_mask"]).float(),
             "ref_element": sq(batch["ref_element"]).float(),
             "ref_atom_name_chars": sq(batch["ref_atom_name_chars"]).float(),
             "ref_space_uid": sq(batch["ref_space_uid"]).float()}
    cl0, plm0 = HP.ref_atom_embed(_sub(_sub(dmsd, "atom_attn_enc"),
                                       "ref_atom_feature_embedder"), feats)

    dev = get_device()
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)
    aux = build_dm_device_aux(
        dev, ft, cl0=cl0, plm0=plm0, atom_mask=atom_mask, atom_to_token_index=a2t,
        npe_q_indices=npe_q, npe_k_indices=npe_k, zij_mask=zij_mask,
        key_block_idxs=key_block_idxs, invalid_mask=invalid_mask, mask_trunked=mask_trunked,
        atom_to_token_mean=a2t_mean, token_mask=token_mask, n_atom=n_atom, n_token=n_token,
        nb=nb, NP=NP, n_tok_pad=n_tok_pad)

    ttnn.from_torch = recording
    try:
        with device_dtype_override(act):
            dc = OF3DiffusionConditioning(_sub(dmsd, "diffusion_conditioning"), None)
            dm = OF3DiffusionModule(dmsd, None)
    finally:
        ttnn.from_torch = orig

    # The conditioning inputs, `modeltraj.run_ours` verbatim: the trunk pair and single are
    # zeroed when the reference drew `use_conditioning=False`, because that is the function
    # the captured cotangent and the published gradient were taken at.
    use_cond = bool(cap["use_conditioning"])
    c_s = int(cap["si_ref"].shape[-1])
    c_z = int(cap["zij_ref"].shape[-1])
    z_trunk = cap["zij_trunk"].reshape(1, n_token, n_token, c_z).float()
    s_trunk = cap["si_trunk"].reshape(1, n_token, c_s).float()
    if not use_cond:
        z_trunk = torch.zeros_like(z_trunk)
        s_trunk = torch.zeros_like(s_trunk)
    tokm = cap["token_mask"].reshape(n_token).double().float()
    n_emb = cap["n_emb"].reshape(n_struct, -1).float()
    d = dict(
        dev=dev, ft=ft, act=act, aux=aux, dc=dc, dm=dm, reg=reg, orient=orient,
        shape_by_name=shape_by_name, n_atom=n_atom, n_token=n_token, n_struct=n_struct,
        n_tok_pad=n_tok_pad, nb=nb, NP=NP, sigma_data=sigma_data, atom_mask=atom_mask,
        relpos_d=ft(cap["relpos"].reshape(1, n_token, n_token, -1)),
        z_trunk_d=ft(z_trunk), s_trunk_d=ft(s_trunk),
        s_input_d=ft(cap["si_input"].reshape(1, n_token, -1)),
        tok_d=ft(tokm.reshape(1, n_token, 1)),
        pair_d=ft((tokm[:, None] * tokm[None, :]).reshape(1, n_token, n_token, 1)),
        nemb_d=[ft(n_emb[i].reshape(1, 1, -1)) for i in range(n_struct)],
        si_trunk_dm=ft(sq(kw["si_trunk"]).float().unsqueeze(0)),
        t_all=kw["t"], xl_all=kw["xl_noisy"], use_conditioning=use_cond)
    return d


def run_ours(G, blocks, cot, *, steps, warmup, log, d_out, brk="none",
             zero_grad_model=False):
    """Our assembled step: the shipped discovery, the shipped optimizer, `train_loop` order.

    ONE TAPE PER NOISE LEVEL, and the whole composition inside it: the pair branch, the single
    branch and the diffusion module. The 48 levels of one accumulation sample accumulate into
    the leaves across tapes, which is what `device_gradient.py` accumulation probe verified.
    Recomputing the pair branch per level rather than per call costs device time and changes no
    gradient: the pair parameters gradient is the sum over the levels whose cotangent reaches
    them, and splitting that sum differently does not move it.
    """
    import math as _m
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import device_dtype_override
    from tt_bio.train.lora import weights_for
    from tt_bio.train.optim import AdamW, af3_lr
    from tt_bio.train.tensors import to_device as _to_device

    dc, dm, ft, act, aux = G["dc"], G["dm"], G["ft"], G["act"], G["aux"]
    n_atom, NP, nb = G["n_atom"], G["NP"], G["nb"]
    n_token, n_tok_pad = G["n_token"], G["n_tok_pad"]
    sigma_data, atom_mask = G["sigma_data"], G["atom_mask"]
    t_all, xl_all = G["t_all"], G["xl_all"]

    class _Composed:
        """One object the shipped walk can reach both halves through. `walk_device_weights`
        walks `__dict__`, so this adds one level of nesting and nothing else."""

        def __init__(self, dc, dm):
            self.dc = dc
            self.dm = dm

    composed = _Composed(dc, dm)

    def level(j, taped):
        tk = float(t_all[0, j])
        xl_k = xl_all[0, j].float()
        rl_k = (xl_k * atom_mask[:, None]) / _m.sqrt(tk * tk + sigma_data ** 2)
        rl_pad = torch.zeros(NP, 3)
        rl_pad[:n_atom] = rl_k
        T = (lambda x: ag.Tensor(x)) if taped else (lambda x: x)
        zij = dc.pair(T(G["z_trunk_d"]), T(G["relpos_d"]), T(G["pair_d"]))
        si = dc.single(T(G["s_trunk_d"]), T(G["s_input_d"]), T(G["nemb_d"][j]), T(G["tok_d"]))
        return dm(T(G["si_trunk_dm"]), si, zij, T(aux["cl0_d"]), T(aux["plm0_d"]),
                  T(ft(rl_pad.unsqueeze(0))), T(ft(xl_k.unsqueeze(0))),
                  aux["amc_d"], aux["amc_na_d"], aux["idx_tt"], aux["flat_tt"],
                  aux["zij_mask_d"], aux["kidx_tt"], aux["valid_d"], aux["mb_d"],
                  aux["pm_d"], aux["mean_d"], aux["tok_pad_tt"], aux["tok_col_pad_tt"],
                  n_atom, NP, nb, n_token, n_tok_pad, tk, sigma_data)

    def fwd():
        with device_dtype_override(act):
            level(0, False)

    params = weights_for(fwd, None, model=composed)
    named = {path: G["reg"].get(id(t.value)) for path, t in params.items()}
    orient_by_name = {G["reg"][i]: o for i, o in G["orient"].items() if i in G["reg"]}
    unnamed = sorted(p for p, n in named.items() if n is None)
    print(f"discovery: {len(params)} device weights walked, "
          f"{sum(1 for v in named.values() if v)} carry a checkpoint name, "
          f"{len(params.slots)} slots", flush=True)

    d = shipped_defaults()
    lr = OPT["learning_rate"]
    opt = AdamW(params, lr=lr, betas=(OPT["beta1"], OPT["beta2"]),
                weight_decay=d["weight_decay"],
                schedule=lambda s: af3_lr(s, lr, warmup_steps=warmup,
                                          plateau_until=d["plateau_until"]))
    permute_report = None
    if brk == "permute":
        # `modeltraj` repaired control: rotate WITHIN each shape class. A rotation over all
        # names hands a (384,768) slot the wrong element count and the optimizer raises before
        # it ever steps, and a control that raises has tested nothing.
        order = sorted(params)
        by_shape = {}
        for n in order:
            by_shape.setdefault(tuple(opt.master[n].shape), []).append(n)
        rot, moved_names, singleton = {}, [], []
        for shp, nms in by_shape.items():
            if len(nms) < 2:
                singleton.extend(nms)
                rot[nms[0]] = nms[0]
                continue
            for i, n in enumerate(nms):
                rot[n] = nms[(i + 1) % len(nms)]
                moved_names.append(n)
        opt.master = {n: opt.master[rot[n]] for n in order}
        permute_report = {"tensors_total": len(order), "tensors_permuted": len(moved_names),
                          "tensors_left_in_place_unique_shape": len(singleton),
                          "shape_classes": len(by_shape),
                          "shape_classes_rotatable": sum(1 for v in by_shape.values()
                                                         if len(v) > 1)}
        print(f"permute control: {json.dumps(permute_report)}", flush=True)

    def master_in_checkpoint_orientation():
        out = {}
        for path, nm in named.items():
            if nm is None:
                continue
            m = opt.master[path]
            want = G["shape_by_name"][nm]
            a = m.reshape(m.shape[-len(want):]) if m.ndim > len(want) else m
            o = orient_by_name.get(nm, "?")
            if o == "T" and a.ndim == 2:
                a = a.T
            elif o != "N" and tuple(a.shape) != want and tuple(a.shape)[::-1] == want:
                a = a.T
            if tuple(a.shape) != want:
                raise AssertionError(f"{nm}: master {tuple(a.shape)} is not {want} after "
                                     f"orientation {o}")
            out[nm[len(PREFIX):] if nm.startswith(PREFIX) else nm] = a.astype(np.float32).copy()
        return out

    t0 = time.time()
    stale_hold = None
    fwd_rel = None
    k0, stale_hold, fwd_rel = _resume.load_ours(d_out, opt, params, _to_device, log,
                                                master_in_checkpoint_orientation)
    if k0:
        t0 -= log[-1]["wall_s"]          # wall_s stays cumulative across a resumed arm
        print(f"ours: resuming at k={k0 + 1}", flush=True)
    for k in range(k0 + 1, steps + 1):
        coefs = []
        for s, idx in enumerate(blocks[k]):
            for j in idx:
                with device_dtype_override(act), ag.tape():
                    out = level(j, True)
                    ag.backward([out], [ft(cot[0, j].float().unsqueeze(0))])
                if k == 1 and s == 0 and fwd_rel is None:
                    o = ttnn.to_torch(out.value if hasattr(out, "value") else out).double()
                    r = G["out_ref"][0, j].double().reshape(-1)
                    o = o.reshape(-1)[:r.numel()]
                    fwd_rel = float(torch.linalg.vector_norm(o - r)
                                    / (torch.linalg.vector_norm(r) + 1e-300))
                    print(f"forward rel at level {j}: {fwd_rel:.6e}", flush=True)
            if zero_grad_model:
                # A16: a model that computes nothing, through the same optimizer and the same
                # scorer rather than asserted.
                for t in params.values():
                    if t.grad is not None:
                        t.grad = ttnn.multiply(t.grad, 0.0)
            coefs.append(opt.clip_and_accumulate()["clip"])
            opt.zero_grad()
        spread = sorted(set(opt.participation.values())) if opt.participation else None
        prev_master = {n: v.copy() for n, v in opt.master.items()} if brk == "stale" else None
        opt.step()
        if brk == "stale" and stale_hold is not None:
            for n, t in params.items():
                t.value = _to_device(stale_hold[n], t.value.device(), dtype=t.value.dtype)
        if brk == "stale":
            stale_hold = prev_master
        moved = 0 if brk == "norebind" else params.rebind()
        # WEIGHT FRESHNESS, measured at every rung. `of3t-rebind` moved the re-keying into
        # `autograd.Tensor.value` setter, so the shipped default should hold every slot; a
        # count below the full set is that defect recurring at this scope.
        resolved = 0
        for nm, (owner, key) in params.slots.items():
            raw = owner[key] if isinstance(owner, (dict, list)) else getattr(owner, key)
            if ag.parameter_for(raw) is not None:
                resolved += 1
        log.append({"k": k, "lr": opt.last_lr, "grad_norm": opt.last_grad_norm,
                    "clip": opt.last_clip, "per_sample": opt.last_per_sample,
                    "per_sample_clip_coefs": coefs, "rebound": moved,
                    "tape_resolves_after_step": resolved, "of_walked": len(params.slots),
                    "participation_spread": spread, "wall_s": time.time() - t0})
        save_step(d_out, k, master_in_checkpoint_orientation())
        _resume.save_ours(d_out, k, opt, log, stale_hold, fwd_rel)
        print(f"[{time.time()-t0:.0f}s] our k={k:2d} lr={opt.last_lr:.6e} "
              f"grad_norm={opt.last_grad_norm} resolves={resolved}/{len(params.slots)} "
              f"rebound={moved} spread={spread}", flush=True)

    run_ours.last = {"n_params": len(params), "n_named": sum(1 for v in named.values() if v),
                     "unnamed": unnamed[:40], "n_unnamed": len(unnamed),
                     "n_slots": len(params.slots),
                     "device_weights_walked": len(params),
                     "forward_rel_level0": fwd_rel,
                     "orientation_recorded_at_load": {
                         o: sum(1 for v in orient_by_name.values() if v == o)
                         for o in ("N", "T", "?")},
                     "permute_control": permute_report}


# --------------------------------------------------------------------------------------- main

def load_w(d, k):
    with np.load(os.path.join(d, f"k{k:02d}.npz")) as z:
        return {n: z[n] for n in z.files}


def do_score(a, out_dir):
    import torch
    dt = wdir(out_dir, a.theirs_arm)
    do = wdir(out_dir, a.arm)
    kt, ko = set(have_steps(dt)), set(have_steps(do))
    ks = sorted(kt & ko)
    if not ks:
        raise SystemExit(f"nothing to score: theirs has {sorted(kt)}, ours has {sorted(ko)}")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    w1t, w1o = load_w(dt, ks[0]), load_w(do, ks[0])
    names = sorted(n for n in w1t
                   if n in w1o and PREFIX + n in sd
                   and tuple(sd[PREFIX + n].shape) == tuple(w1t[n].shape))
    W0 = {n: sd[PREFIX + n].to(torch.float64).numpy().astype(np.float32) for n in names}
    nw0 = math.sqrt(sum(float(W0[n].astype(np.float64).ravel()
                              @ W0[n].astype(np.float64).ravel()) for n in names))
    # lr(1) is exactly 0, so each side's k=1 dump IS its own w_0. Verified, not assumed:
    # over the 549 scored tensors our k=1 is bit-exactly the checkpoint on 285 and
    # bit-exactly bf16(checkpoint) on 264, with none left over, so our side moved nothing
    # at k=1 and this baseline is our starting point rather than a step of it.
    W0o = load_w(do, ks[0]) if a.w0 == "own" else None
    scored = []
    for k in ks:
        wt, wo = load_w(dt, k), load_w(do, k)
        r = score_step(names, k, wo, wt, W0, nw0, W0o)
        scored.append(r)
        print(f"k={r['k']:2d} rel_d={r['rel_d']:.6e} floor={r['fp32_differencing_floor']:.3e} "
              f"worst={r['worst_per_tensor']:.3e} ({r['worst_tensor']})", flush=True)
    return scored, names, W0, nw0, ks


def scope_share(names):
    """The scope as a SHARE OF THE SQUARED GRADIENT NORM, measured against the reference
    gradients this trajectory is driven by, with everything outside it named."""
    import torch
    D = torch.load(DIFFCAP, map_location="cpu", weights_only=False)
    g = D["grad_f64"]
    tot = sum(float(v.double().pow(2).sum()) for v in g.values() if torch.is_tensor(v))
    got = sum(float(g[n].double().pow(2).sum()) for n in names if n in g)
    missing = sorted(set(g) - set(names))
    # Every name, not a sample. The brief asks for everything outside the scope listed NOT
    # COVERED and why, and a list truncated at 40 of 188 answers neither question. The
    # families carry the "why": each one is a share of the squared gradient norm, so a
    # reader can see what the gap costs instead of counting tensors.
    fam = {}
    for n in missing:
        k = re.sub(r"\.\d+\.", ".N.", n)
        k = k[:k.index(".N.") + 3] if ".N." in k else k.rsplit(".", 1)[0]
        e = fam.setdefault(k, {"n": 0, "sq": 0.0})
        e["n"] += 1
        e["sq"] += float(g[n].double().pow(2).sum())
    for e in fam.values():
        e["pct_of_model_sq_grad_norm"] = 100.0 * e["sq"] / MODEL_SQ_NORM
    return {"reference_tensors_at_this_boundary": len(g),
            "tensors_scored": len(names),
            "tensors_in_reference_not_scored": missing,
            "not_covered_families": dict(sorted(fam.items(),
                                                key=lambda kv: -kv[1]["sq"])),
            "n_tensors_in_reference_not_scored": len(missing),
            "diffusion_module_sq_norm": tot,
            "scored_sq_norm": got,
            "pct_of_diffusion_module": 100.0 * got / tot if tot else None,
            "model_sq_norm": MODEL_SQ_NORM,
            "pct_of_model_sq_grad_norm": 100.0 * got / MODEL_SQ_NORM,
            "not_covered_pct_of_model": 100.0 * (MODEL_SQ_NORM - got) / MODEL_SQ_NORM}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="", choices=["", "theirs", "ours"])
    ap.add_argument("--arm", default="shipped")
    ap.add_argument("--theirs-arm", default="theirs", dest="theirs_arm")
    ap.add_argument("--per-step", type=int, default=4, dest="per_step")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--warmup", type=int, default=SCHED["warmup_no_steps"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out-dir", default=RUNS, dest="out_dir")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--w0", default="ckpt", choices=["ckpt", "own"],
                    help="baseline for d_k = w_k - w_0. `ckpt` is the inherited form, both "
                         "sides against the float64 checkpoint. `own` gives each side its "
                         "own w_0, which is what d_k means once our weights are not stored "
                         "in the checkpoint's dtype.")
    ap.add_argument("--aa-in-process", action="store_true", dest="aa_in_process",
                    help="reference side twice in ONE process, compared bit-exactly at every "
                         "rung. The A/A that says whether any magnitude below is readable")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(a.threads))
    refpath.install()

    import torch
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)

    D = torch.load(DIFFCAP, map_location="cpu", weights_only=False)
    kw, cot = D["kwargs"], D["cot"]
    n_sample = int(kw["xl_noisy"].shape[1])
    blocks = partition(n_sample, a.per_step, a.steps)
    print(f"[{time.time()-t0:.0f}s] boundary {DIFFCAP}: {n_sample} noise levels, "
          f"{a.per_step} accumulation samples/step, blocks of {[len(b) for b in blocks[1]]}",
          flush=True)

    log = []

    # The steplog is written after EVERY rung, not once at the end. A reset at k=14 used to
    # leave fourteen w_k dumps on disk and no record that they were taken, which is how the
    # 2026-09-21 reboot turned a nearly finished arm into nothing.
    def write_steplog(evidence=None):
        if not a.side:
            return
        sl = os.path.join(a.out_dir, f"steplog_{a.arm}.json")
        tmp = sl + ".part"
        json.dump({"arm": a.arm, "side": a.side, "ref_tree": REF_TREE,
                   "steps": log, "evidence": evidence, "n_steps": len(log),
                   "complete": evidence is not None,
                   "wall_s": time.time() - t0}, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, sl)
        return sl

    global STEPLOG_SINK
    STEPLOG_SINK = write_steplog

    if a.side == "theirs":
        d_out = wdir(a.out_dir, a.arm)
        kwv = {k: (v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in kw.items()}
        run_theirs(torch.float64, blocks, cot, kwv, steps=a.steps, warmup=a.warmup,
                   log=log, d_out=d_out, also_second=a.aa_in_process)
        side_ev = run_theirs.last
    elif a.side == "ours":
        import ttnn
        act = ttnn.float32
        G = build_ours(torch.load(CAP, map_location="cpu", weights_only=False), kw, act,
                       a.arm)
        G["out_ref"] = D["out"]
        d_out = wdir(a.out_dir, a.arm)
        run_ours(G, blocks, cot, steps=a.steps, warmup=a.warmup, log=log, d_out=d_out,
                 brk=(a.arm if a.arm in ("norebind", "permute", "stale") else "none"),
                 zero_grad_model=(a.arm == "zero"))
        side_ev = run_ours.last
    else:
        side_ev = None

    if a.side:
        STEPLOG_SINK = None
        sl = write_steplog(side_ev)
        print(f"wrote {sl} ({len(log)} steps, {time.time()-t0:.0f}s)", flush=True)

    if not a.score:
        return 0

    scored, names, W0, nw0, ks = do_score(a, a.out_dir)
    ev = {}
    tp = sys.modules.get("tt_bio.taped_ttnn")
    if tp is not None:
        ev["_SOFTMAX_BW_RENORM"] = bool(tp._SOFTMAX_BW_RENORM)
    tt = sys.modules.get("tt_bio.tenstorrent")
    if tt is not None:
        ev["HOST_F64_SOFTMAX_STATS"] = dict(tt.HOST_F64_SOFTMAX_STATS)

    def steplog(arm):
        p = os.path.join(a.out_dir, f"steplog_{arm}.json")
        return json.load(open(p)) if os.path.exists(p) else None

    their_tree = REF_TREE or (steplog(a.theirs_arm) or {}).get("ref_tree")
    if os.path.realpath(their_tree or "") != os.path.realpath(OF3PKG):
        raise SystemExit(
            f"reference arm {a.theirs_arm!r} was taken against {their_tree!r}, not the "
            f"tree under test {OF3PKG!r}. 0.4.3 and 0.5.0 are a different FUNCTION at "
            f"this boundary (D120: 1.94959719e-05 against 7.66979728e-01, f32/f32), so "
            f"this pairing is not scorable. Re-run the reference side. See D149.")

    res = {
        "arm": a.arm,
        "w0_baseline": a.w0,
        "ref_tree": their_tree,
        "scope": scope_share(names),
        "steps_scored": ks, "steps_asked": a.steps,
        "accumulate_grad_batches": a.per_step, "warmup_no_steps": a.warmup,
        "noise_level_partition_step1": blocks[1],
        "shipped_defaults": {k: v for k, v in shipped_defaults().items()
                             if k in ("betas", "weight_decay", "plateau_until", "lr",
                                      "warmup_steps")},
        "d1": {"ours_norm": scored[0]["d_ours_norm"], "theirs_norm": scored[0]["d_theirs_norm"],
               "rel_d": scored[0]["rel_d"],
               "zero_both_sides": scored[0]["d_ours_norm"] == 0.0 == scored[0]["d_theirs_norm"]},
        "growth_k2_20": growth(scored),
        "per_step": scored,
        "our_step_log": (steplog(a.arm) or {}).get("steps"),
        "their_step_log": (steplog(a.theirs_arm) or {}).get("steps"),
        "our_evidence": (steplog(a.arm) or {}).get("evidence"),
        "their_evidence": (steplog(a.theirs_arm) or {}).get("evidence"),
        "flag_reach": ev,
        "env": {k: os.environ.get(k) for k in
                ("TT_BIO_SOFTMAX_BW_RENORM", "TT_BIO_HOST_F64_SOFTMAX_AB",
                 "TT_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
        "timing_s": {"total": time.time() - t0},
        "sha256": {"diffusion_boundary": sha256(DIFFCAP), "cond_boundary": sha256(CAP),
                   "checkpoint": sha256(CKPT),
                   "grad_manager.py": sha256("/home/ttuser/of3t_traj20/upstream/grad_manager.py"),
                   "lr_schedulers.py":
                       sha256("/home/ttuser/of3t_traj20/upstream/lr_schedulers.py")},
    }
    out = a.out or (f"perf/of3t_trajwide/traj_{a.arm}.json" if a.w0 == "ckpt"
                    else f"perf/of3t_trajwide/traj_{a.arm}_w0own.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_step", "our_step_log", "their_step_log")},
                     indent=1, default=str)[:4000])
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
