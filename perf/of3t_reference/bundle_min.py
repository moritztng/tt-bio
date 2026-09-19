#!/usr/bin/env python3
"""BUNDLE-MIN: one frozen batch, its float64 gradients, its random draws, validated by finite
differences.

This is the artifact instrument A is performed against, so it is built to be attackable:

  * The gradient is upstream's own model and loss, differentiated by torch. Nothing here
    reimplements the step.
  * It is computed in **float64** (PROTOCOL 3c), not cast to float64 after an fp32 run.
  * It is validated against **float64 central finite differences** on the forward it claims to
    differentiate, on a sample of individual parameter entries. A stochastic forward makes that
    check meaningless unless both evaluations draw the same randomness, so every forward in the
    check replays a pinned RNG state.
  * The **gradient-presence pattern** is stored per parameter, with None kept as None. Filling an
    absent gradient with zeros is the defect in `zero-filled-missing-gradient-hides-an-untrained-
    model` and PROTOCOL 3b makes presence a compared property.
  * The **random draws** are stored (PROTOCOL 4a): the trunk recycle count is drawn per step from
    U{0..3}, the diffusion head noises 48 structures, and `use_conditioning` is a coin flip. A
    comparison whose two stacks drew differently is comparing different work and is void.

The optimizer step is deliberately NOT part of this bundle. Clipping, the LR schedule, Adam and
the EMA are instruments B and C, verified exactly without a model; mixing them in here would only
blur what a gradient mismatch means. The clip coefficient their grad_manager WOULD apply is
recorded so the optimizer-facing gradient is derivable: at world size 1 with
accumulate_grad_batches 1 it is exactly `grad * clip_coef`.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class DrawRecorder:
    """Records the stochastic draws a training step makes, by observing and calling through."""

    def __init__(self):
        self.randn = []
        self.random = []
        self._torch_randn = None
        self._random_random = None

    def __enter__(self):
        self._torch_randn = torch.randn
        self._random_random = random.random

        def randn(*args, **kwargs):
            out = self._torch_randn(*args, **kwargs)
            self.randn.append(out.detach().cpu().clone())
            return out

        def rnd():
            v = self._random_random()
            self.random.append(v)
            return v

        torch.randn = randn
        random.random = rnd
        return self

    def __exit__(self, *exc):
        torch.randn = self._torch_randn
        random.random = self._random_random
        return False


def rng_state(model=None):
    """Every stream a training step draws from, including the one nobody thinks of.

    `torch.manual_seed`, `np.random.set_state` and `random.setstate` do NOT reach
    `OpenFold3.synced_generator`, a private `np.random.default_rng` the model keeps as an
    attribute (`model.py:77`). It is the stream the trunk RECYCLE COUNT is drawn from
    (`model.py:650`, U{0..3}), and it advances on every forward. Restore the other three and a
    "replayed" forward still silently runs a different number of trunk passes.

    That is not hypothetical: with this generator left out, finite differences on the same
    parameter came back either perfect (2.2e-08 relative) or exactly 1.0, with the bad ones
    sharing a difference quotient of +-5.3286e+03 -- one fixed jump in the loss, which is what a
    discrete change in the amount of work looks like. PROTOCOL 4a says the draws are inputs to the
    update rule; this is what it costs to forget one.
    """
    s = {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if model is not None:
        s["of3_synced_generator"] = model.synced_generator.bit_generator.state
    return s


def set_rng_state(s, model=None):
    torch.set_rng_state(s["torch_cpu"])
    if s["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(s["torch_cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if model is not None and "of3_synced_generator" in s:
        model.synced_generator.bit_generator.state = s["of3_synced_generator"]


class no_autocast:
    """Run their model as a genuine float64 reference.

    Upstream OF3 is not float64-clean, and it is worth saying exactly where, because it is the
    only reason this class exists:

      1. Their modules wrap work in `torch.amp.autocast(device_type=..., dtype=torch.float32)`.
         On CPU torch disables those blocks itself -- their own test suite asserts the
         "Disabling autocast" warning -- but on CUDA they downcast a float64 graph.
      2. `projects/of3_all_atom/model.py:325`, `run_trunk` ends with
         `return s_input.float(), s.float(), z.float()`, an UNCONDITIONAL cast. So the trunk hands
         float32 to the diffusion conditioning no matter what dtype the model is in, and the first
         LayerNorm downstream raises `expected scalar type Float but found Double`.

    Both are precision FLOORS for their bf16-mixed and fp32 paths, and both become downcasts
    inside a float64 model. This context removes them for the duration of one reference forward:
    autocast becomes a no-op, and `Tensor.float()` becomes identity **on tensors that are already
    float64** and is untouched for every other dtype, so an integer tensor still promotes to
    float32 exactly as before.

    Every change here only ever REMOVES a downcast. The reference is therefore computed at least
    as precisely as their own path, never less, and the finite-difference check below is run
    against this same forward rather than against the unpatched one.
    """

    def __enter__(self):
        self._orig_autocast = torch.amp.autocast
        self._orig_float = torch.Tensor.float
        orig_float = self._orig_float

        def keep_double(self_t, *a, **k):
            return self_t if self_t.dtype is torch.float64 else orig_float(self_t, *a, **k)

        torch.amp.autocast = lambda *a, **k: _null()
        torch.Tensor.float = keep_double
        return self

    def __exit__(self, *exc):
        torch.amp.autocast = self._orig_autocast
        torch.Tensor.float = self._orig_float
        return False


def build(dtype, seed, device):
    import pytorch_lightning as pl
    from openfold3.core.loss.loss_module import OpenFold3Loss
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
    cfg.architecture.shared.use_confidence_emb_prob = 0.8
    cfg.architecture.shared.diffusion.use_conditioning_prob = 0.8
    pl.seed_everything(seed, workers=True)
    model = OpenFold3(cfg).to(device=device, dtype=dtype)
    model.train()
    loss_fn = OpenFold3Loss(config=cfg.architecture.loss_module).to(device)
    return cfg, model, loss_fn


def move(batch, device, dtype):
    def conv(t):
        if not torch.is_tensor(t):
            return t
        t = t.to(device)
        return t.to(dtype) if t.is_floating_point() else t

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return conv(x)

    return walk(batch)


def forward_loss(model, loss_fn, batch, recorder=None, disable_autocast=False):
    """One forward + loss on a PRIVATE copy of the batch.

    Their `forward` mutates the batch it is handed: it pops `ref_space_uid_to_perm`, unsqueezes a
    sampling dimension into every tensor, and the permutation alignment rewrites the ground-truth
    coordinates in place. Re-running on the same dict therefore evaluates a DIFFERENT function,
    which silently destroys any finite-difference check -- the second evaluation is not the same
    forward, so the difference quotient measures the mutation, not a derivative. Measured before
    this copy was added: two forwards from an identical pinned RNG state disagreed in the loss at
    the 1e-1 scale, which at h = 1e-5 produced difference quotients of order 1e4 against analytic
    gradients of order 1e0.
    """
    ctx = recorder if recorder is not None else _null()
    ac = no_autocast() if disable_autocast else _null()
    private = copy.deepcopy(batch)
    with ac, ctx:
        b, out = model(private)
        loss, breakdown = loss_fn(b, out, _return_breakdown=True)
    return loss, breakdown, out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--batch-sha256", help="expected sha256 of --batch, checked before use")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--fd-samples", type=int, default=16,
                    help="parameter entries validated by central finite differences")
    ap.add_argument("--fd-h", type=float, default=1e-5)
    ap.add_argument("--fd-min-grad", type=float, default=1e-6,
                    help="only validate entries whose analytic gradient is at least this large")
    ap.add_argument("--clip-val", type=float, default=10.0)
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    got = sha256_file(args.batch)
    if args.batch_sha256 and got != args.batch_sha256:
        raise SystemExit(f"batch sha256 {got} != expected {args.batch_sha256}")

    raw_batch = torch.load(args.batch, weights_only=False)
    cfg, model, loss_fn = build(dtype, args.seed, device)

    # w_0, exactly the weights the gradient below is taken at.
    w0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(w0, args.out / "w0.pt")

    batch = move(raw_batch, device, dtype)

    # Pin the draws. Every forward in this script restores this state first, so the finite
    # difference compares two evaluations of the SAME function rather than two samples of a
    # stochastic one.
    pinned = rng_state(model)

    t0 = time.time()
    set_rng_state(pinned, model)
    rec = DrawRecorder()
    no_ac = dtype is torch.float64
    loss, breakdown, out = forward_loss(model, loss_fn, batch, rec, disable_autocast=no_ac)
    t_fwd = time.time() - t0

    t0 = time.time()
    loss.backward()
    t_bwd = time.time() - t0

    grads, presence = {}, {}
    for name, p in model.named_parameters():
        presence[name] = p.grad is not None
        grads[name] = p.grad.detach().to(torch.float64).cpu().clone() if p.grad is not None else None
    torch.save(grads, args.out / "grads_f64.pt")
    (args.out / "grad_presence.json").write_text(
        json.dumps(presence, indent=0, sort_keys=True) + "\n"
    )

    # The clip coefficient their PerSampleGradManager would apply to this per-sample gradient.
    # Their expression, evaluated on their global norm: max_norm / max(global_norm, max_norm).
    per_tensor = [torch.linalg.vector_norm(g.double()) for g in grads.values() if g is not None]
    global_norm = float(torch.linalg.vector_norm(torch.stack(per_tensor)))
    clip_coef = args.clip_val / max(global_norm, args.clip_val)

    draws = {
        "num_recycles": int(out["recycles"]),
        "noise_level": out["noise_level"].detach().cpu().clone(),
        "python_random": rec.random,
        "torch_randn": rec.randn,
        "rng_state_before_step": pinned,
    }
    torch.save(draws, args.out / "draws.pt")

    # Central finite differences on the forward this gradient claims to differentiate.
    # PROTOCOL 3c: never validate a reference against another approximation.
    # Sampling rule, fixed before the numbers exist: an entry whose analytic gradient is exactly
    # zero cannot validate anything -- the relative error is 0/0 or 1 by construction -- and on a
    # padded 56-token batch a large share of entries ARE exactly zero, because masked channels
    # never contribute. So entries are drawn uniformly from those whose |analytic| is at least
    # --fd-min-grad, and the share of exactly-zero entries is reported rather than hidden.
    flat = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    rs = random.Random(args.seed)
    n_probe, n_zero = 4000, 0
    for _ in range(n_probe):
        _, p = flat[rs.randrange(len(flat))]
        if float(p.grad[tuple(rs.randrange(s) for s in p.shape)]) == 0.0:
            n_zero += 1
    zero_entry_fraction = n_zero / n_probe

    checks = []
    tries = 0
    while len(checks) < args.fd_samples and tries < args.fd_samples * 4000:
        tries += 1
        name, p = flat[rs.randrange(len(flat))]
        idx = tuple(rs.randrange(s) for s in p.shape)
        analytic = float(p.grad[idx])
        if abs(analytic) < args.fd_min_grad:
            continue
        original = p.data[idx].item()
        with torch.no_grad():
            p.data[idx] = original + args.fd_h
        set_rng_state(pinned, model)
        lp = float(forward_loss(model, loss_fn, batch, disable_autocast=no_ac)[0])
        with torch.no_grad():
            p.data[idx] = original - args.fd_h
        set_rng_state(pinned, model)
        lm = float(forward_loss(model, loss_fn, batch, disable_autocast=no_ac)[0])
        with torch.no_grad():
            p.data[idx] = original
        fd = (lp - lm) / (2 * args.fd_h)
        denom = max(abs(analytic), abs(fd), 1e-30)
        checks.append({
            "param": name, "index": list(idx), "analytic": analytic, "finite_difference": fd,
            "abs_err": abs(analytic - fd), "rel_err": abs(analytic - fd) / denom,
        })
        print(f"fd {name}{list(idx)}: analytic={analytic:.6e} fd={fd:.6e} "
              f"rel={checks[-1]['rel_err']:.3e}", flush=True)

    # A determinism control for the check itself: the same pinned state must reproduce the loss
    # bit for bit, or every finite difference above is noise rather than a derivative.
    set_rng_state(pinned, model)
    loss_again = float(forward_loss(model, loss_fn, batch, disable_autocast=no_ac)[0])

    import openfold3

    manifest = {
        "bundle": "BUNDLE-MIN",
        "batch": {
            "file": args.batch.name,
            "sha256": got,
            "pdb_id": raw_batch["pdb_id"],
            "n_tokens": int(raw_batch["token_mask"].sum()),
        },
        "seed": args.seed,
        "dtype": args.dtype,
        "their_fp32_autocast_blocks_disabled": bool(dtype is torch.float64),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {
            "openfold3": getattr(openfold3, "__version__", "0.5.0 (git checkout)"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "loss": float(loss),
        "loss_replayed_same_rng": loss_again,
        "loss_bit_identical_on_replay": loss_again == float(loss),
        "loss_terms": {k: float(v) for k, v in breakdown.items()},
        "draws": {
            "num_recycles": draws["num_recycles"],
            "noise_level": draws["noise_level"].reshape(-1).tolist()[:64],
            "n_torch_randn_calls": len(rec.randn),
            "torch_randn_shapes": [list(t.shape) for t in rec.randn],
            "python_random_values": rec.random,
        },
        "gradient": {
            "n_params": len(presence),
            "n_with_gradient": sum(presence.values()),
            "n_absent": sum(1 for v in presence.values() if not v),
            "global_norm": global_norm,
            "clip_coef_their_grad_manager_would_apply": clip_coef,
            "note": ("grads_f64.pt holds the per-sample gradient BEFORE clipping. At world size 1 "
                     "with accumulate_grad_batches 1 the optimizer-facing gradient is exactly "
                     "grad * clip_coef_their_grad_manager_would_apply."),
        },
        "finite_difference_validation": {
            "h": args.fd_h,
            "min_abs_analytic_sampled": args.fd_min_grad,
            "zero_gradient_entry_fraction": zero_entry_fraction,
            "n_samples": len(checks),
            "max_rel_err": max(c["rel_err"] for c in checks) if checks else None,
            "median_rel_err": float(np.median([c["rel_err"] for c in checks])) if checks else None,
            "worst": max(checks, key=lambda c: c["rel_err"]) if checks else None,
            "checks": checks,
        },
        "timings_s": {"forward": t_fwd, "backward": t_bwd},
        "files": {},
    }
    for f in ("w0.pt", "grads_f64.pt", "draws.pt", "grad_presence.json"):
        p = args.out / f
        manifest["files"][f] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (args.out / "manifest.json").write_text(body)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("files", "finite_difference_validation")}, indent=2)[:2000])
    print("fd max rel err:", manifest["finite_difference_validation"]["max_rel_err"])
    print("manifest sha256:", hashlib.sha256(body.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
