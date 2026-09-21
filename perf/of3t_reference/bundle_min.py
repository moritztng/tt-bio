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
import os
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
    """Records the stochastic draws a training step makes, and optionally replays recorded ones.

    Replay is not a convenience. Restoring an RNG state reproduces a STATE, not a VALUE: the same
    state on another device, or on the same device after any other consumer of that stream is
    added or removed, yields different numbers. Both happen here. Putting the Pairformer's
    Dropout in eval removes 61 consumers of the CUDA generator, so the very next `torch.randn`
    in the diffusion head returns something else -- measured, the noise levels go from
    [23.705, 5.853, 10.730, ...] to [2.612, 1.467, 5.936, ...] with nothing else changed. PROTOCOL
    4a makes those draws inputs to the update rule, so two stacks that did not consume the same
    ones are evaluating different functions and their gradients are not comparable.

    Pass `replay` (a draws dict) to consume recorded values in order instead of sampling. Entering
    the context resets the cursor, so every finite-difference forward replays the same sequence
    from the start.
    """

    def __init__(self, replay=None):
        self.randn = []
        self.random = []
        self.replay = replay
        self.i_randn = self.i_random = 0
        self.mismatch = []
        self._torch_randn = None
        self._random_random = None

    def __enter__(self):
        self._torch_randn = torch.randn
        self._random_random = random.random
        self.randn, self.random = [], []
        self.i_randn = self.i_random = 0
        self.mismatch = []

        def randn(*args, **kwargs):
            # Always draw, even when replaying, so the generator advances by the same amount it
            # would have: anything downstream that reads the same stream stays in step.
            out = self._torch_randn(*args, **kwargs)
            if self.replay is not None:
                rec = self.replay["torch_randn"]
                if self.i_randn >= len(rec):
                    self.mismatch.append(f"randn call {self.i_randn}: no recorded draw left")
                elif tuple(rec[self.i_randn].shape) != tuple(out.shape):
                    self.mismatch.append(
                        f"randn call {self.i_randn}: recorded "
                        f"{tuple(rec[self.i_randn].shape)} != asked {tuple(out.shape)}")
                else:
                    out = rec[self.i_randn].to(device=out.device, dtype=out.dtype)
                self.i_randn += 1
            self.randn.append(out.detach().cpu().clone())
            return out

        def rnd():
            if self.replay is not None:
                rec = self.replay["python_random"]
                if self.i_random < len(rec):
                    v = rec[self.i_random]
                    self.i_random += 1
                    self._random_random()   # keep the stream advancing in step
                    self.random.append(v)
                    return v
                self.mismatch.append(f"random.random call {self.i_random}: none left")
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


def pin_deterministic_kernels(enable: bool) -> dict:
    """Make the gradient a function of its inputs rather than of the reduction order.

    Without this, two fresh processes on the same box with the same pinned RNG state produce
    gradients that agree to a median 2.6e-14 and disagree by up to 1.98 relative L2 on 55 of
    4,147 tensors -- measured, `reproduction_A13.json` from the first r = 0 rebuild. The 55 are
    all `layer_norm_z.bias`, whose gradient is a sum over tokens that very nearly cancels, so a
    last-bit change in the order cuBLAS and the backward's atomics accumulate in is amplified
    to O(1) *relative* while staying at 1e-16 absolute. PROTOCOL A13 asks whether the artifact
    reproduces, and a reference that is only reproducible to 1.98 on some tensors is not one.

    CUBLAS_WORKSPACE_CONFIG has to be in the environment before cuBLAS initialises, so it is set
    here, before the first CUDA call, and also by rebuild_r0.sh for the case where something
    touches CUDA earlier.
    """
    if not enable:
        return {"enabled": False}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "enabled": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "warn_only": True,
        "why": ("two nondeterministic runs disagreed by 1.98 worst relative L2 on 55 of 4,147 "
                "tensors, all layer-norm biases whose gradient sum cancels"),
    }


def build(dtype, seed, device, num_recycles=None):
    import pytorch_lightning as pl
    from openfold3.core.loss.loss_module import OpenFold3Loss
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
    cfg.architecture.shared.use_confidence_emb_prob = 0.8
    cfg.architecture.shared.diffusion.use_conditioning_prob = 0.8
    if num_recycles is not None:
        # Their trunk draws the recycle count from U{0..num_recycles} and tapes only the FINAL
        # cycle (model.py:230). Pinning the config to 0 makes the drawn count 0, so the taped
        # function and the evaluated function are the same one -- which is the only regime in
        # which a finite difference can validate a trunk gradient at all. It is a change to the
        # amount of work, not to the update rule, and it is recorded in the manifest.
        cfg.architecture.shared.num_recycles = num_recycles
    pl.seed_everything(seed, workers=True)
    model = OpenFold3(cfg).to(device=device, dtype=dtype)
    model.train()
    dropout = disable_dropout(model)
    loss_fn = OpenFold3Loss(config=cfg.architecture.loss_module).to(device)
    return cfg, model, loss_fn, dropout


def disable_dropout(model):
    """Pin every Dropout to r = 0, which is what makes this artifact a reference at all.

    Taped in train mode, the published gradient is ONE DRAW. The Pairformer block carries a
    Dropout at rate 0.25 whose mask comes from generators an RNG-state snapshot does not
    reproduce across devices, and `of3t-gradients` measured the cost on the identical captured
    boundary: two seeds with dropout live read worst 1.167 / median 0.550 with 47 of 57 tensors
    over PROTOCOL 3d's 5.0e-02 bar, and a second pair read worst 2.419 / median 0.481. The same
    pair with dropout off read 0.000e+00 worst AND median over 57 of 57, bit-identical. An
    artifact whose own draw-to-draw floor is an order of magnitude above the bar it is used at
    is a measurement, not a reference.

    So this is an **r = 0 reference**, and it is the function our side already computes:
    `tt_bio/train/lora.py:46` records that the tape carries no dropout op. Dropout is a
    regulariser, not part of the update rule, and PROTOCOL 4a's requirement is that both stacks
    draw the same randomness -- which for a mask neither stack can reproduce means drawing none.

    Two mechanisms, because one of them has to survive a later `.train()`. Eval mode makes their
    `Dropout.forward` return its input unchanged (`dropout.py:56`), and rate 0 makes `nn.Dropout`
    identity as well: verified on this torch that p = 0 in train mode returns the input bit for
    bit and does not advance the RNG.

    Only the Dropout submodules move. `OpenFold3.training` stays True, because it selects the
    memory settings, the recycle-count draw and the confidence path (`model.py:150, 650, 707`),
    and flipping the whole model would change the function in ways that have nothing to do with
    the mask.
    """
    from openfold3.core.model.primitives.dropout import Dropout

    rates, n_inner = [], 0
    for m in model.modules():
        if isinstance(m, Dropout):
            rates.append(float(m.r))
            m.eval()
            m.r = 0.0
        if isinstance(m, torch.nn.Dropout):
            n_inner += 1
            m.eval()
            m.p = 0.0
    assert all(not m.training and m.p == 0.0
               for m in model.modules() if isinstance(m, torch.nn.Dropout))
    return {
        "disabled": True,
        "n_of3_dropout_modules": len(rates),
        "n_nn_dropout_modules": n_inner,
        "rates_before": sorted(set(rates)),
        "why": "r = 0 reference: the mask is not reproducible across devices, so the published "
               "gradient would be one draw. See disable_dropout() in bundle_min.py.",
    }


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
    ap.add_argument("--num-recycles", type=int,
                    help="pin shared.num_recycles. Set 0 to make the drawn count 0, which is the "
                         "only regime where a finite difference can validate a trunk gradient: "
                         "with more cycles the FD measures the total derivative through all of "
                         "them while the analytic gradient is the partial derivative through the "
                         "final one.")
    ap.add_argument("--replay-draws", type=Path,
                    help="a draws.pt to consume instead of sampling. PROTOCOL 4a: the draws are "
                         "inputs to the update rule, and restoring an RNG state reproduces a "
                         "state rather than a value, so a run that must be comparable with an "
                         "earlier one has to be handed the earlier one's draws.")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="do NOT pin deterministic kernels. Off by default, and leaving it off "
                         "is what makes this artifact reproducible: cuBLAS split-k and the "
                         "scatter/index_add atomics in the backward reorder their reductions "
                         "between runs, which is a 1e-16 perturbation on most tensors and an "
                         "O(1) relative one on the LayerNorm biases whose gradient is a sum "
                         "that cancels.")
    ap.add_argument("--checkpoint", type=Path,
                    help="trained weights to load before taking the gradient. Without this the "
                         "gradient is taken at a random initialisation, where 2,271 of 4,890 "
                         "tensors are exactly zero by design and the step-1 gradient reaches only "
                         "the zero-initialised output projections -- see the doc.")
    args = ap.parse_args()

    deterministic = pin_deterministic_kernels(not args.nondeterministic)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    got = sha256_file(args.batch)
    if args.batch_sha256 and got != args.batch_sha256:
        raise SystemExit(f"batch sha256 {got} != expected {args.batch_sha256}")

    raw_batch = torch.load(args.batch, weights_only=False)
    cfg, model, loss_fn, dropout = build(dtype, args.seed, device, args.num_recycles)

    ckpt_info = None
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
              for k, v in sd.items()}
        incompatible = model.load_state_dict(sd, strict=False)
        ckpt_info = {
            "file": args.checkpoint.name,
            "sha256": sha256_file(args.checkpoint),
            "n_loaded": len(sd),
            "n_missing": len(incompatible.missing_keys),
            "n_unexpected": len(incompatible.unexpected_keys),
            "missing_sample": list(incompatible.missing_keys)[:8],
            "unexpected_sample": list(incompatible.unexpected_keys)[:8],
        }
        print(f"checkpoint {args.checkpoint.name}: loaded {len(sd)} tensors, "
              f"{ckpt_info['n_missing']} missing, {ckpt_info['n_unexpected']} unexpected",
              flush=True)

    # w_0, exactly the weights the gradient below is taken at.
    w0 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(w0, args.out / "w0.pt")

    batch = move(raw_batch, device, dtype)

    # Pin the draws. Every forward in this script restores this state first, so the finite
    # difference compares two evaluations of the SAME function rather than two samples of a
    # stochastic one.
    pinned = rng_state(model)

    replay = replay_info = None
    if args.replay_draws:
        replay = torch.load(args.replay_draws, map_location="cpu", weights_only=False)
        replay_info = {
            "file": args.replay_draws.name,
            "sha256": sha256_file(args.replay_draws),
            "n_torch_randn_recorded": len(replay["torch_randn"]),
            "n_python_random_recorded": len(replay["python_random"]),
            "recorded_num_recycles": int(replay["num_recycles"]),
        }

    t0 = time.time()
    set_rng_state(pinned, model)
    rec = DrawRecorder(replay)
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

    # Stratified by TENSOR, one entry each, which is the sampling this row owed on re-issue.
    # Drawing entries uniformly from the whole parameter vector concentrates the sample in the
    # largest tensors: on the previous draw ten of twelve entries landed in the pairformer stack
    # and the diffusion decoder was represented once. Stratifying costs nothing and makes the
    # coverage statement -- how many DISTINCT tensors the check touched -- a real one. A tensor
    # with no entry above --fd-min-grad anywhere is recorded as uncovered, not silently skipped.
    order = list(range(len(flat)))
    rs.shuffle(order)
    checks, uncovered = [], []
    for ti in order:
        if len(checks) >= args.fd_samples:
            break
        name, p = flat[ti]
        # Exact, not probed: on this batch most individual entries are exactly zero, so drawing
        # blind indices and retrying finds a valid entry by luck and reports "uncovered" for
        # tensors that are perfectly well covered. Enumerate the entries above the floor and
        # draw from those, and record how many there were.
        above = (p.grad.abs() >= args.fd_min_grad).flatten().nonzero(as_tuple=True)[0]
        n_above = int(above.numel())
        if n_above == 0:
            uncovered.append({"param": name, "numel": int(p.numel())})
            continue
        idx = np.unravel_index(int(above[rs.randrange(n_above)]), tuple(p.shape))
        idx = tuple(int(i) for i in idx)
        analytic = float(p.grad[idx])
        del above
        original = p.data[idx].item()
        with torch.no_grad():
            p.data[idx] = original + args.fd_h
        set_rng_state(pinned, model)
        lp = float(forward_loss(model, loss_fn, batch, rec, disable_autocast=no_ac)[0])
        with torch.no_grad():
            p.data[idx] = original - args.fd_h
        set_rng_state(pinned, model)
        lm = float(forward_loss(model, loss_fn, batch, rec, disable_autocast=no_ac)[0])
        with torch.no_grad():
            p.data[idx] = original
        fd = (lp - lm) / (2 * args.fd_h)
        denom = max(abs(analytic), abs(fd), 1e-30)
        checks.append({
            "param": name, "index": list(idx), "analytic": analytic, "finite_difference": fd,
            "abs_err": abs(analytic - fd), "rel_err": abs(analytic - fd) / denom,
            "n_entries_above_floor": n_above, "numel": int(p.numel()),
        })
        print(f"fd {name}{list(idx)}: analytic={analytic:.6e} fd={fd:.6e} "
              f"rel={checks[-1]['rel_err']:.3e}", flush=True)

    # A determinism control for the check itself: the same pinned state must reproduce the loss
    # bit for bit, or every finite difference above is noise rather than a derivative.
    set_rng_state(pinned, model)
    loss_again = float(forward_loss(model, loss_fn, batch, rec, disable_autocast=no_ac)[0])

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
        "num_recycles_pinned": args.num_recycles,
        "checkpoint": ckpt_info,
        "dtype": args.dtype,
        "their_fp32_autocast_blocks_disabled": bool(dtype is torch.float64),
        "dropout": dropout,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {
            "openfold3": getattr(openfold3, "__version__", "0.5.0 (git checkout)"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "deterministic_kernels": deterministic,
        "replayed_draws": (None if replay_info is None else
                           {**replay_info, "n_randn_consumed": rec.i_randn,
                            "n_python_random_consumed": rec.i_random,
                            "n_mismatch": len(rec.mismatch), "mismatch": rec.mismatch[:8]}),
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
            "n_nonzero": sum(
                1 for g in grads.values()
                if g is not None and float(torch.linalg.vector_norm(g)) > 0
            ),
            "n_zero_init_weight_tensors": sum(
                1 for v in w0.values() if v.numel() and float(v.abs().max()) == 0.0
            ),
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
            "sampling": "stratified by tensor, one entry per tensor",
            "n_samples": len(checks),
            "n_tensors_covered": len({c["param"] for c in checks}),
            "n_tensors_total": len(flat),
            "n_tensors_uncovered": len(uncovered),
            "uncovered_sample": uncovered[:8],
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
