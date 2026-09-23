#!/usr/bin/env python3
"""The full training step of `perf/of3t_trainfwd/trainfwd_run.py --arm full`, in upstream 0.4.3.

    ref_step.py --mode f64|bf16 --out-dir D [--replay-draws draws.pt] [--fd] [--denoise]

Our step is: trunk (one cycle) -> distogram head; the 20-step rollout, detached, one sample;
the confidence Pairformer on the rolled-out structure; the `af3` objective in float64 on host,
whose seeds go back through the graph. This file computes that function with UPSTREAM's modules,
built, loaded and cast by `bundle_min`'s own functions (imported, not copied), and differentiates
it with torch.

Where our adapter's loss definition departs from upstream's training forward, this follows OUR
step, because the question is which of our two arithmetic stacks is nearer the float64 value of
the step we run. Each departure is a finding about the adapter, recorded in the output under
`adapter_departures`, and none is fixed here:

  * representative atom = `start_atom_index` (each token's first atom), where upstream's
    `get_token_representative_atoms` takes CB (CA for glycine);
  * the confidence Pairformer is NOT detached from the trunk, where upstream detaches
    si_input / si / zij before it (head_modules.py);
  * `resolved_logits` reach the objective as [N_token, 23 * 2] and `losses.resolved` reads that
    as 46 classes, where upstream gathers [N_atom, 2] under `max_atom_per_token_mask`.

MSA subsampling is off (our step feeds every MSA row; this batch has 2, both valid) and dropout
is r = 0 (`bundle_min.disable_dropout`).

`--mode f64` samples the rollout draws under `torch.manual_seed(--seed)` and writes them before
the backward starts, so the device arms can begin replaying them. `--mode bf16` replays them and
is upstream's own bf16-autocast recipe (`bundle_min.cast_policy("bf16")`, float32 parameters).

`--denoise` (of3t-denoise) adds the adapter's one-step denoise arm, which is what trains the
diffusion module: the ground truth noised with the adapter's own draw
(`tt_bio.train.openfold3.denoise_draw`, the seed and the atom count, nothing else), upstream's
`diffusion_module` called the way upstream's training step calls it with one sample, the denoised
atoms gathered to the token axis at `start_atom_index`, and `pred_xyz` / `pred_dist` handed to the
same objective with the same EDM scale. `use_conditioning` is fixed True (upstream draws it with
probability `use_conditioning_prob`) and there is no `centre_random_augmentation` of the truth;
both are adapter departures and recorded as such.

`--fd` (f64 only) is the A40 gating control: the loss re-evaluated from a fresh forward at the
same point, and a central finite difference along u = g / ||g||, with the rollout's structure held
at the base forward's. The gradient is of the step with the detached rollout frozen, so that is
the function the check differentiates.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[0:0] = [str(REPO / "perf" / "of3t_reference"), str(HERE), str(REPO)]
import bundle_min as bm  # noqa: E402
from draws import Draws  # noqa: E402

SECTIONS = ("aux_heads.distogram.", "aux_heads.", "sample_diffusion.", "diffusion_module.", "")
HEAD = {"aux_heads.distogram.": "distogram", "aux_heads.": "confidence",
        "sample_diffusion.": "diffusion", "diffusion_module.": "diffusion", "": "trunk"}


def head_of(name: str) -> str:
    return next(HEAD[p] for p in SECTIONS if name.startswith(p))


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class RssGuard(threading.Thread):
    """pc is Moritz's box and has 30 GB. Past the cap this process exits rather than swap it."""

    def __init__(self, cap_gb: float):
        super().__init__(daemon=True)
        self.cap, self.peak = cap_gb, 0.0

    @staticmethod
    def rss_gb() -> float:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 2 ** 20
        return 0.0

    def run(self):
        n = 0
        while True:
            r = self.rss_gb()
            self.peak = max(self.peak, r)
            n += 1
            if n % 50 == 0:
                print(f"RSS {time.strftime('%H:%M:%S')} {r:.1f} GB (peak {self.peak:.1f})",
                      file=sys.stderr, flush=True)
            if r > self.cap:
                print(f"RSS {r:.1f} GB over the {self.cap} GB cap -- exiting", flush=True)
                os._exit(99)
            time.sleep(0.2)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _DiskChain(torch.autograd.Function):
    """Reentrant checkpointing with the block inputs on disk instead of in RAM.

    The float64 step at width 384 peaks past pc's 30 GB with upstream's own per-block
    checkpointing, because the 48 Pairformer block inputs (151 MB of pair each) stay resident
    while one block is recomputed. This is the same recompute-in-backward, same order, same
    arithmetic; only where the 48 inputs wait differs. Parameter gradients accumulate inside
    backward exactly as in torch's reentrant checkpoint.
    """

    @staticmethod
    def forward(ctx, blocks, tmpdir, s, z):
        ctx.blocks, ctx.files = blocks, []
        for i, b in enumerate(blocks):
            f = Path(tmpdir) / f"pf_in_{i:02d}.pt"
            torch.save((s, z), f)
            ctx.files.append(f)
            s, z = b(s, z)
        return s, z

    @staticmethod
    def backward(ctx, gs, gz):
        for b, f in zip(reversed(ctx.blocks), reversed(ctx.files)):
            gc.collect()
            s, z = torch.load(f, weights_only=False)
            f.unlink()
            s, z = s.detach().requires_grad_(), z.detach().requires_grad_()
            with torch.enable_grad():
                so, zo = b(s, z)
            torch.autograd.backward([so, zo], [gs if gs is not None else torch.zeros_like(so),
                                               gz if gz is not None else torch.zeros_like(zo)])
            gs, gz = s.grad, z.grad
        return None, None, gs, gz


class DiskSaved:
    """Saved-for-backward tensors of the forward parked on disk.

    Where the tensors of the graph wait, nothing else: the backward reads back the bytes it
    would have read from RAM. The test is the size of the STORAGE, not of the saved tensor: a
    small view kept in RAM pins its whole base (a 1 MB slice of a 151 MB pair tensor keeps
    151 MB alive), so a view of a large storage is parked as a compact copy of its own bytes.
    Tensors saved during the backward itself (recomputes) stay in RAM.
    """

    def __init__(self, tmpdir, min_bytes=8 << 20):
        self.dir, self.min, self.n, self.bytes = Path(tmpdir), min_bytes, 0, 0
        self.ram = {}

    def pack(self, t):
        st = t.untyped_storage()
        if isinstance(t, torch.nn.Parameter):   # alive anyway
            return t
        if st.nbytes() < self.min or torch._C._current_graph_task_id() != -1:
            self.ram[st.data_ptr()] = st.nbytes()
            return t
        own = t.numel() * t.element_size()
        f = self.dir / f"sv_{self.n:05d}.pt"
        self.n += 1
        self.bytes += own
        torch.save(t.detach().clone() if own < st.nbytes() else t.detach(), f)
        return (f, t.requires_grad)

    @staticmethod
    def unpack(x):
        if isinstance(x, tuple):
            return torch.load(x[0], weights_only=False)
        return x

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)


LEAVES = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end", "pair_transition",
          "attn_pair_bias", "single_transition", "outer_product_mean", "msa_att_row",
          "msa_transition")


def checkpoint_leaves(stack):
    """Recompute each Pairformer sub-layer in its own backward (non-reentrant checkpoint).

    Nested inside the block recompute: one sub-layer's float64 graph is alive at a time instead of
    a whole block's (two triangle attentions' softmax at 384^2 x 4 heads is 3.6 GB alone). Same
    arithmetic, recomputed; only the peak changes.
    """
    for m in stack.modules():
        for name in LEAVES:
            leaf = getattr(m, name, None)
            if isinstance(leaf, torch.nn.Module) and not getattr(leaf, "_ckpt_leaf", False):
                fwd = leaf.forward

                def wrapped(*args, _fwd=fwd, **kw):
                    if not torch.is_grad_enabled():
                        return _fwd(*args, **kw)
                    return torch.utils.checkpoint.checkpoint(_fwd, *args, use_reentrant=False,
                                                             **kw)
                leaf.forward, leaf._ckpt_leaf = wrapped, True


def disk_checkpoint(stack, tmpdir):
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    checkpoint_leaves(stack)

    def forward(s, z, single_mask, pair_mask, chunk_size=None, use_deepspeed_evo_attention=False,
                use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False,
                inplace_safe=False, _mask_trans=True):
        blocks = stack._prep_blocks(s, z, single_mask, pair_mask, chunk_size,
                                    use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                                    use_triton_triangle_kernels, use_lma, inplace_safe,
                                    _mask_trans)
        if not torch.is_grad_enabled():
            for b in blocks:
                s, z = b(s, z)
            return s, z
        return _DiskChain.apply(blocks, tmpdir, s, z)
    stack.forward = forward


def _trim():
    """Hand freed heap back to the OS so the RSS guard reads live memory, not glibc's high-water."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def load(dtype, checkpoint: Path, seed: int, chunk_size=None):
    cfg, model, _loss, dropout = bm.build(dtype, seed, "cpu", num_recycles=0)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    n_sd = len(sd)
    del ck, sd
    gc.collect()
    _trim()
    if inc.unexpected_keys:   # bundle_min's key gate (D23/R126)
        raise SystemExit(f"KEY GATE FAILED: {len(inc.unexpected_keys)} unexpected tensors")
    emb = model.msa_module_embedder
    emb.subsample_main_msa = emb.subsample_all_msa = False
    if chunk_size:
        # Chunked attention and transitions: an exact decomposition over a batch-like axis, so
        # the function is unchanged; it bounds the per-block transient in the float64 backward.
        # The tuner asserts eval mode, so it is removed rather than consulted.
        with model.settings.unlocked() if hasattr(model.settings, "unlocked") else _Null():
            model.settings.memory.train.chunk_size = chunk_size
        for m in model.modules():
            if getattr(m, "chunk_size_tuner", None) is not None:
                m.chunk_size_tuner = None
    return cfg, model, dropout, {"file": checkpoint.name, "sha256": sha256_file(checkpoint),
                                 "n_loaded": n_sd, "n_missing": len(inc.missing_keys),
                                 "n_unexpected": 0}


def labels(batch_path: Path):
    """The objective's labels, built by our own dataset class from the same file."""
    from tt_bio.train.openfold3 import OpenFold3Dataset
    return OpenFold3Dataset(batch_path).batch([0])


def safe_pdist(p):
    """Pairwise distance whose gradient is 0 where the distance is 0.

    `losses.dist_grad_to_coords`, the device arm's backward, divides by max(d, 1e-10) and so
    sends nothing through a zero distance (the diagonal, and pad tokens parked at the origin).
    A bare sqrt would send NaN there.
    """
    sq = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    nz = sq > 0
    return torch.where(nz, torch.sqrt(torch.where(nz, sq, torch.ones_like(sq))),
                       torch.zeros_like(sq))


def denoise_arm(model, batch, s_input, s, z, draw, chunk):
    """The adapter's one-step denoise, in upstream's modules. `draw` is `denoise_draw`'s."""
    sigma, eps = draw
    dt = s.dtype
    amask = batch["atom_mask"]                                           # [1, n_atom]
    xl_true = batch["ground_truth"]["atom_positions"][0] * amask[0, :, None]
    xl_noisy = (xl_true + sigma * torch.as_tensor(eps, dtype=dt)) * amask[0, :, None]
    # Upstream's forward gives every batch feature and representation a sample axis before
    # `_train_diffusion` sees them (model.py, "Expand sampling dimension"); so does this.
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    b1 = tensor_tree_map(lambda t: t.unsqueeze(1),
                         {k: v for k, v in batch.items() if k != "ref_space_uid_to_perm"})
    xl = model.diffusion_module(
        batch=b1, xl_noisy=xl_noisy[None, None], token_mask=b1["token_mask"],
        atom_mask=b1["atom_mask"], t=torch.full((1, 1), sigma, dtype=dt),
        si_input=s_input.unsqueeze(1), si_trunk=s.unsqueeze(1), zij_trunk=z.unsqueeze(1),
        use_conditioning=True, chunk_size=chunk,
        use_high_precision_attention=True, _mask_trans=True)[0, 0]          # [n_atom, 3]
    tok = batch["token_mask"][0]
    real = torch.nonzero(tok > 0, as_tuple=True)[0]
    rep = batch["start_atom_index"][0].long()[real]
    pred = xl.new_zeros(tok.shape[-1], 3).index_put((real,), xl[rep])
    return {"pred_xyz": pred, "pred_dist": safe_pdist(pred)}


def trunk_and_heads(model, batch, repr_x=None, cfg=None, seed=None, replay=None, draw=None):
    """One forward of the step. Returns outputs, the rolled-out atoms and the draw recorder.

    `repr_x` given: the rollout is skipped and that token-scope structure is scored (FD).
    `draw` given: the denoise arm runs on it (`denoise_arm`).
    """
    chunk = model.settings.memory.train.chunk_size
    from openfold3.core.model.structure.diffusion_module import create_noise_schedule
    from openfold3.core.utils.tensor_utils import tensor_tree_map

    s_input, s, z = model.run_trunk(batch=batch, num_cycles=1, inplace_safe=False)
    out = {"distogram_logits": model.aux_heads.distogram(z=z)}
    if draw is not None:
        out.update(denoise_arm(model, batch, s_input, s, z, draw, chunk))

    tok = batch["token_mask"]                                  # [1, N]
    xl, rec = None, None
    if repr_x is None:
        b1 = {k: v for k, v in batch.items() if k != "ref_space_uid_to_perm"}
        b1 = tensor_tree_map(lambda t: t.unsqueeze(1), b1)
        with torch.no_grad():
            ns = create_noise_schedule(no_rollout_steps=20,
                                       **cfg.architecture.noise_schedule,
                                       dtype=s_input.dtype, device=s_input.device)
            if replay is None:
                torch.manual_seed(seed)
            with Draws(replay) as rec:
                xl = model.sample_diffusion(
                    batch=b1, si_input=s_input.unsqueeze(1), si_trunk=s.unsqueeze(1),
                    zij_trunk=z.unsqueeze(1), noise_schedule=ns, no_rollout_samples=1,
                    use_conditioning=True, _mask_trans=True)[0, 0]      # [n_atom, 3]
        rep = batch["start_atom_index"][0].long()
        repr_x = torch.zeros(tok.shape[-1], 3, dtype=xl.dtype)
        real = torch.nonzero(tok[0] > 0, as_tuple=True)[0]
        repr_x[real] = xl.detach()[rep[real]]

    pe = model.aux_heads.pairformer_embedding
    pair = tok[..., None] * tok[..., None, :]
    si_c, zij_c = pe.pairformer_emb(si_input=s_input, si=s, zij=z,
                                    x_pred=repr_x.to(z.dtype)[None], single_mask=tok,
                                    pair_mask=pair, chunk_size=chunk, _mask_trans=True)
    er = model.aux_heads.experimentally_resolved
    out["resolved_logits"] = er.linear(er.layer_norm(si_c))  # [1, N, 23 * 2], our layout
    return out, xl, repr_x, rec


def objective(lab, out):
    """Our `af3` objective in float64 on host. A denoise run's EDM scale is already on `lab`."""
    from tt_bio.train import objectives
    from tt_bio.train.losses import of3_loss_weights
    host = {k: v.detach().to(torch.float64).numpy() for k, v in out.items()}
    w = of3_loss_weights("initial_training", "weighted-pdb")
    loss, breakdown, seeds = objectives.objective("af3")(lab, host, weights=w)
    return float(loss), breakdown, seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("f64", "bf16"))
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--batch-sha256", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260922, help="the rollout draw seed")
    ap.add_argument("--build-seed", type=int, default=20260919)
    ap.add_argument("--replay-draws", type=Path)
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--denoise", action="store_true",
                    help="add the adapter's one-step denoise arm (of3t-denoise)")
    ap.add_argument("--fd-h", default="1e-3,1e-4,1e-5")
    ap.add_argument("--fd-from", type=Path,
                    help="run --fd against a banked grads_f64.pt instead of a fresh backward")
    ap.add_argument("--expect-loss", type=float,
                    help="with --fd-from: the loss the banked gradient's run logged, matched exactly")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--rss-cap-gb", type=float, default=22.0)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--disk-checkpoint", type=Path,
                    help="park the trunk Pairformer's 48 block inputs here during the step")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    guard = RssGuard(a.rss_cap_gb)
    guard.start()
    t0 = time.time()

    got = sha256_file(a.batch)
    if got != a.batch_sha256:
        raise SystemExit(f"batch sha256 {got} != {a.batch_sha256}")
    det = bm.pin_deterministic_kernels(True)
    dtype = torch.float64 if a.mode == "f64" else torch.float32
    cfg, model, dropout, ck = load(dtype, a.checkpoint, a.build_seed, a.chunk_size)
    lab = labels(a.batch)
    if a.disk_checkpoint:
        a.disk_checkpoint.mkdir(parents=True, exist_ok=True)
        disk_checkpoint(model.pairformer_stack, a.disk_checkpoint / "trunk")
        disk_checkpoint(model.aux_heads.pairformer_embedding.pairformer_stack,
                        a.disk_checkpoint / "confidence")
        # The MSA module and template stack run the same pair sub-layers without a per-block
        # checkpoint; their whole-block float64 backward is what broke the cap after the trunk.
        checkpoint_leaves(model.msa_module)
        checkpoint_leaves(model.template_embedder)
    batch = bm.move(torch.load(a.batch, weights_only=False), "cpu", dtype)
    draw = None
    if a.denoise:
        from tt_bio.train import losses
        from tt_bio.train.openfold3 import denoise_draw
        gt = batch["ground_truth"]
        if not bool((gt["start_atom_index"][0].long() == batch["start_atom_index"][0][
                torch.nonzero(batch["token_mask"][0] > 0, as_tuple=True)[0]].long()).all()):
            raise SystemExit("ground_truth.start_atom_index != start_atom_index[real]: the "
                             "adapter and this gather would pick different atoms")
        n_atom = int(gt["atom_positions"].shape[-2])
        draw = denoise_draw(a.seed, n_atom)
        sigma_data = float(model.diffusion_module.sigma_data)
        lab["edm_scale"] = float(losses.edm_scale(draw[0], sigma_data))
    replay = None
    if a.replay_draws:
        replay = torch.load(a.replay_draws, weights_only=False)["torch_randn"]
    elif a.mode != "f64":
        raise SystemExit("only the float64 run samples draws; pass --replay-draws")

    import openfold3
    rec = {"row": "of3t-denoise" if a.denoise else "of3t-fullstep64", "mode": a.mode,
           "host": socket.gethostname(),
           "threads": a.threads,
           "malloc_env": {k: v for k, v in os.environ.items() if k.startswith("MALLOC_")}, "openfold3": openfold3.__file__, "torch": torch.__version__,
           "batch": {"file": a.batch.name, "sha256": got}, "checkpoint": ck,
           "dropout": dropout, "deterministic": det, "draw_seed": a.seed, "chunk_size": a.chunk_size,
           "disk_checkpoint": str(a.disk_checkpoint) if a.disk_checkpoint else None,
           "adapter_departures": [
               "representative atom = start_atom_index (first atom); upstream: CB / CA(gly)",
               "confidence Pairformer not detached from the trunk; upstream detaches its inputs",
               "resolved_logits [N_token, 46] read as 46 classes; upstream [N_atom, 2] gathered"]}
    if a.denoise:
        rec["denoise"] = {
            "draw": "tt_bio.train.openfold3.denoise_draw(seed, n_atom)", "seed": a.seed,
            "n_atom": n_atom, "sigma": draw[0], "edm_scale": lab["edm_scale"],
            "sigma_data": sigma_data,
            "noise_sha256_f32": hashlib.sha256(np.asarray(draw[1], np.float32).tobytes()).hexdigest(),
            "noise_sha256_f64": hashlib.sha256(np.asarray(draw[1], np.float64).tobytes()).hexdigest()}
        rec["adapter_departures"] += [
            "denoise: one sample, use_conditioning fixed True; upstream no_samples and a "
            "use_conditioning_prob draw",
            "denoise: no centre_random_augmentation of the ground truth before noising",
            "denoise: pred_xyz at token scope (start_atom_index) scored by our mse / smooth_lddt; "
            "upstream scores every atom"]

    policy = bm.cast_policy("removed" if a.mode == "f64" else "bf16", "cpu")
    t_f = time.time()
    saved = DiskSaved(a.disk_checkpoint) if a.disk_checkpoint else None
    with policy, (saved.hooks() if saved else _Null()):
        out, xl, repr_x, drec = trunk_and_heads(model, batch, cfg=cfg, seed=a.seed, replay=replay,
                                                draw=draw)
    if saved:
        rec["disk_saved"] = {"n": saved.n, "gb": saved.bytes / 2 ** 30,
                             "ram_storage_gb": sum(saved.ram.values()) / 2 ** 30}
        print(f"saved: {rec['disk_saved']}, rss {guard.rss_gb():.1f} GB", flush=True)
    rec["forward_s"] = time.time() - t_f
    rec["cast_policy"] = policy.report()
    rec["draws"] = {"n_calls": len(drec.recorded), "replayed": replay is not None,
                    "mismatch_count": len(drec.mismatch), "mismatch": drec.mismatch[:20]}
    if a.mode == "f64":
        dp = a.out_dir / "draws.pt"
        torch.save({"torch_randn": drec.recorded, "seed": a.seed,
                    "note": "sampled by the float64 reference; every other arm replays these"}, dp)
        rec["draws"]["file"], rec["draws"]["sha256"] = str(dp), sha256_file(dp)
        print(f"DRAWS {dp} {rec['draws']['sha256']} n={len(drec.recorded)}", flush=True)
    torch.save({"atoms": xl.detach().to(torch.float64), "repr_x": repr_x.to(torch.float64)},
               a.out_dir / f"rollout_{a.mode}.pt")

    loss, breakdown, seeds = objective(lab, out)
    rec["loss"], rec["seeded_outputs"] = loss, sorted(seeds)
    rec["breakdown"] = {k: {kk: vv for kk, vv in v.items() if kk != "derived"}
                        for k, v in breakdown.items()}
    print(f"[{time.time()-t0:.0f}s] loss {loss!r} seeds {sorted(seeds)}", flush=True)

    if a.fd_from:
        # The banked gradient is only this forward's if the forward is bitwise the one it came
        # from: same draws (sha256 above) and the same loss to the last bit.
        assert a.mode == "f64" and a.fd and a.expect_loss is not None
        if loss != a.expect_loss:
            raise SystemExit(f"loss {loss!r} != the banked run's {a.expect_loss!r}")
        del out
        gp = a.fd_from
        grads = torch.load(gp, weights_only=False)
        rec["fd_from"] = {"expect_loss": a.expect_loss, "loss_equal": True}
    else:
        t_b = time.time()
        with policy:
            rdt = torch.float64 if a.mode == "f64" else torch.float32
            torch.autograd.backward([out[k].to(rdt) for k in seeds],
                                    [torch.from_numpy(np.asarray(seeds[k])).to(rdt) for k in seeds])
        rec["backward_s"] = time.time() - t_b
        del out
        grads = {}
        for n, p in model.named_parameters():
            grads[n] = p.grad.detach().to(torch.float64) if p.grad is not None else None
            p.grad = None
        gc.collect()
        _trim()
        gp = a.out_dir / f"grads_{a.mode}.pt"
        torch.save(grads, gp)
    rec["grads"] = {"file": str(gp), "sha256": sha256_file(gp),
                    "n_params": len(grads), "n_with_grad": sum(g is not None for g in grads.values()),
                    "n_nonzero": sum(g is not None and bool(g.any()) for g in grads.values())}
    by = {}
    for n, g in grads.items():
        if g is not None:
            e = by.setdefault(head_of(n), {"n": 0, "sq": 0.0})
            e["n"] += 1
            e["sq"] += float((g ** 2).sum())
    rec["by_head"] = by
    rec["squared_gradient_norm"] = sum(e["sq"] for e in by.values())
    print(f"[{time.time()-t0:.0f}s] |g|^2 {rec['squared_gradient_norm']!r} {by}", flush=True)

    if a.fd:
        assert a.mode == "f64"
        params = [(n, p) for n, p in model.named_parameters() if grads[n] is not None]
        gnorm = rec["squared_gradient_norm"] ** 0.5
        base = {n: p.detach().clone() for n, p in params}

        def loss_at(h):
            with torch.no_grad():
                for n, p in params:
                    p.copy_(base[n] + h * (grads[n] / gnorm))
                with bm.cast_policy("removed", "cpu"):
                    o, _, _, _ = trunk_and_heads(model, batch, repr_x=repr_x, draw=draw)
                return objective(lab, o)[0]

        fd = {"direction": "u = g / ||g||, every parameter with a gradient",
              "expected_directional_derivative": gnorm, "rollout": "frozen at the base forward",
              "denoise": "same sigma and noise" if draw is not None else None}
        t_fd = time.time()
        l0 = loss_at(0.0)
        fd["loss_refresh"] = l0
        fd["loss_refresh_abs_diff"] = abs(l0 - loss)
        print(f"FD loss refresh {l0!r} vs {loss!r}: {abs(l0 - loss):.3e}", flush=True)
        fd["h"] = []
        for h in [float(x) for x in a.fd_h.split(",")]:
            lp, lm = loss_at(h), loss_at(-h)
            q = (lp - lm) / (2 * h)
            fd["h"].append({"h": h, "loss_plus": lp, "loss_minus": lm, "quotient": q,
                            "rel_err": abs(q - gnorm) / gnorm})
            print(f"FD h={h:g}: {q!r} vs {gnorm!r}, rel {abs(q - gnorm) / gnorm:.3e}", flush=True)
        with torch.no_grad():
            for n, p in params:
                p.copy_(base[n])
        fd["best_rel_err"] = min(e["rel_err"] for e in fd["h"])
        fd["pass"] = bool(fd["loss_refresh_abs_diff"] <= 1e-12 and fd["best_rel_err"] <= 1e-6)
        fd["seconds"] = time.time() - t_fd
        rec["control_A40"] = fd

    rec["peak_rss_gb"] = guard.peak
    rec["total_s"] = time.time() - t0
    tag = a.mode.upper() + ("_FD" if a.fd_from else "")
    (a.out_dir / f"REF_{tag}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"done {a.mode} in {rec['total_s']:.0f}s, peak RSS {guard.peak:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
