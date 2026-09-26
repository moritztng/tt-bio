#!/usr/bin/env python3
"""A FULL OpenFold3 training step on the card, timed by part.

D32's second half: "no training throughput can be projected from an inference measurement".
Every figure this campaign calls a training step is the TRUNK -- `perf/of3t_perf/step.py`
stops at `z_trunk` with a synthetic unit seed, and `perf/of3t_l1/ladder.py` is a MEMORY
ladder whose wall clock is the allocator probe's. A step is four parts and the trunk is one
of them, so this runs all four in one process, in one tape, in the shipped order:

  trunk      the no_grad recycle prefix plus ONE taped cycle, which is what their loop runs
  diffusion  `OF3DiffusionModule` on N independently noised structures, each at its OWN
             noise level, ALL INSIDE the same tape as the trunk -- so the diffusion loss's
             cotangent reaches the trunk's weights, which is the coupling that makes a
             training step more than a trunk forward plus a separate module
  losses     `tt_bio.train.objectives.af3_loss` at `of3_loss_weights(stage)`, on host, which
             is where it lives: it consumes downloaded coordinates and RETURNS the seeds
  optimizer  `tt_bio.train.optim.AdamW`, constructed as `recipes.py` constructs it

THE INPUT IS THE SHIPPED PIPELINE'S OWN. `perf/of3t_perf/step.py`'s `capture` intercepts a
real `predict_one` at `OF3Trunk.__call__` and at `OF3SampleDiffusion.__call__`, so the
featuriser, the MSA resolve and the input embedder are the production path and there is no
second featurisation to drift. That row's file is imported, not forked.

THE TRUNK OUTPUT IS SUBSTITUTED INTO THE DIFFUSION CONDITIONING. `si_trunk` and `zij_trunk`
are the sampler's arguments 1 and 3; this harness replaces the captured (untaped) pair with
the TAPED trunk's own output, so the tape is one graph from the noised structure back to the
pairformer. A harness that ran the diffusion off the captured conditioning would time the
same kernels and measure a backward that stops at the diffusion boundary.

THE LOSS VALUE IS NOT A CLAIM. The fixture is an inference target with no deposited
structure, so the ground truth handed to `af3_loss` is derived from the model's own
prediction plus noise (SEED fixed, recorded). That makes every term FIRE at the right shape
and the right term count -- which is what a cost breakdown reads -- and makes the loss VALUE
and the gradient DIRECTION meaningless. Neither is reported. `of3t-gradients` owns those.

THE NOISE LEVELS ARE THE CAPTURED SCHEDULE'S, sampled without replacement, so N samples
carry N DISTINCT sigmas -- upstream's `_train_diffusion` shape -- without inventing an EDM
constant this row has no business fixing. Timing reads the sample COUNT, not the value.

    fullstep.py --tokens 384 --cycles 4 --samples 4 --reps 3 --out <json>
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_perf import step as S                                  # noqa: E402

SEED = 20260921


def _dram(dev):
    import ttnn
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)


def _mem_available_gib():
    """Host MemAvailable, beside every timing that has a host half.

    `of3t-p10host` measured the same loss arithmetic at 0.248 s with 17 GiB available and
    1.03-1.21 s at ~1 GiB, so a host timing without this next to it is not comparable to
    another host timing. A chunk loop's own resident set is a perf variable, which is why it
    is recorded per chunk and not once per run.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 3)
    except OSError:
        pass
    return None


def _to_tokens(root, rep):
    """One denoised structure downloaded to host at token scope, detached."""
    import numpy as np
    import ttnn
    from tt_bio import autograd as ag
    raw = ag._unwrap(root) if isinstance(root, ag.Tensor) else root
    return ttnn.to_torch(raw).float().reshape(-1, 3).numpy().astype(np.float64)[rep]


def _cotangents(roots, seeds, dev):
    """The loss's seeds, back on the card as the roots' cotangents."""
    import numpy as np
    import torch
    import ttnn
    from tt_bio import autograd as ag
    cot = []
    for r, g in zip(roots, seeds):
        raw = ag._unwrap(r)
        shp = tuple(int(d) for d in raw.shape)
        h = (torch.from_numpy(np.asarray(g, "float32")).reshape(shp)
             if g is not None else torch.ones(shp))
        cot.append(ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=raw.dtype))
    return cot


def _slot_rank(item):
    """Order the walk's names so the slot a weight gets is one the FORWARD reads.

    `_w_tt` (diffusion transformer, diffusion module, atom transformer) uploads a weight once
    and keeps it in two places: `self._wc[key]` and the attribute the forward reads, e.g.
    `self.w_la`. The walk finds one tensor under both paths, the dedupe keeps whichever name
    sorts first, and `_` sorts before every letter -- so the cache path won and `rebind()` was
    writing AdamW's new weight into a dict nothing reads after `__init__`. The model kept the
    pre-step handle, which the value setter has already de-registered as a tape leaf, so from
    the second rep on those weights took no gradient and were never trained: 270 of 3,152 at
    crop 384, the whole `2,944 -> 2,674` drop of `step_exact_off_384.json` and the
    `2,935 -> 2,660` drop of `step_rekey_b_384.json`. Cache path last, name second.
    """
    n = item[0]
    return ("._wc." in n or n.endswith("._wc"), n)


def declare_all(trunk, sampler, out):
    """Every device weight of the trunk AND the diffusion half, as one tape-leaf set.

    Deduped by tensor IDENTITY, not by name: `tape()` keys a trainable weight on the raw
    handle, so declaring one tensor under two paths would put the same leaf in twice and
    the optimizer would step it twice.
    """
    from tt_bio import autograd as ag
    from tt_bio.train.lora import Parameters
    found = {}
    stats = {}
    for label, root, pre in (("trunk", trunk, "trunk."),
                             ("diffusion", sampler, "diffusion.")):
        f, st = S.walk_weights(root, prefix=pre)
        stats[label] = {"found": len(f), "walk_depth": st["max_depth"],
                        "truncated": st["truncated"]}
        found.update(f)
    # The SLOTS come with the walk: walk_weights already returns the owner object and the
    # attribute key it found each tensor under, which is exactly what _Params.rebind needs.
    # Building a plain dict instead is what made this harness rep 1 train nothing. AdamW.step
    # replaces t.value, and the engine value setter re-keys the TAPE registry for it
    # (autograd.py:220), but the MODEL still holds the handle the walk saw, and only writing
    # it back closes that half.
    by_id, flat, slots, unwritable = {}, {}, {}, []
    for n, (o, k, t) in sorted(found.items(), key=_slot_rank):
        if id(t) in by_id:
            continue
        by_id[id(t)] = n
        flat[n] = ag.parameter(t)
        if isinstance(o, tuple):
            # rebind refuses a tuple slot, correctly: there is nowhere to write the new
            # weight back to. Counted here rather than raised mid-arm, because a timing run
            # that dies ten minutes into its backward tells you nothing about the timing.
            unwritable.append(n)
            continue
        slots[n] = (o, k)
    params = Parameters(flat, slots=slots)
    out["params"] = {
        "declared": len(params),
        "slots": len(params.slots),
        "slots_unwritable": len(unwritable),
        "elements": sum(S._numel(t.value) for t in params.values()),
        "per_root": stats,
        "duplicate_handles_dropped": len(found) - len(params),
        "pairformer_weights": sum(1 for n in params if ".pairformer." in n),
        "diffusion_weights": sum(1 for n in params if n.startswith("diffusion.")),
    }
    print(f"[weights] {len(params)} tensors "
          f"({out['params']['pairformer_weights']} pairformer, "
          f"{out['params']['diffusion_weights']} diffusion), "
          f"{out['params']['elements'] / 1e6:.1f}M elements", flush=True)
    return params


def trunk_forward(trunk, held, cycles, taped):
    """One trunk forward at a pinned cycle count. Same body as `step.cycle_once`, except the
    tape is the CALLER's -- the diffusion half has to be inside it."""
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    snap_args, snap_kwargs = held["trunk_snap"]
    args = S._rehydrate(snap_args, dev)
    kwargs = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items() if k != "progress_fn"}
    trunk.num_cycles = cycles
    if taped:
        return trunk(*args, **kwargs)
    with ag.no_grad():
        return trunk(*args, **kwargs)


def diffusion_pre(sampler, sargs, s_trunk, z_trunk, n_samples, rng, out):
    """Everything the replicate loop does NOT repeat: the pair branch and the sigma draw.

    `_train_diffusion` differentiates `no_samples` independently noised structures at
    `no_samples` distinct sigmas. The pair branch is a loop invariant on their side as on
    ours, so `dc.pair` runs once and the per-replicate work is `dc.single` plus the module,
    which is what makes the per-replicate rate a rate. Hoisting it into its own function is
    also what makes the replicates CHUNKABLE: the invariants sit on the trunk's side of the
    cut, so they are differentiated once whatever the chunk count is.
    """
    import numpy as np
    import ttnn
    from tt_bio.openfold3_sample_diffusion import pad_dim

    (xl_init_dev, si_trunk_cap, si_input_dev, zij_trunk_cap, relpos_dev,
     token_mask_dev, pair_mask_dev, tok_mask_dev, cl0_dev, plm0_dev,
     atom_mask_col_dev, atom_mask_col_na_dev, atom_to_token_idx_tt,
     npe_flat_idx_tt, npe_zij_mask, enc_key_block_idxs_tt, enc_valid_mask,
     enc_mask_bias, enc_pair_mask, atom_to_token_mean_tt,
     token_mask_pad_tt, tok_mask_col_pad_tt,
     n_atom, NP, nb, n_token, n_tok_pad,
     noise_schedule, rots_list, trans_list, noise_list, t_list, c_tau_list,
     step_scale) = sargs[:35]

    # THE COUPLING. Arguments 1 and 3 are the trunk's own output; the taped pair replaces the
    # captured one so the backward runs from the denoised structure into the pairformer.
    si_trunk_dev = s_trunk if s_trunk is not None else si_trunk_cap
    zij_trunk_dev = z_trunk if z_trunk is not None else zij_trunk_cap
    out["coupled_to_trunk"] = s_trunk is not None and z_trunk is not None

    dc = sampler.dc
    dt = sampler._act_dtype
    atom_mask_host = ttnn.to_torch(atom_mask_col_na_dev).float().reshape(n_atom)
    xl_host = ttnn.to_torch(xl_init_dev).float().reshape(n_atom, 3)
    zij_dev = dc.pair(zij_trunk_dev, relpos_dev, pair_mask_dev)
    zij_pad = pad_dim(zij_dev, dt, n_token, n_tok_pad, dims=2)

    # N DISTINCT noise levels, drawn from the captured schedule without replacement.
    pool = sorted({float(x) for x in t_list}, reverse=True)
    idx = rng.choice(len(pool), size=min(n_samples, len(pool)), replace=False)
    sigmas = [pool[int(i)] for i in idx]
    while len(sigmas) < n_samples:                 # schedule shorter than the sample count
        sigmas.append(pool[int(rng.integers(len(pool)))])
    out["sigmas"] = [round(s, 4) for s in sigmas]
    out["distinct_sigmas"] = len(set(sigmas))

    # THE WHOLE NOISE SET, DRAWN HERE, INDEXED BY REPLICATE. `tt_bio/sample_chunks.py` states
    # the codebase invariant: "the samplers draw every sample's initial noise, augmentation and
    # step noise over the whole sample axis on the host before any chunking, so a sample gets
    # the same draws at every width." This harness used to thread one generator into the
    # replicate loop and draw inside it, which makes a replicate's noise a function of its
    # POSITION IN THE DRAW ORDER rather than of its index. On one chip, in order, that is
    # invisible; it breaks the moment the axis is partitioned any other way -- a sample-batched
    # call, a re-run chunk, a rank holding a round-robin slice -- and then a batched arm and a
    # per-replicate arm are different computations and the gradient A/B compares nothing.
    # Drawing the set in one call at this point in the stream is bit-identical to the 48
    # sequential draws it replaces (`tests/test_replicate_noise_is_indexed.py`), so no banked
    # number moves; what changes is that C=4, C=8, a batch of 4 and a 2-rank split now agree
    # by construction.
    out["noise_draw"] = "whole sample axis, drawn before any chunking, indexed by replicate"
    noise = rng.standard_normal((n_samples, int(n_atom), 3)).astype("float32")
    out["n_samples"] = n_samples
    out["n_atom"] = int(n_atom)
    out["n_token"] = int(n_token)

    # The representative atom per token, taken from the shipped atom->token map rather than
    # from a fixture field this harness does not have. `af3_loss` is a TOKEN-scope objective
    # -- `losses.distogram` documents its logits as "[N, N, 64] on representative atoms" --
    # so handing it n_atom coordinates builds three [n_atom, n_atom, 64] logit arrays, which
    # at 384 tokens is 4.6 GB each. The first run of this harness did exactly that and was
    # killed at 41.8 GB RSS.
    a2t = ttnn.to_torch(atom_to_token_idx_tt).reshape(-1).long()[:n_atom].numpy()
    rep = np.full(int(n_token), -1, dtype=np.int64)
    for ai, ti in enumerate(a2t):
        if 0 <= ti < n_token and rep[ti] < 0:
            rep[ti] = ai
    out["tokens_without_an_atom"] = int((rep < 0).sum())
    rep = np.clip(rep, 0, n_atom - 1)
    out["rep_atom_index"] = "first atom of each token, from atom_to_token_idx"
    return {"sargs": sargs, "sigmas": sigmas, "noise": noise, "rep": rep, "zij_pad": zij_pad,
            "si_trunk": si_trunk_dev, "atom_mask_host": atom_mask_host, "xl_host": xl_host}


def cut(t):
    """A detached leaf over the trunk's own buffer: where a chunk's tape ends.

    The trunk's ~20 s of backward is per backward ENTRY, not per replicate -- checkpointing
    makes its 3,537 nodes get recomputed and traversed several times each while a replicate's
    1,590 nodes are traversed once. So a chunk loop whose every backward reached the trunk
    would pay that 20 s twelve times. Cutting here instead lets each chunk's cotangent
    ACCUMULATE on this leaf, and the trunk is differentiated once, after the last chunk.

    The buffer is the trunk's, so the leaf must never free it: `evictable` off is what stops
    a chunk's backward from moving a handle the trunk tape still holds.
    """
    from tt_bio import autograd as ag
    d = ag.Tensor(ag._unwrap(t), requires_grad=True)
    d.evictable = False
    return d


def cut_tree(v, pairs):
    """`cut` every taped tensor in a cache entry, recording (original, leaf) for the seeding.

    A cache entry is a tensor or a tuple/list of them (`openfold3_sample_diffusion._free_cached`
    says so), and an entry that is a raw handle has no node and is passed through: there is
    nothing to differentiate and nothing to seed.
    """
    from tt_bio import autograd as ag
    if isinstance(v, ag.Tensor):
        leaf = cut(v)
        pairs.append((v, leaf))
        return leaf
    if isinstance(v, (tuple, list)):
        return type(v)(cut_tree(x, pairs) for x in v)
    return v


def diffusion_chunk(sampler, pre, idxs, out, si_trunk=None, zij_pad=None,
                    cache=None, batch=1):
    """One chunk of replicates: `dc.single` plus the denoiser, over `idxs` noised structures.

    Every denoised structure stays on the card as a tape root; there is no EDM update
    chaining one to the next and no host download between them. `si_trunk` / `zij_pad`
    override the invariants with the CUT leaves when the step is chunked, so this chunk's
    tape ends at them instead of running on into the trunk.

    `cache` is the replicate loop's deep invariant, the DiT per-block pair bias. Rebuilding
    it per chunk costs 3.75 s of forward per chunk (measured: sample 0 of each chunk reads
    4.15-4.32 s against 0.39-0.51 s for the rest, `arm3_s48_c4_384`) and differentiates the
    same subgraph once per chunk on the way back. So the caller primes it once on the TRUNK's
    side of the cut and hands the cut copy in here; `cache=None` rebuilds, which is what an
    unchunked step does anyway.

    `batch` is the SAMPLE AXIS: how many of this chunk's replicates go through one call of
    the diffusion module. At 1 the loop is what it always was. Above 1 the module runs the
    atom-level encoder and decoder once per replicate and the 24-block token DiT ONCE for
    all of them, which is where the dispatch cost is -- see
    `OF3DiffusionModule._denoise_samples`. The width is resolved by the tree's own
    `resolve_sample_chunk_width`, so a chunk of 4 at batch 3 runs 2+2 rather than 3+1.

    The replicates are named by INDEX, not by a sigma sublist: a replicate's noise comes from
    `pre["noise"][k]`, so the same k is the same noised structure at any batch width, in any
    chunk, on any rank.
    """
    import math
    import torch
    import ttnn
    from tt_bio.openfold3_sample_diffusion import fourier_noise_emb, pad_dim

    (xl_init_dev, si_trunk_cap, si_input_dev, zij_trunk_cap, relpos_dev,
     token_mask_dev, pair_mask_dev, tok_mask_dev, cl0_dev, plm0_dev,
     atom_mask_col_dev, atom_mask_col_na_dev, atom_to_token_idx_tt,
     npe_flat_idx_tt, npe_zij_mask, enc_key_block_idxs_tt, enc_valid_mask,
     enc_mask_bias, enc_pair_mask, atom_to_token_mean_tt,
     token_mask_pad_tt, tok_mask_col_pad_tt,
     n_atom, NP, nb, n_token, n_tok_pad,
     noise_schedule, rots_list, trans_list, noise_list, t_list, c_tau_list,
     step_scale) = pre["sargs"][:35]

    from tt_bio.sample_chunks import resolve_sample_chunk_width

    dc, dm = sampler.dc, sampler.dm
    dt = sampler._act_dtype
    si_dev_trunk = pre["si_trunk"] if si_trunk is None else si_trunk
    zij = pre["zij_pad"] if zij_pad is None else zij_pad
    atom_mask_host, xl_host = pre["atom_mask_host"], pre["xl_host"]
    sigmas_all, noise_all = pre["sigmas"], pre["noise"]
    inv_cache = {} if cache is None else cache
    idxs = list(idxs)

    if batch > 1 and not getattr(dm.dit, "supports_multiplicity", False):
        # A capability that is reachable but silently declines is this fleet's documented way
        # of losing a whole campaign's win (`rfd3_bias.py:229`). Refuse loudly instead.
        raise RuntimeError("--dit-batch asked for a sample axis the DiT does not declare")
    width = resolve_sample_chunk_width(len(idxs), batch) if batch > 1 else 1

    def prep(k):
        """Replicate k's four per-sample tensors. Everything here is host or `dc.single`."""
        t = sigmas_all[k]
        xl_noisy = xl_host + torch.from_numpy(noise_all[k]) * t
        n_emb = fourier_noise_emb(t, sampler.sigma_data, sampler.fourier_w, sampler.fourier_b)
        si_dev = dc.single(si_dev_trunk, si_input_dev,
                           sampler._to_dev(n_emb.reshape(1, 1, 256)), tok_mask_dev)
        si_pad = pad_dim(si_dev, dt, n_token, n_tok_pad)
        rl_noisy = xl_noisy * atom_mask_host[:, None] / math.sqrt(t * t + sampler.sigma_data ** 2)
        rl_noisy_dev = sampler._to_dev(sampler._pad_atoms_host(rl_noisy, n_atom, NP))
        xl_noisy_dev = sampler._to_dev((xl_noisy * atom_mask_host[:, None]).unsqueeze(0))
        return si_pad, rl_noisy_dev, xl_noisy_dev, t

    def denoise(si, rl, xl, t, samples=None):
        return dm(si_dev_trunk, si, zij, cl0_dev, plm0_dev, rl, xl,
                  atom_mask_col_dev, atom_mask_col_na_dev, atom_to_token_idx_tt,
                  npe_flat_idx_tt, npe_zij_mask, enc_key_block_idxs_tt, enc_valid_mask,
                  enc_mask_bias, enc_pair_mask, atom_to_token_mean_tt,
                  token_mask_pad_tt, tok_mask_col_pad_tt,
                  n_atom, NP, nb, n_token, n_tok_pad, t, sampler.sigma_data,
                  cache=inv_cache, samples=samples)

    roots, per_sample, batches = [], [], []
    for start in range(0, len(idxs), width):
        part = idxs[start:start + width]
        t0 = time.perf_counter()
        built = [prep(k) for k in part]
        if len(built) == 1:
            got = [denoise(*built[0])]
        else:
            got = denoise(None, None, None, None, samples=built)
        ttnn.synchronize_device(dm.device if hasattr(dm, "device") else dc.device)
        dt_s = time.perf_counter() - t0
        roots.extend(got)
        # THE STAMP. An A/B that agrees must read as "the axis did not reach" before it reads
        # as "the axis did not help", so the width that actually ran is in the artifact, per
        # call, beside the shape the DiT saw.
        batches.append({"first": part[0], "n": len(part), "s": round(dt_s, 3)})
        per_sample.extend([round(dt_s / len(part), 3)] * len(part))
        print(f"  [diffusion] samples {part[0]}..{part[-1]} (batch {len(part)}) "
              f"sigma {sigmas_all[part[0]]:8.3f}  {dt_s:7.3f}s  "
              f"{dt_s / len(part):6.3f}s/replicate", flush=True)
    out.setdefault("per_sample_s", []).extend(per_sample)
    out.setdefault("batches", []).extend(batches)
    out["dit_batch_width"] = width
    out["per_sample_s_basis"] = ("measured per replicate at width 1; the batch wall divided "
                                 "by its width above 1")
    return roots


def host_losses(roots, rep, weights, rng, out):
    """`af3_loss` on host, at token scope, returning the per-root cotangent on `pred_xyz`.

    The loss is where it lives. `tt_bio.train.objectives.af3_loss` is numpy: a training step
    downloads the predicted coordinates, evaluates every term on host and hands the seeds
    back, so the "loss heads" part of a step is a PCIe round trip plus host arithmetic, and
    pricing it needs both halves timed together. They are.
    """
    import numpy as np
    import ttnn
    from tt_bio.train import losses as L
    from tt_bio.train.objectives import af3_loss
    from tt_bio import autograd as ag

    dl_s, host_s, seeds_out, fired = 0.0, 0.0, [], None
    print(f"  [losses] {len(roots)} roots, token scope n={len(rep)}", flush=True)
    for k, r in enumerate(roots):
        t0 = time.perf_counter()
        raw = ag._unwrap(r) if isinstance(r, ag.Tensor) else r
        atoms = ttnn.to_torch(raw).float().reshape(-1, 3).numpy().astype(np.float64)
        n_atom = atoms.shape[0]
        pred = atoms[rep]                      # token scope, one atom per token
        dl_s += time.perf_counter() - t0
        print(f"  [losses] root {k} downloaded {n_atom} atoms -> {pred.shape[0]} tokens "
              f"in {time.perf_counter() - t0:.2f}s", flush=True)

        t0 = time.perf_counter()
        n = pred.shape[0]
        # The fixture has no deposited structure, so the ground truth is the prediction plus
        # noise at a fixed seed. Every term fires at the right shape; no VALUE is claimed.
        true_xyz = pred + rng.standard_normal(pred.shape) * 1.0
        coord_mask = np.ones(n)
        is_nuc = np.zeros(n, bool)
        is_poly = np.ones(n, bool)
        true_dist, pred_dist = L._pdist(true_xyz), L._pdist(pred)
        pair_mask = coord_mask[:, None] * coord_mask[None, :]
        lddt, lddt_w = L.atom_bespoke_lddt(pred, true_xyz, is_nuc, is_poly,
                                           coord_mask.astype(bool))
        idxs = np.arange(n)
        lg = lambda *s: rng.standard_normal(s) * 0.5
        labels = {"true_xyz": true_xyz, "coord_mask": coord_mask, "true_dist": true_dist,
                  "lddt_pair_mask": L.lddt_mask(true_dist, pair_mask, is_nuc),
                  "bond_mask": np.zeros((n, n)),
                  "per_atom_lddt": lddt, "per_atom_weight": lddt_w,
                  "frame_atom_index": np.stack(
                      [np.clip(idxs - 1, 0, n - 1), idxs, np.clip(idxs + 1, 0, n - 1)], -1),
                  "is_dna": np.zeros(n), "is_rna": np.zeros(n), "is_ligand": np.zeros(n)}
        outputs = {"pred_xyz": pred, "pred_dist": pred_dist,
                   "distogram_logits": lg(n, n, 64), "pde_logits": lg(n, n, 64),
                   "pae_logits": lg(n, n, 64), "plddt_logits": lg(n, 50),
                   "resolved_logits": lg(n, 2)}
        total, breakdown, seeds = af3_loss(labels, outputs, weights)
        host_s += time.perf_counter() - t0
        g = seeds.get("pred_xyz")
        # The cotangent goes back on the ATOM tensor the module returned: the loss touched
        # one atom per token, so every other atom's seed is a true zero, not a dropped term.
        g_atom = np.zeros((n_atom, 3))
        if g is not None:
            np.add.at(g_atom, rep, np.asarray(g, np.float64))
        seeds_out.append(g_atom)
        print(f"  [losses] root {k} af3_loss done, {sum(1 for v in breakdown.values() if v['value'] is not None)} terms", flush=True)
        if fired is None:
            fired = {k: {"weight": v["weight"], "skipped": v.get("skipped"),
                         "fired": bool(v["weight"] != 0.0 and v["value"] is not None)}
                     for k, v in breakdown.items()}
    out["terms"] = fired
    out["terms_fired"] = sum(1 for v in (fired or {}).values() if v["fired"])
    out["scope"] = "token, one representative atom per token"
    out["download_s"] = round(dl_s, 3)
    out["host_loss_s"] = round(host_s, 3)
    out["value_claimed"] = False
    return seeds_out


# The three terms of `af3_loss` whose input is the replicate's OWN structure. Everything else
# reads the trunk pair representation and the detached rollout once per step, whatever the
# replicate count is, so those five must not ride inside a chunk loop.
REPLICATE_TERMS = ("mse", "smooth_lddt", "bond")


def step_fixture(roll, rep, rng, out):
    """One label set and one set of head logits per STEP, not per replicate.

    The fixture is an inference target with no deposited structure, so the ground truth is
    the model's own rollout plus noise at a fixed seed. Every term fires at the right shape
    and the right term count, which is what a cost breakdown reads; the loss VALUE and the
    gradient DIRECTION stay meaningless and neither is reported.

    Drawn ONCE, because a training step has one ground truth and one rollout. `host_losses`
    drew a fresh set per root and ran all seven terms against it, and the three compounding
    errors in that -- whole set per root, fixture inside the stopwatch, host memory pressure
    -- are what made a 0.248 s loss set read as 6.3 s and as the sample axis
    (`of3t-p10host`, 2026-09-26).
    """
    import numpy as np
    from tt_bio.train import losses as L

    t0 = time.perf_counter()
    n = roll.shape[0]
    true_xyz = roll + rng.standard_normal(roll.shape) * 1.0
    coord_mask = np.ones(n)
    is_nuc = np.zeros(n, bool)
    is_poly = np.ones(n, bool)
    true_dist = L._pdist(true_xyz)
    pair_mask = coord_mask[:, None] * coord_mask[None, :]
    lddt, lddt_w = L.atom_bespoke_lddt(roll, true_xyz, is_nuc, is_poly, coord_mask.astype(bool))
    idxs = np.arange(n)
    labels = {"true_xyz": true_xyz, "coord_mask": coord_mask, "true_dist": true_dist,
              "lddt_pair_mask": L.lddt_mask(true_dist, pair_mask, is_nuc),
              "bond_mask": np.zeros((n, n)),
              "per_atom_lddt": lddt, "per_atom_weight": lddt_w,
              "frame_atom_index": np.stack(
                  [np.clip(idxs - 1, 0, n - 1), idxs, np.clip(idxs + 1, 0, n - 1)], -1),
              "is_dna": np.zeros(n), "is_rna": np.zeros(n), "is_ligand": np.zeros(n)}
    out["labels_s"] = round(time.perf_counter() - t0, 3)

    # The five head terms read logits the confidence module produces from the trunk pair
    # representation and the DETACHED rollout. This harness draws them instead of computing
    # them, so they cost a fixture draw and no device work -- and it is timed separately,
    # because drawing three (n, n, 64) float64 arrays inside the stopwatch was 0.220 s of the
    # 0.485 s that got priced as model cost.
    t0 = time.perf_counter()
    lg = lambda *s: rng.standard_normal(s) * 0.5
    logits = {"distogram_logits": lg(n, n, 64), "pde_logits": lg(n, n, 64),
              "pae_logits": lg(n, n, 64), "plddt_logits": lg(n, 50),
              "resolved_logits": lg(n, 2)}
    out["fixture_logits_s"] = round(time.perf_counter() - t0, 3)
    return labels, logits


def replicate_losses(roots, rep, labels, weights, out):
    """The per-replicate half of `af3_loss`: mse, smooth_lddt, bond, per noised structure.

    `bond` is at weight 0 in `initial_training`, so the two that fire are 0.003 s between
    them plus the replicate's own `_pdist`. Returns the cotangent on each root's coordinates.
    """
    import numpy as np
    import ttnn
    from tt_bio.train import losses as L
    from tt_bio.train.objectives import af3_loss
    from tt_bio import autograd as ag

    w = {k: v for k, v in weights.items() if k in REPLICATE_TERMS}
    dl_s, host_s, seeds_out = 0.0, 0.0, []
    for k, r in enumerate(roots):
        t0 = time.perf_counter()
        raw = ag._unwrap(r) if isinstance(r, ag.Tensor) else r
        atoms = ttnn.to_torch(raw).float().reshape(-1, 3).numpy().astype(np.float64)
        n_atom = atoms.shape[0]
        pred = atoms[rep]                      # token scope, one atom per token
        dl_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        total, breakdown, seeds = af3_loss(
            labels, {"pred_xyz": pred, "pred_dist": L._pdist(pred)}, w)
        host_s += time.perf_counter() - t0
        g = seeds.get("pred_xyz")
        # The cotangent goes back on the ATOM tensor the module returned: the loss touched
        # one atom per token, so every other atom's seed is a true zero, not a dropped term.
        g_atom = np.zeros((n_atom, 3))
        if g is not None:
            np.add.at(g_atom, rep, np.asarray(g, np.float64))
        seeds_out.append(g_atom)
        if "terms" not in out:
            out["terms"] = {}
        out["terms"].update({t: {"weight": v["weight"], "skipped": v.get("skipped"),
                                 "fired": bool(v["weight"] != 0.0 and v["value"] is not None),
                                 "scope": "per replicate"}
                             for t, v in breakdown.items()})
    out["download_s"] = round(out.get("download_s", 0.0) + dl_s, 3)
    out["replicate_loss_s"] = round(out.get("replicate_loss_s", 0.0) + host_s, 3)
    out["replicates_scored"] = out.get("replicates_scored", 0) + len(roots)
    print(f"  [losses] {len(roots)} replicates scored, {dl_s:.3f}s download "
          f"{host_s:.3f}s arithmetic", flush=True)
    return seeds_out


def step_losses(roll, labels, logits, weights, out):
    """The once-per-step half: distogram, plddt, pde, pae, resolved.

    Their seeds land on the LOGITS (`objectives._SEED`), which upstream builds under
    `no_grad` from the trunk pair representation and a detached rollout, so no cotangent from
    these reaches the replicate graph. Running them once is the model's shape, not a saving
    this harness invented.
    """
    from tt_bio.train import losses as L
    from tt_bio.train.objectives import af3_loss

    w = {k: v for k, v in weights.items() if k not in REPLICATE_TERMS}
    t0 = time.perf_counter()
    outputs = dict(logits)
    outputs["pred_xyz"] = roll
    outputs["pred_dist"] = L._pdist(roll)
    total, breakdown, seeds = af3_loss(labels, outputs, w)
    out["step_loss_s"] = round(time.perf_counter() - t0, 3)
    out.setdefault("terms", {}).update(
        {t: {"weight": v["weight"], "skipped": v.get("skipped"),
             "fired": bool(v["weight"] != 0.0 and v["value"] is not None),
             "scope": "once per step"} for t, v in breakdown.items()})
    out["terms_fired"] = sum(1 for v in out["terms"].values() if v["fired"])
    out["scope"] = "token, one representative atom per token"
    out["shape"] = ("model: per-replicate terms per root, head terms once per step")
    out["value_claimed"] = False
    print(f"  [losses] step heads {out['step_loss_s']:.3f}s, "
          f"{out['terms_fired']} of {len(out['terms'])} terms fired", flush=True)


def grad_snapshot(params, held, row, bankdir):
    """Bank the unchunked gradient, then grade the chunked one against it.

    Chunking changes the ORDER the per-replicate contributions are summed in, nothing else:
    the diffusion loss is a sum over independent noised structures given the trunk output, so
    the derivative of the sum is the sum of the per-chunk derivatives. That is an identity,
    and this is the measurement that says the code implements it -- per parameter, because a
    whole-model norm would hide one dead tensor among 3,152.

    The bank goes to DISK, one `.npy` per parameter, and the comparison streams it back one
    parameter at a time. Holding 381.3 M gradient elements in host memory beside a step whose
    own resident set is ~10 GiB is what got the first attempt at this arm OOM-killed on a
    30 GiB box.
    """
    import numpy as np
    import ttnn
    bankdir = Path(bankdir)
    if not held:
        bankdir.mkdir(parents=True, exist_ok=True)
        names = {}
        for i, (n, t) in enumerate(params.items()):
            g = getattr(t, "grad", None)
            if g is None:
                continue
            f = bankdir / f"{i}.npy"
            np.save(f, ttnn.to_torch(g).float().numpy().ravel().astype(np.float32))
            names[n] = f
        print(f"  [grad_ab] banked {len(names)} unchunked gradients to {bankdir}", flush=True)
        return {"bank": names}

    bank = held["bank"]
    worst_cos, worst_l2, worst_cos_n, worst_l2_n = 2.0, 0.0, None, None
    missing, extra, compared = [], [], 0
    for n, t in params.items():
        g = getattr(t, "grad", None)
        if g is None:
            if n in bank:
                missing.append(n)
            continue
        if n not in bank:
            extra.append(n)
            continue
        b = np.load(bank[n])
        c = ttnn.to_torch(g).float().numpy().ravel().astype(np.float32)
        if c.shape != b.shape:
            missing.append(f"{n}:shape {b.shape}->{c.shape}")
            continue
        nb, nc = float(np.linalg.norm(b)), float(np.linalg.norm(c))
        if nb == 0.0 and nc == 0.0:
            cos, l2 = 1.0, 0.0
        elif nb == 0.0 or nc == 0.0:
            cos, l2 = 0.0, 1.0
        else:
            cos = float(np.dot(b, c) / (nb * nc))
            l2 = float(np.linalg.norm(c - b) / nb)
        compared += 1
        if cos < worst_cos:
            worst_cos, worst_cos_n = cos, n
        if l2 > worst_l2:
            worst_l2, worst_l2_n = l2, n
        del b, c
    rep = {"compared": compared, "banked": len(bank),
           "grad_missing_in_chunked": missing[:8], "grad_only_in_chunked": extra[:8],
           "n_missing": len(missing), "n_extra": len(extra),
           "worst_cos": round(worst_cos, 9), "worst_cos_param": worst_cos_n,
           "worst_rel_l2": round(worst_l2, 9), "worst_rel_l2_param": worst_l2_n,
           "bar": "cos >= 0.9999 or rel_l2 <= 1e-2, per parameter",
           "verdict": "PASS" if (compared and not missing and not extra
                                 and (worst_cos >= 0.9999 or worst_l2 <= 1e-2)) else "FAIL"}
    row["grad_ab"] = rep
    print(f"  [grad_ab] {compared} parameters compared, worst cos {worst_cos:.7f} "
          f"({worst_cos_n}), worst rel_l2 {worst_l2:.3e} -> {rep['verdict']}", flush=True)
    return {"bank": bank, "report": rep}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=4,
                    help="trunk cycles to PIN; their training draws it from U{1..4} per step")
    ap.add_argument("--samples", type=int, default=4,
                    help="diffusion samples differentiated per step; theirs is 48 in "
                         "initial_training and 32 in finetune_1/2")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=0,
                    help="replicates per chunk. The chunk tape is cut at the trunk output, "
                         "so the trunk backward still runs exactly ONCE per step whatever "
                         "this is. 0 keeps every replicate in one tape")
    ap.add_argument("--dit-batch", type=int, default=1,
                    help="replicates through ONE diffusion-module call (the sample axis). "
                         "1 is the per-replicate loop; above 1 the 24-block token DiT runs "
                         "once for the batch and the atom-level stages stay at batch 1. "
                         "Capped by --chunk, and the width is rebalanced so the widest "
                         "batch is as narrow as the batch count allows.")
    ap.add_argument("--chunk-per-rep", default="",
                    help="comma-separated C per rep, one warm process. The replicate "
                         "subgraph's backward is what the campaign's 5.6x projection rests "
                         "on and nobody has measured it past 4 replicates; four chunk sizes "
                         "in one process against one clock is that measurement")
    ap.add_argument("--grad-ab", type=int, default=0,
                    help="the identity control: rep 0 unchunked, rep 1 chunked at this C, "
                         "one process, one set of weights, the SAME replicate noise, no "
                         "optimizer between them. Reports per-parameter cos and rel_l2, "
                         "which is what says the chunked gradient IS the unchunked one")
    ap.add_argument("--loss-shape", choices=("model", "harness"), default="model",
                    help="model: the per-replicate terms per root and the five head terms "
                         "once per step, which is what their step runs. harness: the whole "
                         "7-term set per root, which is what this file did before "
                         "of3t-p10host took it apart and is kept as the control")
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--no-tape", action="store_true",
                    help="run the same scope UNTAPED, for D32's ratio at step scope")
    ap.add_argument("--no-optimizer", action="store_true")
    ap.add_argument("--renorm-per-rep", default="",
                    help="comma-separated 1/0 per rep, flipping ag.SOFTMAX_BW_RENORM in "
                         "THIS process. The lever is a module global read inside the "
                         "backward closure, so an ON rep and an OFF rep can share one warm "
                         "process and one set of weights, which is the only way to price it "
                         "against a run-to-run spread this large")
    ap.add_argument("--no-exact", action="store_true",
                    help="run the whole step inside ag.exact_training(False), which no "
                         "measurement at the shipped shape has ever been taken in. There is "
                         "no environment variable for the switch (autograd.py:1491), so this "
                         "flag is the only way to reach it from a command line")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.dit_batch < 1:
        ap.error("--dit-batch is a sample-axis width, so 1 is the narrowest it can be")
    if a.grad_ab:
        # One process, one weight set, no step between the two arms: anything else compares
        # two different models rather than two ways of summing one gradient.
        a.reps, a.no_optimizer, a.chunk = 2, True, 0

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "renorm": os.environ.get("TT_BIO_SOFTMAX_BW_RENORM"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": os.getloadavg()},
        "config": {"crop": a.tokens, "batch": 1, "cards": 1, "cycles_pinned": a.cycles,
                   "diffusion_samples": a.samples, "stage": a.stage,
                   "taped": not a.no_tape, "chunk": a.chunk or None,
                   "loss_shape": a.loss_shape,
                   "exact_training": not a.no_exact,
                   "rng": "diffusion noise and loss fixture drawn from SEPARATE streams, so "
                          "a chunked arm and an unchunked one noise the same structures"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        """Write the artifact, clock included, every time.

        The clock used to be written once, after the rep loop, from `during.__exit__`. A run
        that is killed mid-step -- and on a 30 GiB host every 48-replicate arm so far has been
        -- loses it, and a speed number without its DURING clock is not a measurement on this
        fleet. `clk` is in scope from the first sample on, so there is no reason to hold it.
        """
        if _clk[0] is not None:
            out.setdefault("env", {})["aiclk_during"] = _clk[0].summary()
        a.out.write_text(json.dumps(out, indent=1, default=str))

    _clk = [None]
    dump()

    with during() as clk, contextlib.ExitStack() as es:
        _clk[0] = clk
        try:
            import numpy as np
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio.tenstorrent import get_device
            from tt_bio.train.losses import of3_loss_weights
            from tt_bio.train.optim import AdamW

            # THE LEVER, entered before the capture so no phase in this process runs under
            # a setting the step does not. `exact_training` is a context manager over a
            # module global, so it has to stay open across the whole rep loop; the ExitStack
            # holds it there without reindenting 200 lines of step around it.
            es.enter_context(ag.exact_training(not a.no_exact))
            out["exact"] = {
                "requested": "OFF" if a.no_exact else "ON",
                "ops_a_tape_would_run_exact": list(ag.exact_training_ops()),
                "softmax_stats_before": dict(ag.EXACT_SOFTMAX_STATS),
                "layer_norm_stats_before": dict(ag.EXACT_LAYER_NORM_STATS),
            }
            print(f"[exact] requested {out['exact']['requested']}, a tape opened now would "
                  f"run {out['exact']['ops_a_tape_would_run_exact'] or 'NO ops'} exact",
                  flush=True)

            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            if "sampler" not in held:
                raise SystemExit("the fold never reached the sampler; no diffusion half")
            sampler, sargs, _skw = held["sampler"]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = declare_all(trunk, sampler, out)
            weights = of3_loss_weights(a.stage)
            out["loss_weights"] = weights
            opt = None if a.no_optimizer else AdamW(params, lr=3e-4)
            out["optimizer"] = {"class": "tt_bio.train.optim.AdamW", "lr": 3e-4,
                                "clip_norm": None if opt is None else opt.clip_norm,
                                "params": 0 if opt is None else len(opt.params)}
            dump()

            # D56's lever, read off the module rather than off the environment, with its own
            # REACH counter beside it: a flag that is set and never fires is the fleet's
            # standing  case and the counter is what tells them
            # apart. Sampled again after the reps so the delta is the work's, not the import's.
            out["renorm"] = {"flag": bool(ag.SOFTMAX_BW_RENORM),
                             "stats_before": dict(ag.SOFTMAX_BW_RENORM_STATS)}
            plan = [bool(int(x)) for x in a.renorm_per_rep.split(",") if x != ""]
            cplan = [int(x) for x in a.chunk_per_rep.split(",") if x != ""]
            out["chunk_per_rep_plan"] = cplan or None
            out["renorm"]["per_rep_plan"] = plan or None
            reps = []
            grad_ab: dict = {}
            for rep in range(a.reps):
                # Two streams, not one. The replicate noise must be identical between a
                # chunked arm and an unchunked one or the gradient A/B compares two different
                # sets of structures; drawing the loss fixture from the same stream made the
                # draw order depend on the loss SHAPE. `rng_diff` is now consumed entirely
                # inside `diffusion_pre` -- the sigma draw and then the whole 48-long noise
                # set -- so the replicate loop draws nothing and the chunk width, the batch
                # width and the rank split cannot reach the numbers.
                seed_rep = 0 if a.grad_ab else rep
                rng_diff = np.random.default_rng(SEED + seed_rep)
                rng_loss = np.random.default_rng(SEED + 10_000 + seed_rep)
                row = {"rep": rep, "cold": rep == 0, "dit_batch": a.dit_batch}
                # In `reps` from the start, so the per-chunk `dump()` inside a 12-chunk step
                # lands in the artifact instead of in a local nobody has written out yet.
                reps.append(row)
                out["reps"] = reps
                if plan:
                    ag.SOFTMAX_BW_RENORM = plan[rep % len(plan)]
                row["renorm_flag"] = bool(ag.SOFTMAX_BW_RENORM)
                rs0 = dict(ag.SOFTMAX_BW_RENORM_STATS)
                ex0 = (dict(ag.EXACT_SOFTMAX_STATS), dict(ag.EXACT_LAYER_NORM_STATS))
                for t in params.values():
                    t.grad = None
                ttnn.synchronize_device(dev)

                # --- 1. trunk -------------------------------------------------------------
                # THE STEP'S OWN CLOCK. `step_s` below is a SUM of the phase timers; this is
                # ONE clock across all of them and their difference is `unaccounted_s`, which
                # is the only way to see the cost that sits between two timers.
                wall0 = t0 = time.perf_counter()
                if a.cycles > 1:
                    trunk_forward(trunk, held, a.cycles - 1, taped=False)
                    ttnn.synchronize_device(dev)
                row["trunk_nograd_prefix_s"] = round(time.perf_counter() - t0, 3)

                ctx = ag.no_grad() if a.no_tape else ag.tape()
                d_out, l_out = {}, {}
                want = (a.grad_ab if (a.grad_ab and rep == 1)
                        else cplan[rep % len(cplan)] if cplan else a.chunk)
                chunk = want if (not a.no_tape and 0 < want < a.samples) else 0
                row["chunk"] = chunk or None
                peak_dram = 0
                with ctx:
                    t0 = time.perf_counter()
                    s_tr, z_tr = trunk_forward(trunk, held, 1, taped=not a.no_tape)
                    ttnn.synchronize_device(dev)
                    row["trunk_taped_cycle_s"] = round(time.perf_counter() - t0, 3)
                    row["trunk_s"] = round(row["trunk_nograd_prefix_s"]
                                           + row["trunk_taped_cycle_s"], 3)
                    row["dram_after_trunk"] = _dram(dev)
                    # ASSERT THE SCOPE FROM THE MECHANISM, NOT FROM THE ARGUMENT. `tape()`
                    # installs the exact ops for its own extent (autograd.py `_training_exact`),
                    # so inside the tape is the only place the answer is readable at all.
                    row["exact_installed_in_trunk_tape"] = {
                        "softmax": ag.exact_softmax_installed(),
                        "layer_norm": ag.exact_layer_norm_installed()}
                    print(f"  [trunk] prefix {row['trunk_nograd_prefix_s']:.2f}s  taped cycle "
                          f"{row['trunk_taped_cycle_s']:.2f}s  exact-in-tape softmax="
                          f"{row['exact_installed_in_trunk_tape']['softmax']} layer_norm="
                          f"{row['exact_installed_in_trunk_tape']['layer_norm']}", flush=True)
                    dump()

                    # --- 2. diffusion ------------------------------------------------------
                    # The invariants are inside the TRUNK's tape whether or not the replicates
                    # are chunked, so `dc.pair` is differentiated exactly once either way.
                    t0 = time.perf_counter()
                    pre = diffusion_pre(sampler, sargs, s_tr, z_tr, a.samples, rng_diff, d_out)
                    diff_s = time.perf_counter() - t0
                    rep_atom = pre["rep"]
                    roots = prime_roots = None
                    prime_cache: dict = {}
                    if chunk:
                        # ONE replicate inside the trunk's tape, to build the deep invariant
                        # (the DiT per-block pair bias) on the trunk's side of the cut. Its
                        # rebuild costs 3.75 s of forward per chunk and is differentiated
                        # once per chunk on the way back; primed here it is paid once. This
                        # replicate is one of the 48 -- its backward is deferred into the
                        # final walk rather than skipped.
                        t0 = time.perf_counter()
                        prime_roots = diffusion_chunk(sampler, pre, range(1),
                                                      d_out, cache=prime_cache)
                        ttnn.synchronize_device(dev)
                        row["prime_replicate_s"] = round(time.perf_counter() - t0, 3)
                        diff_s += row["prime_replicate_s"]
                        print(f"  [prime] 1 replicate inside the trunk tape "
                              f"{row['prime_replicate_s']:.2f}s", flush=True)
                        dump()
                    if not chunk:
                        t0 = time.perf_counter()
                        roots = diffusion_chunk(sampler, pre, range(a.samples), d_out,
                                                batch=a.dit_batch)
                        diff_s += time.perf_counter() - t0
                        row["dram_after_diffusion"] = _dram(dev)
                        peak_dram = row["dram_after_diffusion"]
                    print("  [tape] leaving the tape context", flush=True)

                print("  [tape] left the tape context", flush=True)
                row["diffusion"] = d_out
                row["losses"] = l_out
                loss_s = seed_s = bwd_s = 0.0
                roll = labels = logits = None

                if chunk:
                    # --- 2-4 chunked. The diffusion loss is a sum over independent noised
                    # structures GIVEN the trunk output, so the derivative of the sum is the
                    # sum of the per-chunk derivatives -- exact, not an approximation. Each
                    # chunk forwards, scores its own replicates, backs up into the
                    # accumulating weight gradients and into the CUT leaves, and drops its
                    # tape. The trunk is entered once, after the last chunk.
                    s_det, z_det = cut(s_tr), cut(pre["zij_pad"])
                    cache_pairs: list = []
                    cut_cache = {k: cut_tree(v, cache_pairs)
                                 for k, v in prime_cache.items()}
                    row["cut"] = ("si_trunk + zij_pad + %d cache entries (%d taped tensors) "
                                  "-> detached leaves" % (len(cut_cache), len(cache_pairs)))
                    chunks = []
                    t0 = time.perf_counter()
                    roll = _to_tokens(prime_roots[0], rep_atom)
                    labels, logits = step_fixture(roll, rep_atom, rng_loss, l_out)
                    prime_seeds = replicate_losses(prime_roots, rep_atom, labels, weights,
                                                   l_out)
                    prime_cot = _cotangents(prime_roots, prime_seeds, dev)
                    loss_s += time.perf_counter() - t0
                    for ci in range(1, a.samples, chunk):
                        part = range(ci, min(ci + chunk, a.samples))
                        c = {"first": ci, "n": len(part), "dit_batch": a.dit_batch}
                        t0 = time.perf_counter()
                        with ag.tape():
                            c["exact_in_chunk_tape"] = [ag.exact_softmax_installed(),
                                                        ag.exact_layer_norm_installed()]
                            roots = diffusion_chunk(sampler, pre, part, d_out,
                                                    si_trunk=s_det, zij_pad=z_det,
                                                    cache=cut_cache, batch=a.dit_batch)
                        ttnn.synchronize_device(dev)
                        c["diffusion_s"] = round(time.perf_counter() - t0, 3)
                        diff_s += c["diffusion_s"]
                        c["dram_after_diffusion"] = _dram(dev)
                        peak_dram = max(peak_dram, c["dram_after_diffusion"])

                        t0 = time.perf_counter()
                        if labels is None:
                            # The rollout the head terms read: the first replicate's own
                            # structure, downloaded and detached, which is the shape upstream
                            # feeds its confidence heads.
                            roll = _to_tokens(roots[0], rep_atom)
                            labels, logits = step_fixture(roll, rep_atom, rng_loss, l_out)
                        seeds = replicate_losses(roots, rep_atom, labels, weights, l_out)
                        c["losses_s"] = round(time.perf_counter() - t0, 3)
                        loss_s += c["losses_s"]

                        t0 = time.perf_counter()
                        cot = _cotangents(roots, seeds, dev)
                        c["seed_upload_s"] = round(time.perf_counter() - t0, 3)
                        seed_s += c["seed_upload_s"]

                        t0 = time.perf_counter()
                        c["tape_nodes"] = len(ag._reverse_topo(list(roots)))
                        ag.backward(list(roots), cot)
                        ttnn.synchronize_device(dev)
                        c["backward_s"] = round(time.perf_counter() - t0, 3)
                        bwd_s += c["backward_s"]
                        del roots, cot, seeds
                        roots = None
                        gc.collect()
                        c["dram_after_backward"] = _dram(dev)
                        c["mem_available_gib"] = _mem_available_gib()
                        chunks.append(c)
                        print(f"  [chunk {len(chunks)}] {c['n']} replicates  "
                              f"fwd {c['diffusion_s']:.2f}s  loss {c['losses_s']:.2f}s  "
                              f"bwd {c['backward_s']:.2f}s  {c['tape_nodes']} nodes  "
                              f"dram {c['dram_after_diffusion'] / 1e9:.2f} GB  "
                              f"memavail {c['mem_available_gib']} GiB", flush=True)
                        row["chunks"] = chunks
                        dump()
                    row["chunk_backward_s"] = round(bwd_s, 3)
                    row["chunk_backward_entries"] = len(chunks)

                    t0 = time.perf_counter()
                    step_losses(roll, labels, logits, weights, l_out)
                    loss_s += time.perf_counter() - t0

                    # THE trunk backward, once. `chunk_backward_entries` above is how many
                    # times the REPLICATE subgraph was entered; this is how many times the
                    # trunk was, and it is the number that decides whether chunking paid.
                    t0 = time.perf_counter()
                    troots = [s_tr, pre["zij_pad"]]
                    tcot = [s_det.grad, z_det.grad]
                    row["cut_cotangent_present"] = [g is not None for g in tcot]
                    seeded_cache = 0
                    for orig, leaf in cache_pairs:
                        if leaf.grad is not None:
                            troots.append(orig)
                            tcot.append(leaf.grad)
                            seeded_cache += 1
                    row["cache_leaves_seeded"] = f"{seeded_cache} of {len(cache_pairs)}"
                    # The primed replicate rides the SAME walk, so it costs no extra trunk
                    # entry: `backward` takes many roots and fires every node once.
                    troots.extend(prime_roots)
                    tcot.extend(prime_cot)
                    row["trunk_tape_nodes"] = len(ag._reverse_topo(troots))
                    ag.backward(troots, tcot)
                    ttnn.synchronize_device(dev)
                    row["trunk_backward_s"] = round(time.perf_counter() - t0, 3)
                    row["trunk_backward_entries"] = 1
                    print(f"  [trunk bwd] {row['trunk_backward_s']:.2f}s over "
                          f"{row['trunk_tape_nodes']} nodes", flush=True)
                    dump()
                    bwd_s += row["trunk_backward_s"]
                    row["tape_nodes"] = (row["trunk_tape_nodes"]
                                         + sum(c["tape_nodes"] for c in chunks))
                    row["roots_taped"] = a.samples
                else:
                    # --- 3. loss heads ----------------------------------------------------
                    t0 = time.perf_counter()
                    if a.loss_shape == "harness":
                        seeds = host_losses(roots, rep_atom, weights, rng_loss, l_out)
                    else:
                        roll = _to_tokens(roots[0], rep_atom)
                        labels, logits = step_fixture(roll, rep_atom, rng_loss, l_out)
                        seeds = replicate_losses(roots, rep_atom, labels, weights, l_out)
                        step_losses(roll, labels, logits, weights, l_out)
                    loss_s = time.perf_counter() - t0

                    # --- 4. backward ------------------------------------------------------
                    row["roots_taped"] = sum(1 for r in roots if isinstance(r, ag.Tensor))
                    if row["roots_taped"] and not a.no_tape:
                        t0 = time.perf_counter()
                        cot = _cotangents(roots, seeds, dev)
                        seed_s = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        row["tape_nodes"] = len(ag._reverse_topo(list(roots)))
                        ag.backward(list(roots), cot)
                        ttnn.synchronize_device(dev)
                        bwd_s = time.perf_counter() - t0
                        row["trunk_backward_entries"] = 1
                    else:
                        row["tape_nodes"] = 0
                        row["backward_note"] = ("UNTAPED ARM -- no backward exists to run. "
                                                "Not a zero cost and not a failure")
                row["diffusion_s"] = round(diff_s, 3)
                row["losses_s"] = round(loss_s, 3)
                row["seed_upload_s"] = round(seed_s, 3)
                row["backward_s"] = round(bwd_s, 3)
                row["dram_peak"] = peak_dram
                row["mem_available_gib"] = _mem_available_gib()
                got = sum(1 for t in params.values() if getattr(t, "grad", None) is not None)
                row["params_with_grad"] = f"{got} of {len(params)}"
                row["backward_valid"] = bool(got) or a.no_tape
                row["dram_after_backward"] = _dram(dev)

                if a.grad_ab:
                    grad_ab = grad_snapshot(params, grad_ab, row,
                                            a.out.parent / f"gradbank_{a.out.stem}")
                    out["grad_ab"] = grad_ab.get("report", {"arm": "unchunked banked"})
                    dump()

                # --- 5. optimizer ---------------------------------------------------------
                if opt is not None and got:
                    t0 = time.perf_counter()
                    upd = opt.step()
                    # Put the optimizer new weights back where the walk found them, which
                    # is the line tt_bio/train/recipes.py:185 runs after its own opt.step().
                    # Without it the next forward reads the pre-step handle and tapes
                    # nothing. moved is published per rep so that is visible rather than
                    # inferred from whether the reach counter happened to hold up.
                    row["rebind_moved"] = params.rebind()
                    # Does the MODEL now hold handles the tape knows as leaves? Both halves
                    # have to hold: the value setter re-keys _PARAMS (autograd.py:248) and
                    # rebind writes the new handle into the slot the walk found. Read here,
                    # one rep before the next backward, so a dead rep shows up at ten
                    # minutes instead of at twenty.
                    live = 0
                    for _n, (_o, _k) in params.slots.items():
                        cur = _o[_k] if isinstance(_o, (dict, list)) else getattr(_o, _k)
                        if ag._PARAMS.get(id(cur)) is params[_n]:
                            live += 1
                    row["leaves_live_after_rebind"] = f"{live} of {len(params.slots)}"
                    ttnn.synchronize_device(dev)
                    row["optimizer_s"] = round(time.perf_counter() - t0, 3)
                    row["optimizer_updated"] = len(upd) if hasattr(upd, "__len__") else None
                else:
                    row["optimizer_s"] = 0.0
                    row["optimizer_note"] = ("no gradient reached a declared weight, so the "
                                             "optimizer had nothing to step" if opt else
                                             "--no-optimizer")
                ag.release_pins()

                parts = ("trunk_s", "diffusion_s", "losses_s", "seed_upload_s",
                         "backward_s", "optimizer_s")
                row["step_s"] = round(sum(row[p] for p in parts), 3)
                row["step_wall_s"] = round(time.perf_counter() - wall0, 3)
                row["unaccounted_s"] = round(row["step_wall_s"] - row["step_s"], 3)
                row["exact_softmax_delta"] = {k: ag.EXACT_SOFTMAX_STATS[k] - ex0[0][k]
                                              for k in ag.EXACT_SOFTMAX_STATS}
                row["exact_layer_norm_delta"] = {k: ag.EXACT_LAYER_NORM_STATS[k] - ex0[1][k]
                                                 for k in ag.EXACT_LAYER_NORM_STATS}
                if a.no_tape:
                    row["step_s_UNTAPED"] = row.pop("step_s")
                    row["step_s"] = None
                row["renorm_applied"] = (ag.SOFTMAX_BW_RENORM_STATS["applied"]
                                         - rs0["applied"])
                row["renorm_declined"] = (ag.SOFTMAX_BW_RENORM_STATS["declined"]
                                          - rs0["declined"])
                print(f"[rep {rep}] trunk {row['trunk_s']:.2f}s  "
                      f"diffusion {row['diffusion_s']:.2f}s  "
                      f"losses {row['losses_s']:.2f}s  "
                      f"backward {row['backward_s']:.2f}s  "
                      f"optimizer {row['optimizer_s']:.2f}s  = "
                      f"{(row['step_s'] or row.get('step_s_UNTAPED')):.2f}s "
                      f"({row['params_with_grad']} weights, {row['tape_nodes']} nodes, "
                      f"renorm {'ON' if row['renorm_flag'] else 'OFF'} "
                      f"{row['renorm_applied']}a/{row['renorm_declined']}d, "
                      f"leaves {row.get('leaves_live_after_rebind')})",
                      flush=True)
                print(f"[rep {rep}] WALL {row['step_wall_s']:.2f}s  phases "
                      f"{row['step_s'] or row.get('step_s_UNTAPED'):.2f}s  unaccounted "
                      f"{row['unaccounted_s']:.2f}s  exact softmax "
                      f"{row['exact_softmax_delta']} layer_norm "
                      f"{row['exact_layer_norm_delta']}", flush=True)
                dump()

            # THE PROOF, differenced across the reps. An arm that silently kept the
            # instrument reads as a catastrophic regression and one that silently dropped it
            # reads as a miracle; these counters are what tells the two apart from the
            # artifact alone.
            out["exact"]["softmax_stats_after"] = dict(ag.EXACT_SOFTMAX_STATS)
            out["exact"]["layer_norm_stats_after"] = dict(ag.EXACT_LAYER_NORM_STATS)
            for half in ("softmax", "layer_norm"):
                before, after = (out["exact"][f"{half}_stats_before"],
                                 out["exact"][f"{half}_stats_after"])
                out["exact"][f"{half}_delta"] = {k: after[k] - before[k] for k in after}
            out["exact"]["counters_all_zero"] = not any(
                v for half in ("softmax", "layer_norm")
                for v in out["exact"][f"{half}_delta"].values())
            out["exact"]["installed_in_any_tape"] = any(
                v for r in reps
                for v in list(r.get("exact_installed_in_trunk_tape", {}).values())
                + [x for c in r.get("chunks", []) for x in c.get("exact_in_chunk_tape", [])])
            out["exact"]["off_proven"] = bool(a.no_exact
                                              and out["exact"]["counters_all_zero"]
                                              and not out["exact"]["installed_in_any_tape"])
            out["renorm"]["stats_after"] = dict(ag.SOFTMAX_BW_RENORM_STATS)
            out["renorm"]["applied_during_reps"] = (
                out["renorm"]["stats_after"]["applied"]
                - out["renorm"]["stats_before"]["applied"])
            out["renorm"]["declined_during_reps"] = (
                out["renorm"]["stats_after"]["declined"]
                - out["renorm"]["stats_before"]["declined"])
            key = "step_s_UNTAPED" if a.no_tape else "step_s"
            vals = [r[key] for r in reps if r.get(key) is not None]
            if vals:
                out["cold_s"] = vals[0]
                steady = vals[1:]
                out["steady"] = {
                    "n": len(steady),
                    "values_s": steady,
                    "median_s": round(statistics.median(steady), 3) if steady else None,
                    "min_s": min(steady) if steady else None,
                    "max_s": max(steady) if steady else None,
                    "spread_s": round(max(steady) - min(steady), 3) if steady else None,
                }
            wall = [r["step_wall_s"] for r in reps if r.get("step_wall_s") is not None]
            if wall:
                out["cold_wall_s"] = wall[0]
                out["steady_wall"] = {
                    "n": len(wall[1:]), "values_s": wall[1:],
                    "median_s": round(statistics.median(wall[1:]), 3) if wall[1:] else None,
                    "unaccounted_s": [r["unaccounted_s"] for r in reps
                                      if r.get("unaccounted_s") is not None]}
        except Exception:
            out["error"] = traceback.format_exc()
            print(out["error"], flush=True)
        out["env"]["aiclk_during"] = clk.summary()
        out["env"]["aiclk_line"] = clk.line(0)
        out["env"]["loadavg_end"] = os.getloadavg()
    dump()
    print(out["env"]["aiclk_line"], flush=True)
    print("loadavg %s -> %s" % (out["env"]["loadavg_start"], out["env"]["loadavg_end"]),
          flush=True)
    print("WROTE", a.out, flush=True)
    return 1 if "error" in out else 0


if __name__ == "__main__":
    sys.exit(main())
