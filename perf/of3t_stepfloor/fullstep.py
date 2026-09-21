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


def declare_all(trunk, sampler, out):
    """Every device weight of the trunk AND the diffusion half, as one tape-leaf set.

    Deduped by tensor IDENTITY, not by name: `tape()` keys a trainable weight on the raw
    handle, so declaring one tensor under two paths would put the same leaf in twice and
    the optimizer would step it twice.
    """
    from tt_bio import autograd as ag
    found = {}
    stats = {}
    for label, root, pre in (("trunk", trunk, "trunk."),
                             ("diffusion", sampler, "diffusion.")):
        f, st = S.walk_weights(root, prefix=pre)
        stats[label] = {"found": len(f), "walk_depth": st["max_depth"],
                        "truncated": st["truncated"]}
        found.update(f)
    by_id, params = {}, {}
    for n, (_o, _k, t) in sorted(found.items()):
        if id(t) in by_id:
            continue
        by_id[id(t)] = n
        params[n] = ag.parameter(t)
    out["params"] = {
        "declared": len(params),
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


def diffusion_train(sampler, sargs, s_trunk, z_trunk, n_samples, rng, out):
    """`_train_diffusion`'s shape: N independently noised structures, N distinct sigmas.

    Their training differentiates `no_samples` noised structures (48 in initial_training, 32
    in finetune_1/2) rather than walking a rollout, so the samples are INDEPENDENT: there is
    no EDM update chaining one to the next and no host download between them. Every denoised
    structure stays on the card as a tape root.

    The pair branch is a loop invariant on their side as it is on ours, so `dc.pair` runs
    once and the per-sample work is `dc.single` plus the module -- which is what makes the
    per-sample rate a rate.
    """
    import numpy as np
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
     step_scale) = sargs[:35]

    # THE COUPLING. Arguments 1 and 3 are the trunk's own output; the taped pair replaces the
    # captured one so the backward runs from the denoised structure into the pairformer.
    si_trunk_dev = s_trunk if s_trunk is not None else si_trunk_cap
    zij_trunk_dev = z_trunk if z_trunk is not None else zij_trunk_cap
    out["coupled_to_trunk"] = s_trunk is not None and z_trunk is not None

    dc, dm = sampler.dc, sampler.dm
    dt = sampler._act_dtype
    atom_mask_host = ttnn.to_torch(atom_mask_col_na_dev).float().reshape(n_atom)
    xl_host = ttnn.to_torch(xl_init_dev).float().reshape(n_atom, 3)

    zij_dev = dc.pair(zij_trunk_dev, relpos_dev, pair_mask_dev)
    zij_pad = pad_dim(zij_dev, dt, n_token, n_tok_pad, dims=2)
    inv_cache: dict = {}

    # N DISTINCT noise levels, drawn from the captured schedule without replacement.
    pool = sorted({float(x) for x in t_list}, reverse=True)
    idx = rng.choice(len(pool), size=min(n_samples, len(pool)), replace=False)
    sigmas = [pool[int(i)] for i in idx]
    while len(sigmas) < n_samples:                 # schedule shorter than the sample count
        sigmas.append(pool[int(rng.integers(len(pool)))])
    out["sigmas"] = [round(s, 4) for s in sigmas]
    out["distinct_sigmas"] = len(set(sigmas))
    out["n_samples"] = n_samples

    roots, per_sample = [], []
    for k, t in enumerate(sigmas):
        t0 = time.perf_counter()
        noise = torch.from_numpy(
            rng.standard_normal((n_atom, 3)).astype("float32")) * t
        xl_noisy = xl_host + noise
        n_emb = fourier_noise_emb(t, sampler.sigma_data, sampler.fourier_w, sampler.fourier_b)
        si_dev = dc.single(si_trunk_dev, si_input_dev,
                           sampler._to_dev(n_emb.reshape(1, 1, 256)), tok_mask_dev)
        si_pad = pad_dim(si_dev, dt, n_token, n_tok_pad)
        rl_noisy = xl_noisy * atom_mask_host[:, None] / math.sqrt(t * t + sampler.sigma_data ** 2)
        rl_noisy_dev = sampler._to_dev(sampler._pad_atoms_host(rl_noisy, n_atom, NP))
        xl_noisy_dev = sampler._to_dev((xl_noisy * atom_mask_host[:, None]).unsqueeze(0))
        xl_denoised_dev = dm(
            si_trunk_dev, si_pad, zij_pad, cl0_dev, plm0_dev, rl_noisy_dev, xl_noisy_dev,
            atom_mask_col_dev, atom_mask_col_na_dev, atom_to_token_idx_tt,
            npe_flat_idx_tt, npe_zij_mask, enc_key_block_idxs_tt, enc_valid_mask,
            enc_mask_bias, enc_pair_mask, atom_to_token_mean_tt,
            token_mask_pad_tt, tok_mask_col_pad_tt,
            n_atom, NP, nb, n_token, n_tok_pad, t, sampler.sigma_data, cache=inv_cache)
        ttnn.synchronize_device(dm.device if hasattr(dm, "device") else dc.device)
        roots.append(xl_denoised_dev)
        per_sample.append(round(time.perf_counter() - t0, 3))
        print(f"  [diffusion] sample {k} sigma {t:8.3f}  {per_sample[-1]:7.3f}s", flush=True)
    out["per_sample_s"] = per_sample
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
    return roots, (zij_pad, inv_cache), rep


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=4,
                    help="trunk cycles to PIN; their training draws it from U{1..4} per step")
    ap.add_argument("--samples", type=int, default=4,
                    help="diffusion samples differentiated per step; theirs is 48 in "
                         "initial_training and 32 in finetune_1/2")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--no-tape", action="store_true",
                    help="run the same scope UNTAPED, for D32's ratio at step scope")
    ap.add_argument("--no-optimizer", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

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
                   "taped": not a.no_tape}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        try:
            import numpy as np
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio.tenstorrent import get_device
            from tt_bio.train.losses import of3_loss_weights
            from tt_bio.train.optim import AdamW

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

            reps = []
            for rep in range(a.reps):
                rng = np.random.default_rng(SEED + rep)
                row = {"rep": rep, "cold": rep == 0}
                for t in params.values():
                    t.grad = None
                ttnn.synchronize_device(dev)

                # --- 1. trunk -------------------------------------------------------------
                t0 = time.perf_counter()
                if a.cycles > 1:
                    trunk_forward(trunk, held, a.cycles - 1, taped=False)
                    ttnn.synchronize_device(dev)
                row["trunk_nograd_prefix_s"] = round(time.perf_counter() - t0, 3)

                ctx = ag.no_grad() if a.no_tape else ag.tape()
                d_out = {}
                with ctx:
                    t0 = time.perf_counter()
                    s_tr, z_tr = trunk_forward(trunk, held, 1, taped=not a.no_tape)
                    ttnn.synchronize_device(dev)
                    row["trunk_taped_cycle_s"] = round(time.perf_counter() - t0, 3)
                    row["trunk_s"] = round(row["trunk_nograd_prefix_s"]
                                           + row["trunk_taped_cycle_s"], 3)
                    row["dram_after_trunk"] = _dram(dev)

                    # --- 2. diffusion ------------------------------------------------------
                    t0 = time.perf_counter()
                    roots, keep, rep = diffusion_train(sampler, sargs, s_tr, z_tr,
                                                       a.samples, rng, d_out)
                    row["diffusion_s"] = round(time.perf_counter() - t0, 3)
                    row["diffusion"] = d_out
                    print("  [tape] leaving the tape context", flush=True)
                    row["dram_after_diffusion"] = _dram(dev)

                print("  [tape] left the tape context", flush=True)
                # --- 3. loss heads --------------------------------------------------------
                l_out = {}
                t0 = time.perf_counter()
                seeds = host_losses(roots, rep, weights, rng, l_out)
                row["losses_s"] = round(time.perf_counter() - t0, 3)
                row["losses"] = l_out

                # --- 4. backward ----------------------------------------------------------
                row["roots_taped"] = sum(1 for r in roots if isinstance(r, ag.Tensor))
                if row["roots_taped"] and not a.no_tape:
                    t0 = time.perf_counter()
                    cot = []
                    for r, g in zip(roots, seeds):
                        raw = ag._unwrap(r)
                        shp = tuple(int(d) for d in raw.shape)
                        import torch
                        h = torch.from_numpy(np.asarray(g, "float32")).reshape(shp) \
                            if g is not None else torch.ones(shp)
                        cot.append(ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT,
                                                   device=dev, dtype=raw.dtype))
                    row["seed_upload_s"] = round(time.perf_counter() - t0, 3)
                    t0 = time.perf_counter()
                    row["tape_nodes"] = len(ag._reverse_topo(list(roots)))
                    ag.backward(list(roots), cot)
                    ttnn.synchronize_device(dev)
                    row["backward_s"] = round(time.perf_counter() - t0, 3)
                else:
                    row["seed_upload_s"] = 0.0
                    row["tape_nodes"] = 0
                    row["backward_s"] = 0.0
                    row["backward_note"] = ("UNTAPED ARM -- no backward exists to run. Not a "
                                            "zero cost and not a failure")
                got = sum(1 for t in params.values() if getattr(t, "grad", None) is not None)
                row["params_with_grad"] = f"{got} of {len(params)}"
                row["backward_valid"] = bool(got) or a.no_tape
                row["dram_after_backward"] = _dram(dev)

                # --- 5. optimizer ---------------------------------------------------------
                if opt is not None and got:
                    t0 = time.perf_counter()
                    upd = opt.step()
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
                if a.no_tape:
                    row["step_s_UNTAPED"] = row.pop("step_s")
                    row["step_s"] = None
                reps.append(row)
                out["reps"] = reps
                print(f"[rep {rep}] trunk {row['trunk_s']:.2f}s  "
                      f"diffusion {row['diffusion_s']:.2f}s  "
                      f"losses {row['losses_s']:.2f}s  "
                      f"backward {row['backward_s']:.2f}s  "
                      f"optimizer {row['optimizer_s']:.2f}s  = "
                      f"{(row['step_s'] or row.get('step_s_UNTAPED')):.2f}s "
                      f"({row['params_with_grad']} weights, {row['tape_nodes']} nodes)",
                      flush=True)
                dump()

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
