#!/usr/bin/env python3
"""Gradient-based hallucination loop on the trunk-only distogram contact objective.

The objective is the field's own trunk-only hallucination form, not a target structure:
maximise confident long-range contacts,

    L = -mean over {|i-j| >= 6} log( sum of p_b over bins below the contact threshold )

so there is nothing to fit to and the sequence is the only free variable. Its gradient with
respect to the distogram logits is analytic,

    dL/dz_b = -(1/M) * p_b * ( [b in contact bins]/P_contact - 1 )

so the loop never differentiates a loss approximation, exactly as in the Phase 3 check.

STEP COUNT IS ColabDesign's OWN DEFAULT for the gradient route: design_3stage runs
soft_iters=300 + temp_iters=100 + hard_iters=10 = 410 gradient steps. Nothing here is reduced
to make a number look better.

THE DESIGN VARIABLE LIVES IN FLOAT32 ON HOST, and that is a measured requirement, not a
preference. Phase 3 measured that a bf16 logit step below bf16's spacing at the logits' own
magnitude is silently rounded away (0.209 of an eps=0.01 step survived), so a loop that kept
its logits and Adam state in bf16 would stop moving as its effective step decayed while still
reporting a healthy gradient. Host fp32 state with a bf16 cast only for the forward keeps the
accumulation above that floor.

`--blocks K` runs K sequential pairformer units, which is what the BindCraft2 floor screen
fits `t(K) = a + b*K` over. The loop body is `run_loop`, called once per sweep point by
`blocksweep.py` inside a single device context.
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.hallgrad.e2e_distogram import (  # noqa: E402
    VOCAB, ClockTrace, make_weights, tt_chain)


def run_loop(ag, ttnn, device, *, n, c_z, c_s, heads, head_dim, bins, chunk, steps, lr,
             contact_bins, min_sep, seed, checkpoint, blocks, block_transition,
             clocks=None, verbose=True):
    """One gradient-descent loop at one (n, blocks, block_transition) point.

    Returns the per-step wall times split into forward and backward, the loss trajectory, the
    per-step reached-node count, and the window boundaries so a clock trace can be attributed
    to the steps that were actually timed.
    """
    cfg = dict(heads=heads, head_dim=head_dim, hidden=c_z, chunk=chunk,
               checkpoint=checkpoint, blocks=blocks, block_transition=block_transition)
    rng = np.random.default_rng(seed)
    Wnp = make_weights(rng, c_s, c_z, heads, head_dim, c_z, bins,
                       blocks=blocks, block_transition=block_transition)
    Wtt = {k: ag.Tensor(ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
        for k, v in Wnp.items()}

    # long-range mask, the pairs a contact objective is allowed to count
    idx = np.arange(n)
    mask = (np.abs(idx[:, None] - idx[None, :]) >= min_sep)
    M = int(mask.sum())
    mask_t = torch.from_numpy(mask).to(torch.float64)
    inC = torch.zeros(bins, dtype=torch.float64)
    inC[:contact_bins] = 1.0

    # fp32 design variable + fp32 Adam state, on host
    logits = torch.from_numpy(rng.standard_normal((n, VOCAB)) * 0.1).to(torch.float32)
    m = torch.zeros_like(logits)
    vv = torch.zeros_like(logits)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    traj, step_times, fwd_times, bwd_times, reached_all, step_starts = [], [], [], [], [], []

    t_start = time.time()
    for step in range(1, steps + 1):
        step_starts.append(time.time())
        t0 = time.perf_counter()
        lt = ag.Tensor(ttnn.from_torch(logits.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=device),
                       requires_grad=True)
        d = tt_chain(ag, ttnn, lt, Wtt, cfg)
        z64 = ttnn.to_torch(d.value).to(torch.float64)   # this synchronises
        t_fwd = time.perf_counter() - t0

        p = torch.softmax(z64, dim=-1)
        pc = p[..., :contact_bins].sum(-1).clamp_min(1e-12)
        loss = float(-(mask_t * torch.log(pc)).sum() / M)
        # analytic gradient of the contact loss w.r.t. the distogram logits
        seed_t = -(p * (inC / pc.unsqueeze(-1) - 1.0)) * (mask_t / M).unsqueeze(-1)

        # Campaign defect D126: `_PARAMS` is keyed on id() of the leaf and replacing a leaf's
        # .value unregisters it, after which backward reaches nothing while the loss still
        # falls. A falling loss is not evidence. `_reverse_topo` is side-effect free.
        reached = len(ag._reverse_topo([d]))
        if reached <= 1:
            raise RuntimeError(f"step {step}: backward reaches {reached} nodes -- the tape is "
                               f"empty, every timing below it is meaningless (D126)")

        t1 = time.perf_counter()
        d.backward(seed=ttnn.from_torch(seed_t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=device))
        if lt.grad is None:
            raise RuntimeError(f"step {step}: no gradient reached the sequence logits (D126)")
        g = ttnn.to_torch(lt.grad).to(torch.float32)     # this synchronises
        t_bwd = time.perf_counter() - t1
        if not bool(torch.isfinite(g).all()):
            raise RuntimeError(f"step {step}: logit gradient is not finite")
        gnorm = float(g.norm())
        if not gnorm > 0.0:
            raise RuntimeError(f"step {step}: logit gradient norm is {gnorm}")

        # Release this step's tape BEFORE the next one is built. Without this the loop dies
        # at step 2: `lt = ...` rebinds only after the right-hand side has evaluated, so the
        # previous step's whole graph is still live while the new forward allocates, and two
        # tapes do not fit. Measured at 256 aa on a 34.23 GB card.
        lt.grad = None
        lt.node = None
        d.grad = None
        d.node = None
        del lt, d, z64, p, pc, seed_t
        # Adam, in fp32 on host, above the bf16 step floor
        m = b1 * m + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        mh = m / (1 - b1 ** step)
        vh = vv / (1 - b2 ** step)
        logits = logits - lr * mh / (vh.sqrt() + eps_adam)

        step_times.append(time.perf_counter() - t0)
        fwd_times.append(t_fwd)
        bwd_times.append(t_bwd)
        reached_all.append(reached)
        traj.append(loss)
        if verbose and (step == 1 or step % 20 == 0 or step == steps):
            print(f"step {step:>4}  loss {loss:.6f}  |g| {gnorm:.4e}  "
                  f"{step_times[-1]:.3f} s  (fwd {t_fwd:.3f} bwd {t_bwd:.3f})  "
                  f"reached {reached}")
            sys.stdout.flush()

    t_end = time.time()
    if len(set(reached_all)) != 1:
        raise RuntimeError(f"reached-node count is not constant across steps: "
                           f"{sorted(set(reached_all))} -- the tape changed shape mid-loop")
    return {"n": n, "blocks": blocks, "block_transition": block_transition,
            "checkpoint": checkpoint, "steps": steps, "traj": traj,
            "step_times": step_times, "fwd_times": fwd_times, "bwd_times": bwd_times,
            "reached": reached_all[0], "t_start": t_start, "t_end": t_end,
            "step_starts": step_starts,
            "clock_window": clocks.window(t_start, t_end) if clocks else None}


def warm_stats(v, warm_from=5):
    """Mean/median over v[warm_from:] -- step 1 is compilation, 2..4 allocator warm-up."""
    w = v[warm_from:] if len(v) > warm_from else v[1:] or v
    return {"mean": float(statistics.mean(w)), "median": float(statistics.median(w)),
            "min": float(min(w)), "max": float(max(w)), "count": len(w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--c-z", type=int, default=256)
    ap.add_argument("--c-s", type=int, default=384)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--steps", type=int, default=410,
                    help="ColabDesign design_3stage default: 300 soft + 100 temp + 10 hard")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--contact-bins", type=int, default=20,
                    help="bins below the contact threshold, of --bins")
    ap.add_argument("--min-sep", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=1,
                    help="sequential pairformer units in the chain")
    ap.add_argument("--block-transition", action="store_true",
                    help="put the pair transition INSIDE each unit (AF2's real pair block)")
    ap.add_argument("--checkpoint", action="store_true",
                    help="recompute each pairformer unit inside its backward")
    ap.add_argument("--warm-from", type=int, default=5,
                    help="first step counted in the warm statistics")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    clocks = ClockTrace(period=1.0).start()

    print(f"# N={args.n} c_z={args.c_z} heads={args.heads} bins={args.bins} chunk={args.chunk} "
          f"blocks={args.blocks} block_transition={args.block_transition} "
          f"checkpoint={args.checkpoint}")
    print(f"# {args.steps} gradient steps, Adam lr={args.lr}")
    print(f"# design variable: float32 on host, cast to bf16 for the forward only")
    sys.stdout.flush()

    r = run_loop(ag, ttnn, device, n=args.n, c_z=args.c_z, c_s=args.c_s, heads=args.heads,
                 head_dim=args.head_dim, bins=args.bins, chunk=args.chunk, steps=args.steps,
                 lr=args.lr, contact_bins=args.contact_bins, min_sep=args.min_sep,
                 seed=args.seed, checkpoint=args.checkpoint, blocks=args.blocks,
                 block_transition=args.block_transition, clocks=clocks)
    clocks.stop()

    st, traj = r["step_times"], r["traj"]
    ws = warm_stats(st, args.warm_from)
    total = sum(st)
    print(f"\n# loss {traj[0]:.6f} -> {traj[-1]:.6f} "
          f"(best {min(traj):.6f} at step {int(np.argmin(traj)) + 1})")
    print(f"# step time: step 1 {st[0]:.3f} s (compilation), warm mean {ws['mean']:.4f} s, "
          f"warm median {ws['median']:.4f} s over steps {args.warm_from + 1}..{len(st)}")
    print(f"# total {total / 60:.2f} min for {args.steps} steps; "
          f"backward reaches {r['reached']} tape nodes every step")
    cw = r["clock_window"]
    if cw and cw.get("attributable"):
        print(f"# AICLK over the timed window: min {cw['min']} median {cw['median']} "
              f"max {cw['max']} MHz over {cw['samples']} samples")
    else:
        print("# AICLK: NO SAMPLES IN THE TIMED WINDOW -- timings unusable")
    print(f"# tt-smi device_info lengths seen: {clocks.summary()['device_counts']}")

    if args.out:
        blob = {"argv": " ".join(sys.argv), "n": args.n, "c_z": args.c_z, "c_s": args.c_s,
                "heads": args.heads, "head_dim": args.head_dim, "bins": args.bins,
                "chunk": args.chunk, "blocks": args.blocks,
                "block_transition": args.block_transition, "checkpoint": args.checkpoint,
                "steps": args.steps, "lr": args.lr, "seed": args.seed,
                "warm_from": args.warm_from, "warm": ws, "reached": r["reached"],
                "traj": traj, "step_times": st, "fwd_times": r["fwd_times"],
                "bwd_times": r["bwd_times"], "clock_window": cw,
                "clock_samples": clocks.samples, "clock_meta": clocks.summary()}
        with open(args.out, "w") as f:
            json.dump(blob, f, indent=1)
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
