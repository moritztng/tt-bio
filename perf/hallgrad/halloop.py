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
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ttuser/.coworker/wt/hallgrad-build")
from perf.hallgrad.e2e_distogram import VOCAB, make_weights, tt_chain  # noqa: E402


def clocks_thread(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=25)
            for dev in json.loads(r.stdout).get("device_info", []):
                c = dev.get("telemetry", {}).get("aiclk")
                if c is not None:
                    out.append(int(c))
        except Exception:
            pass
        time.sleep(5.0)


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
    ap.add_argument("--checkpoint", action="store_true",
                    help="recompute the pairformer block inside its backward")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    stop, clocks = threading.Event(), []
    threading.Thread(target=clocks_thread, args=(stop, clocks), daemon=True).start()

    n, bins = args.n, args.bins
    cfg = dict(heads=args.heads, head_dim=args.head_dim, hidden=args.c_z,
               chunk=args.chunk, checkpoint=args.checkpoint)
    rng = np.random.default_rng(args.seed)
    Wnp = make_weights(rng, args.c_s, args.c_z, args.heads, args.head_dim, args.c_z, bins)
    Wtt = {k: ag.Tensor(ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
        for k, v in Wnp.items()}

    # long-range mask, the pairs a contact objective is allowed to count
    idx = np.arange(n)
    mask = (np.abs(idx[:, None] - idx[None, :]) >= args.min_sep)
    M = int(mask.sum())
    mask_t = torch.from_numpy(mask).to(torch.float64)
    print(f"# N={n} c_z={args.c_z} heads={args.heads} bins={bins} chunk={args.chunk}")
    print(f"# objective: long-range contact, {M} pairs at |i-j| >= {args.min_sep}, "
          f"contact = bins 0..{args.contact_bins - 1} of {bins}")
    print(f"# {args.steps} gradient steps (ColabDesign design_3stage default), Adam lr={args.lr}")
    print(f"# design variable: float32 on host, cast to bf16 for the forward only")
    sys.stdout.flush()

    # fp32 design variable + fp32 Adam state, on host
    logits = torch.from_numpy(rng.standard_normal((n, VOCAB)) * 0.1).to(torch.float32)
    m = torch.zeros_like(logits)
    vv = torch.zeros_like(logits)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    traj, step_times = [], []

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        lt = ag.Tensor(ttnn.from_torch(logits.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=device),
                       requires_grad=True)
        d = tt_chain(ag, ttnn, lt, Wtt, cfg)
        z64 = ttnn.to_torch(d.value).to(torch.float64)
        p = torch.softmax(z64, dim=-1)
        pc = p[..., :args.contact_bins].sum(-1).clamp_min(1e-12)
        loss = float(-(mask_t * torch.log(pc)).sum() / M)
        # analytic gradient of the contact loss w.r.t. the distogram logits
        inC = torch.zeros(bins, dtype=torch.float64)
        inC[:args.contact_bins] = 1.0
        seed = -(p * (inC / pc.unsqueeze(-1) - 1.0)) * (mask_t / M).unsqueeze(-1)
        d.backward(seed=ttnn.from_torch(seed.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=device))
        g = ttnn.to_torch(lt.grad).to(torch.float32)
        # Release this step's tape BEFORE the next one is built. Without this the loop dies
        # at step 2: `lt = ...` rebinds only after the right-hand side has evaluated, so the
        # previous step's whole graph is still live while the new forward allocates, and two
        # tapes do not fit. Measured at 256 aa on a 34.23 GB card.
        lt.grad = None
        lt.node = None
        d.grad = None
        d.node = None
        del lt, d, z64, p, pc, seed
        # Adam, in fp32 on host, above the bf16 step floor
        m = b1 * m + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        mh = m / (1 - b1 ** step)
        vh = vv / (1 - b2 ** step)
        logits = logits - args.lr * mh / (vh.sqrt() + eps_adam)
        step_times.append(time.perf_counter() - t0)
        traj.append(loss)
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"step {step:>4}  loss {loss:.6f}  |g| {float(g.norm()):.4e}  "
                  f"{step_times[-1]:.3f} s")
            sys.stdout.flush()

    med = statistics.median(step_times)
    warm = statistics.median(step_times[1:]) if len(step_times) > 1 else med
    total = sum(step_times)
    print(f"\n# loss {traj[0]:.6f} -> {traj[-1]:.6f} "
          f"(best {min(traj):.6f} at step {int(np.argmin(traj)) + 1})")
    print(f"# step time: median {med:.3f} s, warm median {warm:.3f} s, "
          f"total {total / 60:.1f} min for {args.steps} steps")
    print(f"# one design = {total / 3600:.3f} h -> {3600 / total:.2f} designs/hour "
          f"on this one-block chain at {n} aa")
    if clocks:
        print(f"# AICLK during: min {min(clocks)} max {max(clocks)} "
              f"median {int(statistics.median(clocks))} MHz over {len(clocks)} samples")
    else:
        print("# AICLK: NO SAMPLES -- timings unusable")
    stop.set()
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n": n, "steps": args.steps, "lr": args.lr, "traj": traj,
                       "step_times": step_times, "clocks": clocks}, f)
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
