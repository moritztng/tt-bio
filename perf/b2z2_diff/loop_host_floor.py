#!/usr/bin/env python3
"""What the Boltz-2 sampler loop's host body costs when nothing else is on the core.

Pass 1 measured 2.418 s/fold (32.2 % of the traced sampler stage) of host torch with the device
idle, and the largest item, `weighted_rigid_align`, cost 5.44 ms/call inside the fold. This runs
the same calls at the same shapes with no device in the process, so the difference between the two
is the cost of the environment rather than the arithmetic. CPU only: it opens no card and can run
while every chip is busy.

The regions mirror `AtomDiffusion.sample`'s loop body one for one, so the total is directly
comparable to the `sample` bracket in `loop_host_split.py`.
"""
from __future__ import annotations

import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def bench(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, default=4480)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--threads", default="0,1,2,4,8", help="torch intra-op thread counts to sweep")
    ap.add_argument("--out", type=Path, default=ROOT / "perf" / "b2z2_diff" / "loop_host_floor.json")
    a = ap.parse_args()

    import torch
    from tt_bio.boltz2 import compute_random_augmentation, weighted_rigid_align
    torch.set_grad_enabled(False)

    N = a.atoms
    default_threads = torch.get_num_threads()
    coords = torch.randn(1, N, 3)
    denoised = torch.randn(1, N, 3)
    mask = torch.ones(1, N)
    R, tr = compute_random_augmentation(1)

    regions = {
        # the loop body, in the order sample() runs it
        "compute_random_augmentation": lambda: compute_random_augmentation(1),
        "center": lambda: coords - coords.mean(dim=-2, keepdims=True),
        "rotate_einsum": lambda: torch.einsum("bmd,bds->bms", coords, R) + tr,
        "noise_draw": lambda: torch.randn(1, N, 3),
        "precondition_in": lambda: 0.123 * coords,
        "precondition_out": lambda: 0.4 * coords + 0.6 * denoised,
        "weighted_rigid_align": lambda: weighted_rigid_align(
            coords.float(), denoised.float(), mask.float(), mask.float()),
        "step_update": lambda: coords + 1.5 * (0.3 - 0.4) * ((coords - denoised) / 0.4),
    }
    # the two type conversions the adapter does per step on the same coords
    regions["to_bfloat16_and_back"] = lambda: coords.to(torch.bfloat16).to(torch.float32)

    res = {"atoms": N, "reps": a.reps, "default_threads": default_threads,
           "torch": torch.__version__, "loadavg": open("/proc/loadavg").read().split()[:3],
           "by_threads": {}}
    for spec in a.threads.split(","):
        th = int(spec)
        torch.set_num_threads(th or default_threads)
        row = {k: round(bench(fn, a.reps), 4) for k, fn in regions.items()}
        # the loop body proper: everything except the dtype round trip, which the adapter owns
        row["loop_body_total_ms"] = round(
            sum(v for k, v in row.items() if k != "to_bfloat16_and_back"), 4)
        row["per_fold_s_at_200_steps"] = round(row["loop_body_total_ms"] * 200 / 1e3, 3)
        res["by_threads"][str(torch.get_num_threads())] = row
        print(f"threads={torch.get_num_threads()}", json.dumps(row), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
