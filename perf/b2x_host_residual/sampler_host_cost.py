#!/usr/bin/env python3
"""What the diffusion sampler's between-step host arithmetic costs, per step, CPU only.

`AtomDiffusion.sample` runs 200 iterations; each one calls the device denoiser once and then does
its own torch arithmetic on (1, M, 3) coordinate tensors. The denoiser is a `TorchWrapper`, so it
hands a torch tensor back and the device has finished before the next host line runs: the loop is
strictly serial, host and device cannot overlap inside it, and the host part lands in the 3.212 s
zero-device residual.

Replays the exact per-step sequence out of `sample` (steering off, which is what the fixture runs:
the contact-guidance branch was already shown free at d0183cc0) against the shipped
`weighted_rigid_align`. M is a parameter, not a guess.
"""
import argparse
import os
import statistics as st
import sys
import time
from math import sqrt

sys.path.insert(0, "/home/moritz/tt-bio")
import torch

torch.set_grad_enabled(False)
from tt_bio.boltz2 import compute_random_augmentation, weighted_rigid_align  # noqa: E402


def one_step(atom_coords, atom_coords_denoised, atom_mask, sigma_tm, sigma_t, gamma,
             noise_scale, step_scale, stages):
    """The host half of one sampling-loop iteration, timed stage by stage."""
    def t(tag, fn):
        t0 = time.perf_counter()
        r = fn()
        stages[tag] = stages.get(tag, 0.0) + time.perf_counter() - t0
        return r

    m = atom_coords.shape[0]
    random_R, random_tr = t("randaug", lambda: compute_random_augmentation(
        m, device=atom_coords.device, dtype=atom_coords.dtype))
    atom_coords = t("center", lambda: atom_coords - atom_coords.mean(dim=-2, keepdims=True))
    atom_coords = t("rotate", lambda: torch.einsum(
        "bmd,bds->bms", atom_coords, random_R) + random_tr)
    atom_coords_denoised = t("center_den", lambda:
                             atom_coords_denoised - atom_coords_denoised.mean(dim=-2,
                                                                              keepdims=True))
    atom_coords_denoised = t("rotate_den", lambda: torch.einsum(
        "bmd,bds->bms", atom_coords_denoised, random_R) + random_tr)

    t_hat = sigma_tm * (1 + gamma)
    noise_var = noise_scale ** 2 * (t_hat ** 2 - sigma_tm ** 2)
    eps = t("randn_eps", lambda: sqrt(noise_var) * torch.randn(
        atom_coords.shape, device=atom_coords.device))
    atom_coords_noisy = t("add_eps", lambda: atom_coords + eps)
    # the device denoiser would run here
    atom_coords_noisy = t("align", lambda: weighted_rigid_align(
        atom_coords_noisy.float(), atom_coords_denoised.float(),
        atom_mask.float(), atom_mask.float()))
    atom_coords_noisy = t("align_cast", lambda: atom_coords_noisy.to(atom_coords_denoised))
    dos = t("step_arith", lambda: (atom_coords_noisy - atom_coords_denoised) / t_hat)
    atom_coords = t("step_arith2", lambda:
                    atom_coords_noisy + step_scale * (sigma_t - t_hat) * dos)
    return atom_coords, atom_coords_denoised


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    print(f"host {os.uname().nodename}  torch {torch.__version__}  "
          f"threads {torch.get_num_threads()}  steps {a.steps}")

    for m in a.atoms:
        torch.manual_seed(0)
        ac = torch.randn(1, m, 3)
        acd = torch.randn(1, m, 3)
        mask = torch.ones(1, m)
        stages = {}
        # warm
        one_step(ac, acd, mask, 40.0, 38.0, 0.8, 1.003, 1.5, {})
        t0 = time.perf_counter()
        for _ in range(a.steps):
            ac, acd = one_step(ac, acd, mask, 40.0, 38.0, 0.8, 1.003, 1.5, stages)
        tot = time.perf_counter() - t0
        print(f"\n--- M = {m} atoms, {a.steps} steps: {tot:.3f} s total, "
              f"{1e3*tot/a.steps:.3f} ms/step ---")
        for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
            print(f"    {k:14s} {1e3*v/a.steps:7.3f} ms/step   {v:6.3f} s / fold   "
                  f"{100*v/tot:5.1f} %")
        acc = sum(stages.values())
        print(f"    {'(sum)':14s} {1e3*acc/a.steps:7.3f} ms/step   {acc:6.3f} s / fold   "
              f"{100*acc/tot:5.1f} %  (rest is the timing calls themselves)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
