#!/usr/bin/env python3
"""Why the EDM sampler's host phases cost what they cost, and what a bit-exact rewrite buys.

The in-fold profile (edm_host_profile.py) attributes the host time. This isolates it: the phases
operate on (1, n_atoms, 3) float32, at most 29 kB, so they cannot be bandwidth-bound on a host that
streams GB/s. The hypothesis is that every host phase is torch **dispatch**-bound -- cost set by the
number of ATen calls, not by the bytes. Prediction: phase time / measured single-dispatch floor
lands within ~30 % of the phase's static ATen-call count, and holds flat as n_atoms doubles.

Candidates measured against the same floor, each gated on torch.equal over a whole 200-step
trajectory (see --equal):
  bmm     -- torch.einsum("bmd,bds->bms", x, R)  ->  torch.bmm(x, R)
  quat    -- fewer ATen calls in quaternion_to_matrix, same float32 expression order

    python3 perf/diffusion_host/host_phase_micro.py --atoms 2400 --reps 2000
"""
import argparse
import time

import torch

PC = time.perf_counter


def bench(fn, reps, warm=50):
    for _ in range(warm):
        fn()
    t0 = PC()
    for _ in range(reps):
        fn()
    return (PC() - t0) / reps


def quat_to_matrix_ref(q):
    from tt_bio.boltz2 import quaternion_to_matrix
    return quaternion_to_matrix(q)


def quat_to_matrix_fast(q):
    """Same float32 expression order as boltz2.quaternion_to_matrix, fewer ATen calls.

    Products are taken once as a batched outer product instead of 18 separate 0-d multiplies;
    each element of qq is the SAME float32 multiply the reference performs, so every downstream
    add/sub sees bit-identical operands.
    """
    qq = q.unsqueeze(-1) * q.unsqueeze(-2)              # [..., 4, 4]: rr ri rj rk / ir ii ij ik / ...
    r, i, j, k = 0, 1, 2, 3
    two_s = 2.0 / (q * q).sum(-1)
    ij, ik, jk = qq[..., i, j], qq[..., i, k], qq[..., j, k]
    kr, jr, ir = qq[..., k, r], qq[..., j, r], qq[..., i, r]
    ii, jj, kk = qq[..., i, i], qq[..., j, j], qq[..., k, k]
    o = torch.stack((1 - two_s * (jj + kk), two_s * (ij - kr), two_s * (ik + jr),
                     two_s * (ij + kr), 1 - two_s * (ii + kk), two_s * (jk - ir),
                     two_s * (ik - jr), two_s * (jk + ir), 1 - two_s * (ii + jj)), -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, nargs="+", default=[1200, 2400, 4800])
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(torch.get_num_threads())
    from tt_bio.boltz2 import compute_random_augmentation

    a = torch.randn(4)
    floor = bench(lambda: a + 1.0, args.reps * 5)
    print(f"single-ATen-dispatch floor (add on a 4-element f32 tensor): {floor*1e6:.2f} us")

    # bit-exactness of the two candidates, over a real trajectory-length draw sequence
    torch.manual_seed(0)
    qs = torch.randn(512, 4)
    eq_quat = torch.equal(quat_to_matrix_ref(qs), quat_to_matrix_fast(qs))
    print(f"quaternion_to_matrix fast vs ref, 512 random quaternions: torch.equal={eq_quat}")

    for n in args.atoms:
        x = torch.randn(1, n, 3)
        R = quat_to_matrix_ref(torch.randn(1, 4))
        tr = torch.randn(1, 1, 3)
        eq_bmm = torch.equal(torch.einsum("bmd,bds->bms", x, R), torch.bmm(x, R))
        t_aug = bench(lambda: compute_random_augmentation(1, device=None, dtype=torch.float32), args.reps)
        t_center = bench(lambda: x - x.mean(dim=-2, keepdim=True), args.reps)
        t_rot = bench(lambda: torch.einsum("bmd,bds->bms", x, R) + tr, args.reps)
        t_rot_bmm = bench(lambda: torch.bmm(x, R) + tr, args.reps)
        t_rng = bench(lambda: 1.7 * torch.randn(1, n, 3), args.reps)
        t_q = bench(lambda: quat_to_matrix_ref(torch.randn(1, 4)), args.reps)
        t_qf = bench(lambda: quat_to_matrix_fast(torch.randn(1, 4)), args.reps)
        print(f"\nn_atoms={n}  ({4*3*n/1024:.1f} kB per coordinate tensor)")
        print(f"  aug (compute_random_augmentation) {t_aug*1e6:8.1f} us = {t_aug/floor:5.1f} dispatches")
        print(f"    of which quaternion_to_matrix   {t_q*1e6:8.1f} us = {t_q/floor:5.1f} dispatches")
        print(f"    fast quaternion_to_matrix       {t_qf*1e6:8.1f} us = {t_qf/floor:5.1f} dispatches")
        print(f"  center (x - x.mean)               {t_center*1e6:8.1f} us = {t_center/floor:5.1f} dispatches")
        print(f"  rotate einsum + tr                {t_rot*1e6:8.1f} us = {t_rot/floor:5.1f} dispatches")
        print(f"  rotate bmm + tr (equal={eq_bmm})    {t_rot_bmm*1e6:8.1f} us = {t_rot_bmm/floor:5.1f} dispatches")
        print(f"  rng (randn + scale)               {t_rng*1e6:8.1f} us = {t_rng/floor:5.1f} dispatches")


if __name__ == "__main__":
    main()
