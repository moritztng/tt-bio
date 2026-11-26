"""`tt_bio.diffdock.loop` against upstream's sampler arithmetic.

The reference below is the body of `utils/sampling.py::sampling` (MIT, gcorso/DiffDock) with
the data-loading, logging and visualisation stripped out -- the step constants, the noise
terms and the NaN rescue, kept in upstream's order and upstream's Python. The port precomputes
all of it into a table; these tests say the table holds the same numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tt_bio.diffdock.loop import Schedule, build_schedule, guard_scores, run_loop
from tt_bio.diffdock.conformer import modify_conformer

from test_diffdock_conformer import chain_ligand, ref_modify_conformer_batch

SIGMAS = {'tr': (0.1, 19.0), 'rot': (0.03, 1.55), 'tor': (0.0314, 3.14)}   # DiffDock-L's own
TOL = 1e-5


def beta_like_schedule(n_steps):
    """Upstream's `get_t_schedule` is a beta ppf; its shape does not matter to this test, only
    that it is decreasing and does not reach 0, so a plain linear ramp stands in for it."""
    return list(np.linspace(1.0, 1.0 / n_steps, n_steps))


def ref_step(pos, tr_score, rot_score, tor_score, t_idx, n_steps, sched, tr_z, rot_z, tor_z,
             bonds, mask_rotate, no_final_step_noise=False):
    """Upstream's per-step arithmetic, verbatim."""
    t_tr, t_rot, t_tor = sched[t_idx], sched[t_idx], sched[t_idx]
    dt_tr = sched[t_idx] - sched[t_idx + 1] if t_idx < n_steps - 1 else sched[t_idx]
    dt_rot, dt_tor = dt_tr, dt_tr
    tr_sigma = SIGMAS['tr'][0] ** (1 - t_tr) * SIGMAS['tr'][1] ** t_tr
    rot_sigma = SIGMAS['rot'][0] ** (1 - t_rot) * SIGMAS['rot'][1] ** t_rot
    tor_sigma = SIGMAS['tor'][0] ** (1 - t_tor) * SIGMAS['tor'][1] ** t_tor

    tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(SIGMAS['tr'][1] / SIGMAS['tr'][0])))
    rot_g = rot_sigma * torch.sqrt(torch.tensor(2 * np.log(SIGMAS['rot'][1] / SIGMAS['rot'][0])))
    tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(SIGMAS['tor'][1] / SIGMAS['tor'][0])))
    if no_final_step_noise and t_idx == n_steps - 1:
        tr_z, rot_z, tor_z = (torch.zeros_like(x) for x in (tr_z, rot_z, tor_z))
    tr_perturb = tr_g ** 2 * dt_tr * tr_score + tr_g * np.sqrt(dt_tr) * tr_z
    rot_perturb = rot_score * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z
    tor_perturb = tor_g ** 2 * dt_tor * tor_score + tor_g * np.sqrt(dt_tor) * tor_z
    return ref_modify_conformer_batch(pos, tr_perturb, rot_perturb, tor_perturb, bonds, mask_rotate)


def fixture(n_steps=19, seed=4):
    pos, bonds, mask, g = chain_ligand(n_atoms=20, n_poses=10, seed=seed)
    sched = beta_like_schedule(n_steps)
    scores, noise = [], []
    for _ in range(n_steps):
        scores.append((torch.randn(10, 3, generator=g) * 0.2,
                       torch.randn(10, 3, generator=g) * 0.2,
                       torch.randn(10, len(bonds), generator=g) * 0.2))
        noise.append((torch.randn(10, 3, generator=g), torch.randn(10, 3, generator=g),
                      torch.randn(10, len(bonds), generator=g)))
    return pos, bonds, mask, sched, scores, noise


def test_the_whole_19_step_loop_matches_upstreams_arithmetic():
    pos, bonds, mask, sched, scores, noise = fixture()
    table = build_schedule(sched, sched, sched, SIGMAS)
    got, n_nan, min_det = run_loop(pos, lambda p, t: scores[t], table,
                                   lambda t: noise[t], bonds, mask)
    want = pos
    for t in range(len(sched)):
        want = ref_step(want, *scores[t], t, len(sched), sched, *noise[t], bonds, mask)
    assert n_nan.item() == 0
    assert (got - want).abs().max().item() < 1e-3, "19 steps of accumulated fp32 rounding"
    assert min_det is not None, "the determinant diagnostic comes back from the loop"


def test_no_final_step_noise_only_changes_the_last_row():
    sched = beta_like_schedule(19)
    plain = build_schedule(sched, sched, sched, SIGMAS)
    trimmed = build_schedule(sched, sched, sched, SIGMAS, no_final_step_noise=True)
    assert torch.equal(plain.score, trimmed.score)
    assert torch.equal(plain.noise[:-1], trimmed.noise[:-1])
    assert trimmed.noise[-1].abs().max().item() == 0.0
    assert plain.noise[-1].abs().max().item() > 0.0


def test_ode_mode_halves_the_score_term_and_drops_the_noise():
    sched = beta_like_schedule(19)
    plain = build_schedule(sched, sched, sched, SIGMAS)
    ode = build_schedule(sched, sched, sched, SIGMAS, ode=True)
    assert ode.noise.abs().max().item() == 0.0
    assert (ode.score - 0.5 * plain.score).abs().max().item() < 1e-6


def test_no_random_zeroes_every_noise_coefficient():
    sched = beta_like_schedule(19)
    table = build_schedule(sched, sched, sched, SIGMAS, no_random=True)
    assert table.noise.abs().max().item() == 0.0


@pytest.mark.parametrize("temp", [0.5, 1.4])
def test_the_temperature_knobs_match_upstreams_lambda(temp):
    sched = beta_like_schedule(19)
    psi, sigma_data = 0.3, 0.5
    table = build_schedule(sched, sched, sched, SIGMAS,
                           temp_sampling=(temp,) * 3, temp_psi=(psi,) * 3,
                           temp_sigma_data=(sigma_data,) * 3)
    lo, hi = SIGMAS['tr']
    for t_idx in (0, 7, 18):
        t = sched[t_idx]
        dt = sched[t_idx] - sched[t_idx + 1] if t_idx < 18 else sched[t_idx]
        sigma = lo ** (1 - t) * hi ** t
        g = sigma * math.sqrt(2 * math.log(hi / lo))
        s_data = math.exp(sigma_data * math.log(hi) + (1 - sigma_data) * math.log(lo))
        lam = (s_data + sigma) / (s_data + sigma / temp)
        want_s = g * g * dt * (lam + temp * psi / 2)
        want_n = g * math.sqrt(dt * (1 + psi))
        assert abs(table.score[t_idx, 0].item() - want_s) < abs(want_s) * TOL + TOL
        assert abs(table.noise[t_idx, 0].item() - want_n) < abs(want_n) * TOL + TOL


def test_the_nan_guard_is_a_no_op_on_a_clean_score():
    g = torch.Generator().manual_seed(1)
    score = torch.randn(10, 3, generator=g)
    clean, n_nan = guard_scores(score)
    assert torch.equal(clean, score)
    assert n_nan.item() == 0


def test_the_nan_guard_replaces_like_upstream_and_counts_the_rows():
    score = torch.tensor([[1.0, -2.0, 3.0], [float('nan'), 0.5, 0.5],
                          [float('inf'), 1.0, -1.0], [float('-inf'), 1.0, 1.0]])
    clean, n_nan = guard_scores(score)
    finite = torch.isfinite(score)
    eps = 0.01 * score.abs()[finite].mean()
    assert n_nan.item() == 3, "every non-finite row is counted, as upstream's mean-isnan does"
    assert torch.isfinite(clean).all()
    assert torch.equal(clean[0], score[0])
    assert clean[1, 0].item() == pytest.approx(eps.item())
    assert clean[2, 0].item() == pytest.approx(eps.item())
    assert clean[3, 0].item() == pytest.approx(-eps.item())
