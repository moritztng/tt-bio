"""The 19-step reverse diffusion loop, with every host-shaped quantity lifted out of it.

`utils/sampling.py::sampling` interleaves three kinds of work: the score network (device), the
pose update (device, :mod:`tt_bio.diffdock.conformer`), and a page of Python that computes
step constants, decides whether to draw noise, and branches on a NaN count. Only the third
kind would break a trace, and none of it depends on anything the device computes:

* **The step constants are functions of the step index alone.** `t_to_sigma` on the schedule,
  the two `g` factors, the `lambda` low-temperature corrections, `sqrt(dt)` -- all of it is
  arithmetic on `tr_schedule[t]`, which is fixed before the run. :func:`build_schedule`
  evaluates all 19 steps up front into one `[steps, 6]` table of (score coefficient, noise
  coefficient) per axis, so the loop body reads a row instead of calling into Python. Every
  variant collapses into that pair: `ode` sets the noise coefficient to zero and halves the
  score one, `no_final_step_noise` zeroes the last row's noise coefficients, and the
  temperature knobs scale them -- so the body has one form and the table carries the options.
* **The noise draws are inputs.** Fixed shape, no dependence on the state, so they are
  written into the same device buffers every step rather than allocated inside the loop.
* **The NaN guard is a count, not a branch.** Upstream checks `isnan(mean(tr_score))`, warns,
  and *then* replaces NaNs with a small value derived from the mean of the finite entries. The
  replacement is a no-op when there are no NaNs, so running it unconditionally gives the same
  numbers; the count it warned on is accumulated on device and read once after the loop, which
  gives the same warning a step later. :func:`guard_scores` does both.

What this leaves inside the loop is the score network, the pose update, and a table lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .conformer import modify_conformer


def t_to_sigma(t: float, sigma_min: float, sigma_max: float) -> float:
    """Upstream's exponential interpolation, `sigma_min^(1-t) * sigma_max^t`."""
    return sigma_min ** (1 - t) * sigma_max ** t


def _axis_coefficients(schedule, t_idx, n_steps, sigma_min, sigma_max, *, ode, final_no_noise,
                       temp_sampling, temp_psi, temp_sigma_data):
    """One axis' `(score_coeff, noise_coeff)` for one step: `perturb = s*score + n*z`."""
    t = schedule[t_idx]
    dt = schedule[t_idx] - schedule[t_idx + 1] if t_idx < n_steps - 1 else schedule[t_idx]
    sigma = t_to_sigma(t, sigma_min, sigma_max)
    g = sigma * math.sqrt(2 * math.log(sigma_max / sigma_min))

    if ode:
        return 0.5 * g * g * dt, 0.0
    score_coeff, noise_coeff = g * g * dt, g * math.sqrt(dt)
    if temp_sampling != 1.0:
        sigma_data = math.exp(temp_sigma_data * math.log(sigma_max)
                              + (1 - temp_sigma_data) * math.log(sigma_min))
        lam = (sigma_data + sigma) / (sigma_data + sigma / temp_sampling)
        score_coeff = g * g * dt * (lam + temp_sampling * temp_psi / 2)
        noise_coeff = g * math.sqrt(dt * (1 + temp_psi))
    if final_no_noise and t_idx == n_steps - 1:
        noise_coeff = 0.0
    return score_coeff, noise_coeff


@dataclass(frozen=True)
class Schedule:
    """`[n_steps, 3]` score coefficients and `[n_steps, 3]` noise coefficients, axes tr/rot/tor."""
    score: Tensor
    noise: Tensor

    @property
    def n_steps(self) -> int:
        return self.score.shape[0]

    def row(self, t_idx: int):
        return self.score[t_idx], self.noise[t_idx]


def build_schedule(tr_schedule, rot_schedule, tor_schedule, sigmas, *, ode=False,
                   no_random=False, no_final_step_noise=False, temp_sampling=(1.0, 1.0, 1.0),
                   temp_psi=(0.0, 0.0, 0.0), temp_sigma_data=(0.5, 0.5, 0.5),
                   dtype=torch.float32) -> Schedule:
    """Every step's constants, computed on the host before the loop starts.

    *sigmas* is `{'tr': (min, max), 'rot': (...), 'tor': (...)}`, the model's own hyper-
    parameters. *no_random* zeroes every noise coefficient (upstream's deterministic mode).
    """
    schedules = {'tr': tr_schedule, 'rot': rot_schedule, 'tor': tor_schedule}
    n_steps = len(tr_schedule)
    score = torch.zeros(n_steps, 3, dtype=torch.float64)
    noise = torch.zeros(n_steps, 3, dtype=torch.float64)
    for axis, (name, sched) in enumerate(schedules.items()):
        lo, hi = sigmas[name]
        for t_idx in range(n_steps):
            s, z = _axis_coefficients(sched, t_idx, n_steps, lo, hi, ode=ode,
                                      final_no_noise=no_final_step_noise,
                                      temp_sampling=temp_sampling[axis], temp_psi=temp_psi[axis],
                                      temp_sigma_data=temp_sigma_data[axis])
            score[t_idx, axis], noise[t_idx, axis] = s, 0.0 if no_random else z
    return Schedule(score.to(dtype), noise.to(dtype))


def guard_scores(score: Tensor) -> tuple[Tensor, Tensor]:
    """Upstream's NaN rescue, unconditional: `(clean_score, n_nan_rows)`.

    Upstream replaces NaN/inf with `0.01 * nanmean(|score|)` only when it has already found a
    NaN; with none present that call changes nothing, so doing it every step is the same
    function of the input and costs one pass. The count comes back as a device tensor to be
    accumulated and read after the loop -- it only ever drove a log line.

    One deliberate difference: upstream's `eps` is `nanmean(|score|)`, which ignores NaN but
    *not* an infinity, so a single `inf` makes upstream's replacement value `inf` and poisons
    the whole batch. This averages the finite entries only. The two agree exactly whenever no
    infinity is present, which is every case upstream's own guard was written for.
    """
    finite = torch.isfinite(score)
    n_finite = finite.sum().clamp_min(1)
    eps = 0.01 * torch.where(finite, score.abs(), torch.zeros_like(score)).sum() / n_finite
    clean = torch.where(finite, score, eps * torch.sign(torch.nan_to_num(score, nan=1.0,
                                                                        posinf=1.0, neginf=-1.0)))
    n_nan = (~torch.isfinite(score.mean(dim=-1))).sum()
    return clean, n_nan


def diffusion_step(pos, tr_score, rot_score, tor_score, score_coeff, noise_coeff,
                   tr_z, rot_z, tor_z, bonds, mask_rotate):
    """One reverse-diffusion step: guard, perturb, move. Returns `(pos, n_nan, det)`.

    Shapes are `[B, N, 3]` positions, `[B, 3]` translation and rotation scores and noise, and
    `[B, R]` torsion score and noise. *score_coeff* and *noise_coeff* are the `[3]` rows from
    :class:`Schedule`. Nothing here reads a value off the device.
    """
    tr_score, n_tr = guard_scores(tr_score)
    rot_score, _ = guard_scores(rot_score)
    tr_perturb = score_coeff[0] * tr_score + noise_coeff[0] * tr_z
    rot_perturb = score_coeff[1] * rot_score + noise_coeff[1] * rot_z
    if tor_score is None:
        tor_perturb = None
    else:
        tor_score, _ = guard_scores(tor_score)
        tor_perturb = score_coeff[2] * tor_score + noise_coeff[2] * tor_z
    pos, det = modify_conformer(pos, tr_perturb, rot_perturb, tor_perturb, bonds, mask_rotate)
    return pos, n_tr, det


def run_loop(pos, score_fn, schedule: Schedule, noise_fn, bonds, mask_rotate):
    """The whole sampler: `score_fn(pos, t_idx)` -> the three scores, `noise_fn(t_idx)` -> the draws.

    Returns `(pos, n_nan_total, min_det)` -- the two diagnostics accumulated on device, read
    once by the caller after the loop, never inside it.
    """
    n_nan = torch.zeros((), dtype=torch.long)
    min_det = None
    for t_idx in range(schedule.n_steps):
        s, z = schedule.row(t_idx)
        tr_score, rot_score, tor_score = score_fn(pos, t_idx)
        tr_z, rot_z, tor_z = noise_fn(t_idx)
        pos, step_nan, det = diffusion_step(pos, tr_score, rot_score, tor_score, s, z,
                                            tr_z, rot_z, tor_z, bonds, mask_rotate)
        n_nan = n_nan + step_nan
        if det is not None:
            d = det.min()
            min_det = d if min_det is None else torch.minimum(min_det, d)
    return pos, n_nan, min_det
