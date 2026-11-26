"""The traced form of `sampling.sample`: one fixed-shape step, and the driver that repeats it.

`tt_bio/diffdock/sampling.py` owns the sampler (D-dfd-orch-7): the schedule, the sigma bounds,
the temperature terms and `modify_conformer`, all graded bit-identical against upstream's own
loop over 133 rollouts. Everything here is defined *by* that module and graded *against* it.
What this module adds is only what a trace needs and the eager sampler does not:

* **The step constants as one table, computed before the loop.** :func:`coefficient_table`
  runs sampling's own arithmetic -- `cfg.schedules`, `bounds.sigmas`, `bounds.g`,
  `_temperature_scale`, the ODE and quiet-step rules -- for every step up front into a
  `[steps, 3]` score coefficient and a `[steps, 3]` noise coefficient. The loop body reads a
  row instead of calling back into Python, and every sampling option becomes table contents
  rather than a branch in the body.
* **The NaN guard without the host branch.** Sampling's `_repair_nans` asks
  `bool(isnan(...).any())` on the host and repairs only then. :func:`guard_scores` computes
  the same repair unconditionally and selects it with that predicate as a *device* tensor:
  equal on every input, and no sync.
* **The pose update without a masked write or an SVD.** :func:`apply_torsions` replaces
  `pos[:, moving] = ...`, whose width is read off the mask at run time, with
  `where(mask, rotated, pos)`; :func:`fixed_update` replaces the SVD Kabsch with Horn's
  quaternion (:mod:`tt_bio.diffdock.kabsch`), which covers upstream's reflection branch
  without branching.
* **The driver**, :func:`run_loop`, which issues the identical program at every step --
  proved op for op by `tests/test_diffdock_trace.py`.

The rotation helper itself is sampling's `axis_angle_to_matrix`, imported, not re-derived: it
is already branch-free (a `where` on the small-angle series), so there was nothing to fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .harmonics import unit
from .kabsch import kabsch_horn
from .sampling import SamplingConfig, SigmaBounds, _temperature_scale, axis_angle_to_matrix


@dataclass(frozen=True)
class CoefficientTable:
    """``perturb = score[t, h] * score_h + noise[t, h] * z_h`` for the three heads.

    ``drawn[t]`` records whether sampling draws noise at step t at all -- it does not on a
    quiet step (``no_random``, or ``no_final_step_noise`` on the last one), and a caller
    feeding the same RNG as the eager sampler has to skip the draw, not multiply it by zero.
    """

    score: Tensor
    noise: Tensor
    drawn: tuple

    @property
    def steps(self) -> int:
        return int(self.score.shape[0])

    def row(self, t_idx: int):
        return self.score[t_idx], self.noise[t_idx]


def coefficient_table(cfg: SamplingConfig, bounds: SigmaBounds,
                      dtype=None) -> CoefficientTable:
    """Sampling's per-step constants for every step, evaluated before the loop starts.

    One row per step, three columns (tr, rot, tor) each. ODE halves the score term and zeroes
    the noise term; a quiet step zeroes the noise term; the low-temperature constants scale
    both. Same expressions as `sampling.sample`, in the same float64 numpy arithmetic, so the
    table is not an approximation of the sampler but a transcription of it. It is built in
    float64 and cast once at the end, so the cast is the only rounding the constants take --
    *dtype* defaults to the ambient default so a float64 rollout gets float64 constants.
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    scheds = cfg.schedules(bounds)
    steps = cfg.steps
    score = np.zeros((steps, 3))
    noise = np.zeros((steps, 3))
    drawn = []
    for t_idx in range(steps):
        last = t_idx == steps - 1
        quiet = cfg.no_random or (cfg.no_final_step_noise and last)
        drawn.append(not (cfg.ode or quiet))
        ts = [float(s[t_idx]) for s in scheds]
        dts = [float(s[t_idx] - s[t_idx + 1]) if not last else float(s[t_idx]) for s in scheds]
        sigmas = bounds.sigmas(*ts)
        for head, (name, sigma, dt) in enumerate(zip(("tr", "rot", "tor"), sigmas, dts)):
            g = bounds.g(name, sigma)
            if cfg.ode:
                score[t_idx, head] = 0.5 * g * g * dt
                continue
            a, w = _temperature_scale(head, cfg, sigma, bounds, name)
            score[t_idx, head] = g * g * dt * a
            noise[t_idx, head] = 0.0 if quiet else g * np.sqrt(dt * w)
    return CoefficientTable(torch.as_tensor(score, dtype=dtype),
                            torch.as_tensor(noise, dtype=dtype), tuple(drawn))


def guard_scores(score: Tensor) -> tuple[Tensor, Tensor]:
    """Sampling's `_repair_nans` for one head, with the host branch removed.

    Upstream (and sampling, following it) replaces every non-finite entry with 1 % of the
    head's mean magnitude, but only after a host-side `bool(isnan(...).any())`. Computing the
    repair unconditionally gives the identical tensor -- the replacement is a no-op when there
    is nothing to replace -- and a `where` on a device predicate keeps the step host-free.
    Returns the repaired score and the number of poses that were non-finite, to be accumulated
    on device and read once after the loop, where it only ever drove a warning.

    One deliberate difference from `nanmean`: an infinity is excluded from the mean rather
    than poisoning it, so `eps` stays finite in the case the guard exists for.
    """
    finite = torch.isfinite(score)
    eps = 0.01 * torch.where(finite, score.abs(), torch.zeros_like(score)).sum() \
        / finite.sum().clamp_min(1)
    sign = torch.sign(torch.nan_to_num(score, nan=1.0, posinf=1.0, neginf=-1.0))
    repaired = torch.where(finite, score, eps * sign)
    n_bad = (~torch.isfinite(score.reshape(score.shape[0], -1).sum(dim=-1))).sum()
    return repaired, n_bad


def apply_torsions(pos: Tensor, bonds, mask_rotate: Tensor, updates: Tensor) -> Tensor:
    """Sampling's `torsion_update_batch` with a select in place of the masked write.

    *pos* `[B, N, 3]`; *bonds* the rotatable `(u, v)` pairs as Python ints, `v` on the moving
    side; *mask_rotate* `[R, N]`; *updates* `[B, R]` radians. The bond count is fixed per
    ligand, so the loop unrolls at capture -- but only if the pairs are ints, which is why
    they are taken that way rather than as a tensor to index.

    The rotation is applied to every atom and the mask chooses, so no shape depends on the
    data. A pad bond (both ends the pad atom) gives a zero-length axis, which :func:`unit`
    sends to zero and `axis_angle_to_matrix` to the identity; its mask row is zero anyway.
    """
    if mask_rotate.shape[0] != len(bonds) or updates.shape[-1] != len(bonds):
        raise ValueError(f"{len(bonds)} bonds, mask {tuple(mask_rotate.shape)}, "
                         f"updates {tuple(updates.shape)}")
    if mask_rotate.shape[1] != pos.shape[1]:
        raise ValueError(f"mask has {mask_rotate.shape[1]} atoms, positions {pos.shape[1]}")
    sel = mask_rotate.to(torch.bool)[:, None, :, None]
    for idx, (u, v) in enumerate(bonds):
        pivot = pos[:, v:v + 1]
        axis = unit(pos[:, u] - pos[:, v])
        rot = axis_angle_to_matrix(axis * updates[:, idx:idx + 1])
        rotated = torch.bmm(pos - pivot, rot.transpose(1, 2)) + pivot
        pos = torch.where(sel[idx], rotated, pos)
    return pos


def fixed_update(pos: Tensor, tr_update: Tensor, rot_update: Tensor,
                 torsion_updates: Tensor | None, bonds, mask_rotate: Tensor | None):
    """Sampling's `modify_conformer`, fixed-shape. Returns `(pos, det)`.

    `det` is the correlation determinant from :func:`~tt_bio.diffdock.kabsch.kabsch_horn`, a
    diagnostic only: negative means the step hit upstream's reflection branch, which Horn's
    quaternion handles without taking one. Accumulate it on device, read it after the loop.
    """
    centre = pos.mean(dim=1, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update)
    rigid = torch.bmm(pos - centre, rot_mat.permute(0, 2, 1)) + tr_update.unsqueeze(1) + centre
    if torsion_updates is None or not len(bonds):
        return rigid, None
    flexible = apply_torsions(rigid, bonds, mask_rotate,
                              torsion_updates.reshape(rigid.shape[0], -1))
    rot, t, det = kabsch_horn(flexible, rigid)
    return torch.bmm(flexible, rot.transpose(1, 2)) + t.transpose(1, 2), det


def diffusion_step(pos, scores, score_coeff, noise_coeff, noise, bonds, mask_rotate):
    """One step: guard the three scores, perturb, move. Returns `(pos, n_bad, det)`.

    *scores* and *noise* are `(tr, rot, tor)` triples, `[B, 3]`, `[B, 3]`, `[B, R]`. Nothing
    here reads a value off the device.
    """
    tr_score, n_tr = guard_scores(scores[0])
    rot_score, _ = guard_scores(scores[1])
    tr_perturb = score_coeff[0] * tr_score + noise_coeff[0] * noise[0]
    rot_perturb = score_coeff[1] * rot_score + noise_coeff[1] * noise[1]
    if scores[2] is None:
        tor_perturb = None
    else:
        tor_score, _ = guard_scores(scores[2])
        tor_perturb = score_coeff[2] * tor_score + noise_coeff[2] * noise[2]
    pos, det = fixed_update(pos, tr_perturb, rot_perturb, tor_perturb, bonds, mask_rotate)
    return pos, n_tr, det


def run_loop(pos, score_fn, table: CoefficientTable, noise_fn, bonds, mask_rotate):
    """Every step of the rollout, same program each time.

    `score_fn(pos, t_idx)` returns the three scores and `noise_fn(t_idx)` the three draws.
    Returns `(pos, n_bad_total, min_det)` -- both diagnostics accumulated on device and read
    once by the caller, never inside the loop.
    """
    n_bad = torch.zeros((), dtype=torch.long)
    min_det = None
    for t_idx in range(table.steps):
        s, z = table.row(t_idx)
        pos, step_bad, det = diffusion_step(pos, score_fn(pos, t_idx), s, z, noise_fn(t_idx),
                                            bonds, mask_rotate)
        n_bad = n_bad + step_bad
        if det is not None:
            min_det = det.min() if min_det is None else torch.minimum(min_det, det.min())
    return pos, n_bad, min_det
