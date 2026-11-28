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
* **Nothing of the score guard.** `sampling.guard_scores` is already branch-free -- it
  repairs unconditionally and selects with a `where` on a device predicate -- so this module
  imports it rather than carrying a second copy (D-dfd-sampling-9).
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
from .sampling import (SamplingConfig, SigmaBounds, _temperature_scale, axis_angle_to_matrix,
                       guard_scores, off_schedule)


@dataclass(frozen=True)
class CoefficientTable:
    """``perturb = score[t, h] * score_h + noise[t, h] * z_h`` for the three heads.

    ``drawn[t]`` records whether sampling draws noise at step t at all -- it does not on a
    quiet step (``no_random``, or ``no_final_step_noise`` on the last one), and a caller
    feeding the same RNG as the eager sampler has to skip the draw, not multiply it by zero.
    ``limit[t, h]`` is `SigmaBounds.score_limit`, the score past which a sample is frozen.
    """

    score: Tensor
    noise: Tensor
    drawn: tuple
    limit: Tensor

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
    limit = np.zeros((steps, 3))
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
            limit[t_idx, head] = bounds.score_limit(name, sigma)
            if cfg.ode:
                score[t_idx, head] = 0.5 * g * g * dt
                continue
            a, w = _temperature_scale(head, cfg, sigma, bounds, name)
            score[t_idx, head] = g * g * dt * a
            noise[t_idx, head] = 0.0 if quiet else g * np.sqrt(dt * w)
    return CoefficientTable(torch.as_tensor(score, dtype=dtype),
                            torch.as_tensor(noise, dtype=dtype), tuple(drawn),
                            torch.as_tensor(limit, dtype=dtype))


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

    Returns `(pos, det, bad)`. `det` is the correlation determinant from
    :func:`~tt_bio.diffdock.kabsch.kabsch_horn`, a diagnostic only: negative means the step
    hit upstream's reflection branch, which Horn's quaternion handles without taking one.
    `bad` `[B]` is the same overflow mask the host rollout freezes on -- one check, both
    paths -- and like `det` it is accumulated on device and read after the loop.
    """
    centre = pos.mean(dim=1, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update)
    rigid = torch.bmm(pos - centre, rot_mat.permute(0, 2, 1)) + tr_update.unsqueeze(1) + centre
    no_overflow = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
    if torsion_updates is None or not len(bonds):
        return rigid, None, no_overflow
    flexible = apply_torsions(rigid, bonds, mask_rotate,
                              torsion_updates.reshape(rigid.shape[0], -1))
    rot, t, det, bad = kabsch_horn(flexible, rigid)
    return torch.bmm(flexible, rot.transpose(1, 2)) + t.transpose(1, 2), det, bad


def diffusion_step(pos, scores, score_coeff, noise_coeff, noise, bonds, mask_rotate,
                   limit, failed):
    """One step: guard the three scores, perturb, move. Returns `(pos, n_bad, det, failed)`.

    *scores* and *noise* are `(tr, rot, tor)` triples, `[B, 3]`, `[B, 3]`, `[B, R]`. Nothing
    here reads a value off the device.

    A sample that came in *failed*, whose score is `sampling.off_schedule` against *limit*
    or whose Kabsch correlation matrix overflowed fp32 keeps the coordinates it came in with
    and is reported in the returned *failed*. That is the host rollout's freeze-and-record
    (`sampling._move`) written as a select, so the traced program keeps its static shapes.
    """
    tr_score, n_tr = guard_scores(scores[0])
    rot_score, _ = guard_scores(scores[1])
    tor_score = None if scores[2] is None else guard_scores(scores[2])[0]
    held = failed | off_schedule((tr_score, rot_score, tor_score), limit)
    tr_perturb = score_coeff[0] * tr_score + noise_coeff[0] * noise[0]
    rot_perturb = score_coeff[1] * rot_score + noise_coeff[1] * noise[1]
    tor_perturb = (None if tor_score is None
                   else score_coeff[2] * tor_score + noise_coeff[2] * noise[2])
    moved, det, overflowed = fixed_update(pos, tr_perturb, rot_perturb, tor_perturb, bonds,
                                          mask_rotate)
    failed = held | overflowed
    return torch.where(failed[:, None, None], pos, moved), n_tr, det, failed


def run_loop(pos, score_fn, table: CoefficientTable, noise_fn, bonds, mask_rotate):
    """Every step of the rollout, same program each time.

    `score_fn(pos, t_idx)` returns the three scores and `noise_fn(t_idx)` the three draws.
    Returns `(pos, n_bad_total, min_det, failed)` -- every diagnostic accumulated on device
    and read once by the caller, never inside the loop. `failed` `[B]` is the samples frozen
    on an off-schedule score or an overflowed Kabsch, the device path's half of `Rollout.failed`.
    """
    n_bad = torch.zeros((), dtype=torch.long)
    failed = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
    min_det = None
    for t_idx in range(table.steps):
        s, z = table.row(t_idx)
        pos, step_bad, det, failed = diffusion_step(
            pos, score_fn(pos, t_idx), s, z, noise_fn(t_idx), bonds, mask_rotate,
            table.limit[t_idx], failed)
        n_bad = n_bad + step_bad
        if det is not None:
            min_det = det.min() if min_det is None else torch.minimum(min_det, det.min())
    return pos, n_bad, min_det, failed
