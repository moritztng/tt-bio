# La-Proteina flow-matching Euler sampler -- ttnn port (pass 4).
#
# SPDX-License-Identifier: Apache-2.0
#
# Port of the La-Proteina flow-matching integration step (the Euler loop body
# around the denoiser). Reference: proteinfoundation/flow_matching/rdn_flow_matcher.py
# `RDNFlowMatcher.simulation_step` + `vf_to_score` / `score_to_vf` +
# `_apply_mask` / `_force_zero_com` (Apache-2.0, vendored under
# _vendor/la-proteina-ref). The denoiser NN itself is parity-verified in
# `denoiser.py`; this module is the integrator that wraps it.
#
# The full sampler LOOP (nsteps around the denoiser, with the FeatureFactory /
# PairReprBuilder building seqs / c / pair_rep from x_t / t each step) is NOT
# ported here -- it is gated on the dataset feature-pipeline port (the same
# gate that blocks real-weight parity and the full end-to-end forward). What
# IS ported and parity-checked is the per-step integrator math: the ODE/SDE
# Euler step for all four sampling modes (vf, vf_ss, sc, vf_ss_sc_sn), the
# score<->vector-field transforms, masking, and the optional center-of-mass
# zeroing. The stochastic noise draw (`eps`) is taken as an explicit input
# (NOT drawn on device) so the parity harness can feed identical draws to the
# device port and the CPU reference (per memory `diffusion-port-parity-shared-draws`).
#
# Shapes: x_t, v, eps [B, N, d] (d = 3 for bb_ca, 8 for local_latents); the last
# dim is NOT tile-aligned, so on device it is TILE-padded to 32 -- elementwise
# ops are unaffected (padded lanes stay zero) and the harness slices [:, :, :d]
# before comparing.

from __future__ import annotations

import torch
import ttnn


def _pcc(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def tt_vf_to_score(x_t, v, t: float):
    # s = (t*v - x_t) / (1 - t)
    num = ttnn.subtract(ttnn.multiply(v, t), x_t)
    return ttnn.multiply(num, 1.0 / (1.0 - t))


def tt_score_to_vf(x_t, score, t: float):
    # v = (x_t + (1 - t)*score) / t
    h = ttnn.add(x_t, ttnn.multiply(score, 1.0 - t))
    return ttnn.multiply(h, 1.0 / t)


def tt_apply_mask(x, mask):
    # mask: [B, N, 1] float (1.0 valid, 0.0 masked)
    return ttnn.multiply(x, mask)


def tt_force_zero_com(x, mask):
    # x - mean_w_mask(x, mask) over n (dim=-2), then * mask
    num = ttnn.sum(mask, dim=-2, keepdim=True)              # [B,1,1]
    xm = ttnn.multiply(x, mask)
    s = ttnn.sum(xm, dim=-2, keepdim=True)                   # [B,1,d_padded]
    inv = ttnn.reciprocal(num)
    s = ttnn.multiply(s, inv)
    out = ttnn.subtract(x, s)
    return ttnn.multiply(out, mask)


class TTEulerStep:
    """ttnn port of `RDNFlowMatcher.simulation_step` (one Euler integration step).

    Inputs (device tensors, TILE_LAYOUT, bf16):
      x_t, v, eps : [B, N, d]   (eps = the shared stochastic draw, fed in)
      mask        : [B, N, 1]   (float 1.0/0.0)
    Scalars (python floats / str): t, dt, gt, sampling_mode, sc_scale_noise,
    sc_scale_score, t_lim_ode, t_lim_ode_below, center_every_step, scale_ref (=1).

    Returns x_next [B, N, d] (device).
    """

    def __init__(self, device, ck, dtype=ttnn.bfloat16):
        self.device = device
        self.ck = ck
        self.dtype = dtype

    def __call__(
        self, x_t, v, eps, mask, t, dt, gt,
        sampling_mode="vf",
        sc_scale_noise=0.0, sc_scale_score=1.0,
        t_lim_ode=0.93, t_lim_ode_below=0.07,
        center_every_step=False,
    ):
        t_e = t  # scalar, same for all samples (asserted in the reference)
        sc_scale_score_def = 1.5
        sc_scale_noise_def = 0.3

        if sampling_mode == "vf":
            delta_x = ttnn.multiply(v, dt)

        elif sampling_mode == "vf_ss":
            if t_e < t_lim_ode_below:
                score = tt_vf_to_score(x_t, v, t_e)
                std_eps = (2.0 * gt * sc_scale_noise_def * dt) ** 0.5
                delta_x = ttnn.add(
                    ttnn.multiply(ttnn.add(v, ttnn.multiply(score, gt)), dt),
                    ttnn.multiply(eps, std_eps),
                )
            else:
                score = tt_vf_to_score(x_t, v, t_e)
                scaled_score = ttnn.multiply(score, sc_scale_score)
                v_scaled = tt_score_to_vf(x_t, scaled_score, t_e)
                delta_x = ttnn.multiply(v_scaled, dt)

        elif sampling_mode == "sc":
            if t_e > t_lim_ode:
                score = tt_vf_to_score(x_t, v, t_e)
                scaled_score = ttnn.multiply(score, sc_scale_score_def)
                v_scaled = tt_score_to_vf(x_t, scaled_score, t_e)
                delta_x = ttnn.multiply(v_scaled, dt)
            else:
                score = tt_vf_to_score(x_t, v, t_e)
                std_eps = (2.0 * gt * sc_scale_noise * dt) ** 0.5
                delta_x = ttnn.add(
                    ttnn.multiply(ttnn.add(v, ttnn.multiply(score, gt)), dt),
                    ttnn.multiply(eps, std_eps),
                )

        elif sampling_mode == "vf_ss_sc_sn":
            if t_e > t_lim_ode:
                score = tt_vf_to_score(x_t, v, t_e)
                scaled_score = ttnn.multiply(score, sc_scale_score_def)
                v_scaled = tt_score_to_vf(x_t, scaled_score, t_e)
                delta_x = ttnn.multiply(v_scaled, dt)
            elif t_e < t_lim_ode_below:
                score = tt_vf_to_score(x_t, v, t_e)
                std_eps = (2.0 * gt * sc_scale_noise_def * dt) ** 0.5
                delta_x = ttnn.add(
                    ttnn.multiply(ttnn.add(v, ttnn.multiply(score, gt)), dt),
                    ttnn.multiply(eps, std_eps),
                )
            else:
                score = tt_vf_to_score(x_t, v, t_e)
                scaled_score = ttnn.multiply(score, sc_scale_score)
                v_scaled = tt_score_to_vf(x_t, scaled_score, t_e)
                std_eps = (2.0 * gt * sc_scale_noise * dt) ** 0.5
                delta_x = ttnn.add(
                    ttnn.multiply(ttnn.add(v_scaled, ttnn.multiply(score, gt)), dt),
                    ttnn.multiply(eps, std_eps),
                )
        else:
            raise ValueError(f"Invalid sampling mode {sampling_mode}")

        x_next = ttnn.add(x_t, delta_x)
        x_next = tt_apply_mask(x_next, mask)
        if center_every_step:
            x_next = tt_force_zero_com(x_next, mask)
        return x_next
