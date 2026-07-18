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

    def __init__(self, device, ck, dtype=ttnn.bfloat16, math_dtype=None):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.math_dtype = math_dtype if math_dtype is not None else dtype

    def __call__(
        self, x_t, v, eps, mask, t, dt, gt,
        sampling_mode="vf",
        sc_scale_noise=0.0, sc_scale_score=1.0,
        t_lim_ode=0.93, t_lim_ode_below=0.07,
        center_every_step=False,
    ):
        # run the score / delta math in math_dtype (fp32) to avoid bf16 rounding
        # in the 1/(1-t) score amplification; cast back to dtype before masking.
        if self.math_dtype != self.dtype:
            x_t = ttnn.typecast(x_t, self.math_dtype)
            v = ttnn.typecast(v, self.math_dtype)
            eps = ttnn.typecast(eps, self.math_dtype)
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
        if self.math_dtype != self.dtype:
            x_next = ttnn.typecast(x_next, self.dtype)
        x_next = tt_apply_mask(x_next, mask)
        if center_every_step:
            x_next = tt_force_zero_com(x_next, mask)
        return x_next



# ---------------------------------------------------------------------------
# Full nsteps flow-matching sampler loop (pass 5).
#
# Wraps the parity-verified denoiser NN (TTLaProteinaDenoiser, feature pipeline
# wired in) + the parity-verified Euler integrator (TTEulerStep) and runs the
# full nsteps loop: x = sample_noise; for each step build features from
# x_t/t (and optional x_sc), run the denoiser, take an Euler step. Mirrors
# proteinfoundation/flow_matching/product_space_flow_matcher.full_simulation
# (guidance_w=1.0, no CFG/AG, n_recycle=0 -- the uncond 160M config).
#
# Stochastic draws: the loop's only randomness is the initial noise (sample_noise)
# and the per-step SDE noise (eps inside simulation_step). For parity vs the
# reference loop these must be SHARED draws (per memory
# diffusion-port-parity-shared-draws). The harness patches torch.randn to draw
# from a seeded generator; this loop draws eps via torch.randn at the SAME
# conditional points and in the SAME per-data-mode order as the reference
# simulation_step, so resetting the generator between the golden and device
# runs yields identical draws. eps is passed explicitly to TTEulerStep.
# ---------------------------------------------------------------------------


def _get_schedule(mode, nsteps, p1, eps=1e-5):
    # Mirrors product_space_flow_matcher.get_schedule.
    if mode == "uniform":
        return torch.linspace(0, 1, nsteps + 1)
    if mode == "power":
        t = torch.linspace(0, 1, nsteps + 1)
        return t ** p1
    if mode == "log":
        t = 1.0 - torch.logspace(-p1, 0, nsteps + 1).flip(0)
        t = t - torch.min(t)
        t = t / torch.max(t)
        return t
    raise IOError(f"Schedule mode not recognized {mode}")


def _get_gt(t, mode, param, clamp_val, eps=1e-2):
    # Mirrors product_space_flow_matcher.get_gt (param=f_pow; 1.0 => no transform).
    t = torch.clamp(t, 0, 1 - 1e-5)
    if mode == "1-t/t":
        gt = (1.0 - t) / (t + eps)
    elif mode == "tan":
        num = torch.sin((1.0 - t) * torch.pi / 2.0)
        den = torch.cos((1.0 - t) * torch.pi / 2.0)
        gt = (torch.pi / 2.0) * num / (den + eps)
    elif mode == "1/t":
        gt = 1.0 / (t + eps)
    else:
        raise NotImplementedError(f"gt not implemented {mode}")
    if param != 1.0:
        log_gt = torch.log(gt)
        mean_log_gt = torch.mean(log_gt)
        log_gt_centered = log_gt - mean_log_gt
        normalized = torch.sigmoid(log_gt_centered)
        normalized = normalized ** param
        log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
        gt = torch.exp(log_gt_centered_rec + mean_log_gt)
    gt = torch.clamp(gt, 0, clamp_val)
    return gt


def _draws_eps_for_step(t_scalar, sampling_mode, t_lim_ode, t_lim_ode_below):
    # Mirrors the reference simulation_step's conditional torch.randn draws.
    if sampling_mode == "vf":
        return False
    if sampling_mode == "vf_ss":
        return t_scalar < t_lim_ode_below
    if sampling_mode == "sc":
        return not (t_scalar > t_lim_ode)
    if sampling_mode == "vf_ss_sc_sn":
        return not (t_scalar > t_lim_ode)
    raise ValueError(sampling_mode)


class TTLaProteinaSampler:
    """Full nsteps flow-matching sampler loop (ttnn port).

    Owns the denoiser (TTLaProteinaDenoiser) + the Euler integrator (TTEulerStep)
    and runs the loop. Inputs are device tensors; the initial noise x0 and the
    per-step eps are drawn by the harness via a patched torch.randn (shared with
    the golden) and passed in. Returns the final x dict {dm: device tensor}.
    """

    def __init__(
        self,
        device,
        ck,
        denoiser,
        data_modes,
        sampling_model_args,
        latent_dims,
        dtype=ttnn.bfloat16,
        math_dtype=None,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.denoiser = denoiser
        self.data_modes = list(data_modes)
        self.args = sampling_model_args
        self.latent_dims = latent_dims  # {dm: dim}
        self.euler = TTEulerStep(device, ck, dtype=dtype, math_dtype=math_dtype)

    def __call__(self, x0, mask_tt, pair_mask_tt, pmb_tt, nsteps, n, self_cond=True):
        dev = self.device
        dt_host = self.dtype
        b = 1
        data_modes = self.data_modes
        # schedules + gt (host)
        ts = {
            dm: _get_schedule(self.args[dm]["schedule"]["mode"], nsteps,
                              self.args[dm]["schedule"]["p"])
            for dm in data_modes
        }
        gt = {
            dm: _get_gt(ts[dm][:-1], self.args[dm]["gt"]["mode"],
                        self.args[dm]["gt"]["p"], self.args[dm]["gt"]["clamp_val"])
            for dm in data_modes
        }
        # initial x (already drawn + masked + on device by the harness)
        x = {dm: x0[dm] for dm in data_modes}
        x_1_pred = None
        for step in range(nsteps):
            t = {dm: float(ts[dm][step]) for dm in data_modes}
            dt = {dm: float(ts[dm][step + 1] - ts[dm][step]) for dm in data_modes}
            gt_step = {dm: float(gt[dm][step]) for dm in data_modes}
            batch = {"x_t": x, "t": t, "mask": mask_tt}
            if self_cond and step > 0 and x_1_pred is not None:
                batch["x_sc"] = x_1_pred
            nn_out = self.denoiser(batch, mask_tt, pair_mask_tt, pmb_tt, b, n)
            # x_1 = x_t + (1 - t) * v, then * mask  (per data mode, on device)
            x_1_pred = {}
            for dm in data_modes:
                v = nn_out[dm]["v"]
                x1 = ttnn.add(x[dm], ttnn.multiply(v, 1.0 - t[dm]))
                x1 = ttnn.multiply(x1, mask_tt)
                x_1_pred[dm] = x1
            # Euler step per data mode (eps drawn via patched torch.randn at the
            # same conditional points + order as the reference simulation_step).
            x_new = {}
            for dm in data_modes:
                p = self.args[dm]["simulation_step_params"]
                do_draw = _draws_eps_for_step(t[dm], p["sampling_mode"],
                                              p["t_lim_ode"], p["t_lim_ode_below"])
                d = self.latent_dims[dm]
                tile_in = ((d + 31) // 32) * 32
                if do_draw:
                    eps_host = torch.randn(b, n, d, dtype=torch.float32)
                else:
                    eps_host = torch.zeros(b, n, d, dtype=torch.float32)
                if tile_in != d:
                    eps_host = torch.cat(
                        [eps_host, torch.zeros(b, n, tile_in - d)], dim=-1)
                eps_tt = ttnn.from_torch(eps_host, layout=ttnn.TILE_LAYOUT,
                                         device=dev, dtype=dt_host)
                x_new[dm] = self.euler(
                    x[dm], nn_out[dm]["v"], eps_tt, mask_tt,
                    t=t[dm], dt=dt[dm], gt=gt_step[dm],
                    sampling_mode=p["sampling_mode"],
                    sc_scale_noise=p["sc_scale_noise"],
                    sc_scale_score=p["sc_scale_score"],
                    t_lim_ode=p["t_lim_ode"], t_lim_ode_below=p["t_lim_ode_below"],
                    center_every_step=p["center_every_step"],
                )
            x = x_new
        return x
