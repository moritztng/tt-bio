"""RFD3 EDM sampler (host-side orchestration) reusing the verified ttnn
``RFD3DiffusionModule`` forward. Faithful to upstream
``rfd3.model.inference_sampler`` (``SampleDiffusionWithMotif`` /
``SampleDiffusionWithSymmetry``), AF3-family EDM solver.

Implements the default solver plus two conditioning modes that are pure
host orchestration around the per-step device forward (no new device code):
  * F7 partial diffusion (``partial_t`` angstroms -> subset the noise schedule;
    start from a real input structure instead of pure noise).
  * Classifier-free guidance (a second "unconditional" forward with the cfg
    features zeroed, combined into the per-step delta).

The same sampler drives both the ttnn device module and the vendored torch
reference (identical ``__call__`` signature), so device-vs-reference parity
with shared random draws isolates the device forward under each mode.
"""

from __future__ import annotations

import math
from typing import Any

import torch


# --- classifier-free guidance helpers (faithful port of rfd3.model.cfg_utils) ---
def strip_f(f: dict[str, torch.Tensor], cfg_features: list[str]) -> dict[str, torch.Tensor]:
    """Zero the cfg conditioning features and crop unindexed atoms/tokens.

    With no unindexed atoms (the common binder-design case) this reduces to
    zeroing the cfg_features; shapes are unchanged. Mirrors upstream
    ``strip_f`` exactly so the unconditional pass matches the reference.
    """
    token_dim = f["is_motif_token_unindexed"].shape[0]
    atom_dim = f["is_motif_atom_unindexed"].shape[0]
    crop = bool(torch.any(f["is_motif_atom_unindexed"]).item())
    atom_crop = (int(torch.where(f["is_motif_atom_unindexed"])[0][0])
                 if crop else f["is_motif_atom_unindexed"].shape[0])
    token_crop = (int(torch.where(f["is_motif_token_unindexed"])[0][0])
                  if crop else f["is_motif_token_unindexed"].shape[0])
    out: dict[str, torch.Tensor] = {}
    for k, v in f.items():
        vc = v
        if token_dim in v.shape:
            if len(v.shape) == 2 and v.shape[0] == v.shape[1]:
                vc = v[:token_crop, :token_crop]
            else:
                vc = v[:token_crop]
        if atom_dim in v.shape:
            if len(v.shape) == 2 and v.shape[0] == v.shape[1]:
                vc = v[:atom_crop, :atom_crop]
            else:
                vc = v[:atom_crop]
        if k in cfg_features:
            vc = torch.zeros_like(vc).to(vc.device, dtype=vc.dtype)
        out[k] = vc
    return out


def strip_X(X_L: torch.Tensor, f_ref: dict[str, torch.Tensor]) -> torch.Tensor:
    """Crop unindexed atoms from X for the unconditional CFG pass."""
    return X_L[..., : f_ref["is_motif_atom_unindexed"].shape[0], :]


class RFD3Sampler:
    """AF3-family EDM sampler (default / partial / CFG).

    Defaults mirror ``configs/model/samplers/edm.yaml``: sigma_data=16, s_min=4e-4,
    s_max=160, p=7, gamma_0=0.6, gamma_min=1.0, noise_scale=1.003, step_scale=1.5.

    ``generator`` makes the noise draws reproducible; two samplers with
    same-seed generators see an identical draw stream (the valid device-vs-
    reference parity metric for a stochastic diffusion port).
    """

    def __init__(self, num_timesteps: int = 200, sigma_data: float = 16.0,
                 s_min: float = 4e-4, s_max: float = 160.0, p: int = 7,
                 gamma_0: float = 0.6, gamma_min: float = 1.0,
                 noise_scale: float = 1.003, step_scale: float = 1.5):
        self.num_timesteps = num_timesteps
        self.sigma_data = sigma_data
        self.s_min, self.s_max, self.p = s_min, s_max, p
        self.gamma_0, self.gamma_min = gamma_0, gamma_min
        self.noise_scale, self.step_scale = noise_scale, step_scale

    def noise_schedule(self, device, partial_t=None):
        t = torch.linspace(0, 1, self.num_timesteps, device=device)
        sched = self.sigma_data * (self.s_max ** (1 / self.p)
                                   + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))) ** self.p
        if partial_t is not None:
            pv = float(partial_t.mean() if torch.is_tensor(partial_t) else partial_t)
            sched = sched[sched <= pv]
            if len(sched) == 0:
                sched = self.noise_schedule(device)[-1:]
        return sched

    def sample(self, diffusion_module, D: int, L: int, coord, f, initializer_outputs,
               is_motif_fixed, *, generator=None, partial_t=None,
               cfg: bool = False, cfg_scale: float = 2.0, cfg_features=(),
               ref_initializer_outputs=None, f_ref=None, n_recycle=None):
        device = coord.device
        sched = self.noise_schedule(device, partial_t=partial_t)
        c0 = sched[0]
        noise0 = torch.zeros((D, L, 3), device=device)
        noise0 = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=device, generator=generator)
        noise0[..., is_motif_fixed, :] = 0
        X_L = noise0 + coord
        traj = []
        for step, (c_tm1, c_t) in enumerate(zip(sched, sched[1:])):
            gamma = self.gamma_0 if c_t > self.gamma_min else 0.0
            t_hat = c_tm1 * (gamma + 1)
            eps = (self.noise_scale * torch.sqrt(torch.square(t_hat) - torch.square(c_tm1))
                   * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=device, generator=generator))
            eps[..., is_motif_fixed, :] = 0
            X_noisy = X_L + eps
            outs = diffusion_module(X_noisy_L=X_noisy, t=t_hat.tile(D), f=f,
                                    n_recycle=n_recycle, **initializer_outputs)
            X_denoised = outs["X_L"]
            delta = (X_noisy - X_denoised) / t_hat
            if cfg and (ref_initializer_outputs is not None) and (f_ref is not None):
                X_ref = strip_X(X_noisy, f_ref)
                outs_ref = diffusion_module(X_noisy_L=X_ref, t=t_hat.tile(D), f=f_ref,
                                             n_recycle=n_recycle, **ref_initializer_outputs)
                d_ref = (X_ref - outs_ref["X_L"]) / t_hat
                if d_ref.shape[1] < delta.shape[1]:
                    d_ref = torch.cat([d_ref, torch.zeros_like(delta[:, d_ref.shape[1]:, :])], dim=1)
                delta = delta + (cfg_scale - 1) * (delta - d_ref)
            d_t = c_t - t_hat
            X_L = X_noisy + self.step_scale * d_t * delta
            traj.append({"X_noisy_L": X_noisy, "X_denoised_L": X_denoised,
                          "t_hat": t_hat, "X_L": X_L})
        return X_L, traj
