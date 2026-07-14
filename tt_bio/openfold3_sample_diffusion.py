"""OpenFold3 ``SampleDiffusion`` (AF3 Algorithm 18) device sampler -- the EDM sampler
loop around the gated ``OF3DiffusionModule``.

This is the P9 leg-2 sub-leg: a host-orchestrated rollout that, per step ``tau``,
runs the gated ``OF3DiffusionConditioning`` (with the step's t-dependent Fourier noise
embedding) + the gated ``OF3DiffusionModule`` to denoise the atom positions, then
applies the EDM update. The random per-step artefacts (``centre_random_augmentation``
rotation/translation, the additive noise) are replayed from a reference golden
(``scripts/of3_sample_diffusion_golden.py`` -> ``sample_diffusion_rollout_real``) so the
gate isolates the device conditioning+DiffusionModule precision from the random host
math -- the same isolation discipline as the other OF3 golden legs. The Fourier noise
embedding ``n = fourier_emb(0.25 * log(t / sigma_data))`` is computed on host
(``fourier_emb.w``/``b`` are in the checkpoint; the host computation is bit-exact vs the
reference, verified) and fed to the conditioning.

Topology (reference ``openfold3.core.model.structure.diffusion_module.SampleDiffusion``):

    xl = noise_schedule[0] * randn
    for tau, c_tau in noise_schedule[1:]:
        xl_aug = centre_random_augmentation(xl, atom_mask)          # replayed (rots, trans)
        t = noise_schedule[tau] * (gamma_0 + 1)                     # gamma_0 if c_tau > gamma_min
        noise = noise_scale * sqrt(t^2 - noise_schedule[tau]^2) * randn_like(xl)   # replayed
        xl_noisy = xl_aug + noise
        si, zij = DiffusionConditioning(si_input, si_trunk, zij_trunk, t)          # device, per step
        xl_denoised = DiffusionModule(xl_noisy, t, si, zij, ...)                   # device, per step
        delta = (xl_noisy - xl_denoised) / t
        xl = xl_noisy + step_scale * (c_tau - t) * delta

This is a reduced-step rollout gate (4 steps, 1 sample) -- it proves the EDM loop +
per-step conditioning compose on device. It is NOT the full ``fold()`` Kabsch merge
gate (the full production rollout is 200 steps x 5 samples = 1000 DiffusionModule
forwards; ``fold()`` additionally needs the trunk + confidence heads, not yet assembled
for inference -- see ``docs/openfold3-port.md``).
"""
from __future__ import annotations

import math

import torch
import ttnn

from .tenstorrent import CORE_GRID_MAIN
from .openfold3_diffusion import OF3DiffusionConditioning
from .openfold3_diffusion_module import OF3DiffusionModule
from .openfold3_weights import _sub


def fourier_noise_emb(t: float, sigma_data: float, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Host replication of OF3 ``FourierEmbedding``: ``n = 0.25*log(t/sigma_data)`` ->
    ``cos(2*pi*(n*w + b))`` -> [c_fourier_emb=256]. Bit-exact vs the reference
    (``fourier_emb.w``/``b`` come from the checkpoint)."""
    n = 0.25 * math.log(t / sigma_data)
    return torch.cos(2.0 * math.pi * (n * w + b)).float()           # [256]


class OF3SampleDiffusion:
    """Device EDM sampler loop around ``OF3DiffusionConditioning`` + ``OF3DiffusionModule``.

    Constructed once with the ``diffusion_module`` sub-dict + compute config + the host
    Fourier embedding buffers. ``__call__`` runs the full rollout, replaying the per-step
    ``(rots, trans, noise, t, c_tau)`` from the golden (host) and the fixed trunk /
    ref-atom / mask aux on device. Returns the final ``xl`` [1, n_atom, 3] on device.

    The fixed device inputs (si_trunk, si_input, zij_trunk, relpos, token_mask, cl0,
    plm0, atom masks, gather aux, atom_to_token_mean) are passed once; only the
    per-step ``(xl, t)`` evolve. The conditioned ``(si, zij)`` are recomputed each step
    (the Fourier noise embedding depends on ``t``).
    """

    def __init__(self, state_dict, compute_kernel_config, fourier_w: torch.Tensor,
                 fourier_b: torch.Tensor, sigma_data: float):
        # state_dict is already the diffusion_module sub-dict.
        self.dc = OF3DiffusionConditioning(_sub(state_dict, "diffusion_conditioning"),
                                           compute_kernel_config)
        self.dm = OF3DiffusionModule(state_dict, compute_kernel_config)
        self.fourier_w = fourier_w
        self.fourier_b = fourier_b
        self.sigma_data = sigma_data
        self.ckc = compute_kernel_config
        self.device = self.dc.device

    def _to_dev(self, x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(x, layout=layout, device=self.device, dtype=dtype)

    def __call__(self, xl_init_dev, si_trunk_dev, si_input_dev, zij_trunk_dev, relpos_dev,
                 token_mask_dev, pair_mask_dev, tok_mask_dev, cl0_dev, plm0_dev,
                 atom_mask_col_dev, atom_mask_col_na_dev, atom_to_token_idx_tt,
                 npe_flat_idx_tt, npe_zij_mask, enc_key_block_idxs_tt, enc_valid_mask,
                 enc_mask_bias, enc_pair_mask, atom_to_token_mean_tt,
                 token_mask_pad_tt, tok_mask_col_pad_tt,
                 n_atom, NP, nb, n_token, n_tok_pad,
                 noise_schedule, rots_list, trans_list, noise_list, t_list, c_tau_list,
                 step_scale):
        """Run the rollout. Per-step host artefacts (rots/trans/noise/t/c_tau) are
        python lists of host tensors/floats from the golden. Returns final xl [1, n_atom, 3] device."""
        atom_mask_host = ttnn.to_torch(atom_mask_col_na_dev).float().reshape(n_atom)  # [n_atom]
        xl_host = ttnn.to_torch(xl_init_dev).float().reshape(n_atom, 3)               # [n_atom, 3]

        for tau in range(len(t_list)):
            rots = rots_list[tau].float()                # [3, 3]
            trans = trans_list[tau].float()              # [3]
            # centre_random_augmentation (host): centre -> rotate -> translate -> mask.
            mean_xl = (xl_host * atom_mask_host[:, None]).sum(0) / atom_mask_host.sum().clamp_min(1.0)
            xl_aug = (xl_host - mean_xl) @ rots.t() + trans
            xl_aug = xl_aug * atom_mask_host[:, None]
            # noise add (host).
            noise = noise_list[tau].float()
            t = float(t_list[tau])
            xl_noisy = xl_aug + noise
            # per-step conditioning: host n_emb(t) -> device conditioning -> (si, zij).
            n_emb = fourier_noise_emb(t, self.sigma_data, self.fourier_w, self.fourier_b)
            si_dev, zij_dev = self.dc(zij_trunk_dev, relpos_dev, si_input_dev,
                                      si_trunk_dev, self._to_dev(n_emb.reshape(1, 1, 256)),
                                      pair_mask_dev, tok_mask_dev)
            # pad conditioned si/zij to n_tok_pad for the DiffusionModule.
            si_pad = self._pad_tokens(si_dev, n_token, n_tok_pad)
            zij_pad = self._pad_pair(zij_dev, n_token, n_tok_pad)
            ttnn.deallocate(si_dev); ttnn.deallocate(zij_dev)
            # rl_noisy = xl_noisy * atom_mask / sqrt(t^2 + sigma_data^2) (host -> device).
            rl_noisy = xl_noisy * atom_mask_host[:, None] / math.sqrt(t * t + self.sigma_data ** 2)
            rl_noisy_dev = self._to_dev(self._pad_atoms_host(rl_noisy, n_atom, NP))
            xl_noisy_masked = xl_noisy * atom_mask_host[:, None]
            xl_noisy_dev = self._to_dev(xl_noisy_masked.unsqueeze(0))  # [1, n_atom, 3]
            xl_denoised_dev = self.dm(
                si_trunk_dev, si_pad, zij_pad, cl0_dev, plm0_dev, rl_noisy_dev, xl_noisy_dev,
                atom_mask_col_dev, atom_mask_col_na_dev, atom_to_token_idx_tt,
                npe_flat_idx_tt, npe_zij_mask, enc_key_block_idxs_tt, enc_valid_mask,
                enc_mask_bias, enc_pair_mask, atom_to_token_mean_tt,
                token_mask_pad_tt, tok_mask_col_pad_tt,
                n_atom, NP, nb, n_token, n_tok_pad, t, self.sigma_data)
            xl_denoised = ttnn.to_torch(xl_denoised_dev).float().reshape(n_atom, 3)
            ttnn.deallocate(si_pad); ttnn.deallocate(zij_pad)
            ttnn.deallocate(rl_noisy_dev); ttnn.deallocate(xl_noisy_dev)
            ttnn.deallocate(xl_denoised_dev)
            # EDM step (host).
            delta = (xl_noisy - xl_denoised) / t
            dt = float(c_tau_list[tau]) - t
            xl_host = xl_noisy + step_scale * dt * delta

        return self._to_dev(xl_host.unsqueeze(0))

    @staticmethod
    def _pad_atoms_host(x, n_atom, NP):
        if NP > n_atom:
            x = torch.nn.functional.pad(x, (0, 0, 0, NP - n_atom))
        return x.unsqueeze(0)

    @staticmethod
    def _pad_tokens(x_dev, n_token, n_tok_pad):
        th = ttnn.to_torch(x_dev).float()
        if n_tok_pad > n_token:
            th = torch.nn.functional.pad(th, (0, 0, 0, n_tok_pad - n_token))
        return ttnn.from_torch(th, layout=ttnn.TILE_LAYOUT, device=x_dev.device(),
                               dtype=ttnn.bfloat16)

    @staticmethod
    def _pad_pair(x_dev, n_token, n_tok_pad):
        th = ttnn.to_torch(x_dev).float()
        if n_tok_pad > n_token:
            th = torch.nn.functional.pad(th, (0, 0, 0, n_tok_pad - n_token, 0, n_tok_pad - n_token))
        return ttnn.from_torch(th, layout=ttnn.TILE_LAYOUT, device=x_dev.device(),
                               dtype=ttnn.bfloat16)
