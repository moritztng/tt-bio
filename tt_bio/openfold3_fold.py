from __future__ import annotations

import math

import torch
import ttnn

from .tenstorrent import Module
from .openfold3_trunk import OF3Trunk
from .openfold3_sample_diffusion import OF3SampleDiffusion
from .openfold3_weights import _sub


def kabsch_rmsd(pred_ca, gt_ca):
    """Optimal-superposition Cα-RMSD (Kabsch). pred_ca, gt_ca: [N, 3].

    Both point clouds are centred, then aligned via the SVD of the correlation
    matrix with a reflection-correcting determinant, matching the
    ``scripts/release_gate.py`` Kabsch used by the other tt-bio model gates.
    """
    p = pred_ca.double() - pred_ca.double().mean(0)
    g = gt_ca.double() - gt_ca.double().mean(0)
    u, _, vt = torch.linalg.svd(p.t() @ g)
    d = torch.sign(torch.det(vt.t() @ u.t()))
    s = torch.eye(3, dtype=torch.float64)
    s[2, 2] = d
    p_aligned = p @ (vt.t() @ s @ u.t()).t()
    return float(torch.sqrt(((p_aligned - g) ** 2).sum(-1).mean()))


def load_pdb_ca(pdb_path):
    """Parse one Cα per residue from a PDB file -> [n_res, 3] float64."""
    pts, seen = [], set()
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            uid = (line[21].strip(), int(line[22:26].strip()))
            if uid in seen:
                continue
            seen.add(uid)
            pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return torch.tensor(pts, dtype=torch.float64)


# AF3 noise schedule (Page 24) and Algorithm 19 augmentation, replicated host-side
# so fold() can run a fresh rollout without the CPU reference package (the device env
# does not import openfold3). Bit-exact vs openfold3.core.model.structure.diffusion_module
# .create_noise_schedule and .augmentation.sample_rotations.

def create_noise_schedule(no_rollout_steps, sigma_data, s_max, s_min, p, dtype=torch.float32):
    t = torch.arange(0, 1 + no_rollout_steps, dtype=dtype) / no_rollout_steps
    return sigma_data * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p


def _quat_to_rot(q):
    # q: [4] normalized quaternion -> [3, 3]. Standard AF3 quat_to_rot via the
    # symmetric outer-product sum (equivalent to openfold3 ...rigid_utils.quat_to_rot).
    b, c, d, a = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c),
        2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b),
        2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d,
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def sample_rotation(dtype=torch.float32):
    q = torch.randn(4, dtype=dtype)
    q = q / torch.linalg.norm(q)
    return _quat_to_rot(q)                      # [3, 3]


def build_dm_device_aux(dev, ft, *, cl0, plm0, atom_mask, atom_to_token_index,
                        npe_q_indices, npe_k_indices, zij_mask, key_block_idxs,
                        invalid_mask, mask_trunked, atom_to_token_mean,
                        token_mask, n_atom, n_token, nb, NP, n_tok_pad):
    """Mirror the device DiffusionModule aux setup validated in
    tests/test_openfold3_sample_diffusion.py (the exact tensor shapes/dtypes the
    gated OF3DiffusionModule consumes), so fold() feeds the sampler identically."""
    cl0_t = torch.zeros(1, NP, 128); cl0_t[0, :n_atom] = cl0.float()
    amc = torch.zeros(1, NP, 1); amc[0, :n_atom, 0] = atom_mask.float()
    amc_na = torch.zeros(1, n_atom, 1); amc_na[0, :, 0] = atom_mask.float()
    idx = torch.zeros(NP, dtype=torch.long); idx[:n_atom] = atom_to_token_index.long()
    idx_tt = ttnn.from_torch(idx.unsqueeze(0), layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=dev, dtype=ttnn.uint32)
    flat = (npe_q_indices.unsqueeze(-1) * n_tok_pad + npe_k_indices.unsqueeze(1)).reshape(1, nb * 32 * 128)
    flat_tt = ttnn.from_torch(flat.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=dev, dtype=ttnn.uint32)
    kidx = key_block_idxs.reshape(1, nb * 128).to(torch.int32)
    kidx_tt = ttnn.from_torch(kidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    valid = (~invalid_mask).float().reshape(1, nb, 128, 1)
    mask_bias = (1e9 * (mask_trunked - 1)).reshape(1, nb, 1, 32, 128)
    pair_mask_m = mask_trunked.reshape(1, nb, 32, 128, 1)
    tok_pad = torch.zeros(n_tok_pad, dtype=torch.float32); tok_pad[:n_token] = token_mask.float()
    return dict(
        cl0_d=ft(cl0_t), plm0_d=ft(plm0.unsqueeze(0)),
        amc_d=ft(amc), amc_na_d=ft(amc_na),
        idx_tt=idx_tt, flat_tt=flat_tt,
        zij_mask_d=ft(zij_mask.unsqueeze(0).unsqueeze(-1)),
        kidx_tt=kidx_tt, valid_d=ft(valid), mb_d=ft(mask_bias), pm_d=ft(pair_mask_m),
        mean_d=ft(atom_to_token_mean.unsqueeze(0)),
        tok_pad_tt=ft(tok_pad.reshape(1, n_tok_pad)),
        tok_col_pad_tt=ft(tok_pad.reshape(1, n_tok_pad, 1)),
    )


class OpenFold3(Module):
    """End-to-end OpenFold3 ``fold()`` on device: trunk -> fresh-rollout SampleDiffusion
    -> atom coordinates.

    Assembles the gated device components into a single forward. The trunk
    (``OF3Trunk``, P10) runs fully on device; its ``(s_trunk, z_trunk)`` feed the
    gated device ``OF3SampleDiffusion`` (P9 leg 2), which runs a *fresh* EDM rollout
    (AF3 Algorithm 18) -- a real noise schedule + per-step random augmentation/noise
    drawn with the caller's seed, NOT a golden-replayed trajectory -- to denoise atom
    positions. Returns ``xl_final`` [n_atom, 3] on host.

    Scope (documented honestly): the InputEmbedder atom-encoder / MSAModuleEmbedder /
    template-feature host-prep legs run on host (their device ports exist and are
    separately PCC-gated, but wiring the full device atom-transformer into fold() is
    deferred); the trunk + diffusion sampler -- the device compute that determines the
    structure -- run on device for real. Confidence heads are optional and not run
    here (the merge gate is the Cα-RMSD, computed by the caller against ground truth).

    Args:
        sd: the full OF3 checkpoint state dict.
        compute_kernel_config: HiFi4 + fp32 dest acc.
        num_cycles: trunk recycle cycles (OF3 default 4 = num_recycles+1).
    """

    def __init__(self, sd, compute_kernel_config, num_cycles: int = 4):
        super().__init__(sd, compute_kernel_config)
        self.sd = sd
        self.ckc = compute_kernel_config
        self.trunk = OF3Trunk(sd, compute_kernel_config, num_cycles=num_cycles)
        fourier_w = sd["diffusion_module.diffusion_conditioning.fourier_emb.w"]
        fourier_b = sd["diffusion_module.diffusion_conditioning.fourier_emb.b"]
        # sigma_data lives under diffusion_module.diffusion_module in the checkpoint.
        sigma_data = float(sd.get("diffusion_module.diffusion_module.sigma_data", 16.0))
        self.sigma_data = sigma_data
        self.sampler = OF3SampleDiffusion(_sub(sd, "diffusion_module"), compute_kernel_config,
                                          fourier_w, fourier_b, sigma_data)
        self.device = self.trunk.device
        # AF3 sample_diffusion + noise_schedule defaults (OF3 model_config).
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5
        self.ns_cfg = dict(sigma_data=sigma_data, s_max=160.0, s_min=4e-4, p=7)

    def _ft(self, x, dtype=ttnn.bfloat16):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=self.device,
                               dtype=dtype)

    def _gen_rollout(self, noise_schedule, n_atom, seed):
        """Fresh per-step (rots, trans, noise, t, c_tau) + xl_init for one sample,
        matching AF3 Algorithm 18/19 RNG call order (rots -> trans -> noise)."""
        torch.manual_seed(seed)
        xl_init = noise_schedule[0] * torch.randn(n_atom, 3, dtype=torch.float32)
        rots_l, trans_l, noise_l, t_l, c_tau_l = [], [], [], [], []
        for tau in range(len(noise_schedule) - 1):
            c_tau = float(noise_schedule[tau + 1])
            gamma = self.gamma_0 if c_tau > self.gamma_min else 0
            t = float(noise_schedule[tau]) * (gamma + 1)
            rots = sample_rotation()                              # [3, 3]
            trans = 1.0 * torch.randn(3, dtype=torch.float32)     # [3]
            noise = self.noise_scale * math.sqrt(max(t * t - float(noise_schedule[tau]) ** 2, 0.0)) \
                * torch.randn(n_atom, 3, dtype=torch.float32)
            rots_l.append(rots); trans_l.append(trans); noise_l.append(noise)
            t_l.append(t); c_tau_l.append(c_tau)
        return xl_init, rots_l, trans_l, noise_l, t_l, c_tau_l

    def fold(self, *, s_init, z_init, template_feat, msa_feat, s_input,
             si_input, relpos, token_mask, dm_aux_host, n_atom, n_token,
             no_rollout_steps, seed, no_samples=1):
        """Run one trunk forward + ``no_samples`` fresh EDM rollouts.

        Host inputs are the real ubiquitin features (P1 data pipeline + reference
        embedder outputs captured in ~/of3_ref_out.pkl -- the existing OF3 fixture, to
        avoid redundant data-pipeline work). ``dm_aux_host`` is the dict of host
        DiffusionModule aux tensors (cl0/plm0/masks/block indices/atom_to_token_mean).

        Returns a list of ``no_samples`` host ``xl_final`` [n_atom, 3] float32 tensors.
        """
        ft = self._ft
        nb = dm_aux_host["nb"]; NP = dm_aux_host["NP"]
        n_tok_pad = ((n_token + 31) // 32) * 32

        # --- device trunk ---
        s_init_d = ft(s_init.unsqueeze(0))
        z_init_d = ft(z_init.unsqueeze(0))
        tmpl_d = {k: ft(v) for k, v in template_feat.items()}
        msa_d = ft(msa_feat.unsqueeze(0))
        s_input_d = ft(s_input.unsqueeze(0))
        s_trunk_d, z_trunk_d = self.trunk(s_init_d, z_init_d, tmpl_d, msa_d, s_input_d)
        si_trunk_d = s_trunk_d
        zij_trunk_d = z_trunk_d

        # --- fixed conditioning inputs ---
        zij_trunk_dev = zij_trunk_d
        relpos_dev = ft(relpos.unsqueeze(0))
        si_input_dev = ft(si_input.unsqueeze(0))
        n_tok = token_mask.shape[0]
        pair_mask = (token_mask[:, None] * token_mask[None, :]).reshape(n_tok, n_tok, 1).unsqueeze(0)
        tok_mask = token_mask.reshape(n_tok, 1).unsqueeze(0)
        pair_mask_dev, tok_mask_dev = ft(pair_mask), ft(tok_mask)
        tm_dev = ft(token_mask.reshape(1, n_tok))

        aux = build_dm_device_aux(
            self.device, ft,
            cl0=dm_aux_host["cl0"], plm0=dm_aux_host["plm0"],
            atom_mask=dm_aux_host["atom_mask"],
            atom_to_token_index=dm_aux_host["atom_to_token_index"],
            npe_q_indices=dm_aux_host["npe_q_indices"],
            npe_k_indices=dm_aux_host["npe_k_indices"],
            zij_mask=dm_aux_host["zij_mask"],
            key_block_idxs=dm_aux_host["key_block_idxs"],
            invalid_mask=dm_aux_host["invalid_mask"],
            mask_trunked=dm_aux_host["mask_trunked"],
            atom_to_token_mean=dm_aux_host["atom_to_token_mean"],
            token_mask=token_mask, n_atom=n_atom, n_token=n_token, nb=nb, NP=NP,
            n_tok_pad=n_tok_pad)

        noise_schedule = create_noise_schedule(no_rollout_steps, **self.ns_cfg)

        samples = []
        for s in range(no_samples):
            xl_init, rots_l, trans_l, noise_l, t_l, c_tau_l = self._gen_rollout(
                noise_schedule, n_atom, seed + s)
            xl_init_dev = ft(xl_init.unsqueeze(0))
            xl_final_dev = self.sampler(
                xl_init_dev, si_trunk_d, si_input_dev, zij_trunk_dev, relpos_dev,
                tm_dev, pair_mask_dev, tok_mask_dev,
                aux["cl0_d"], aux["plm0_d"], aux["amc_d"], aux["amc_na_d"],
                aux["idx_tt"], aux["flat_tt"], aux["zij_mask_d"], aux["kidx_tt"],
                aux["valid_d"], aux["mb_d"], aux["pm_d"], aux["mean_d"],
                aux["tok_pad_tt"], aux["tok_col_pad_tt"],
                n_atom, NP, nb, n_token, n_tok_pad,
                noise_schedule, rots_l, trans_l, noise_l, t_l, c_tau_l,
                self.step_scale)
            xl_final = torch.Tensor(ttnn.to_torch(xl_final_dev)).float().reshape(n_atom, 3)
            ttnn.deallocate(xl_init_dev)
            ttnn.deallocate(xl_final_dev)
            samples.append(xl_final)
            print(f"  [fold] sample {s}: xl_final std={float(xl_final.std()):.4f} "
                  f"range=[{float(xl_final.min()):.2f},{float(xl_final.max()):.2f}]")
        return samples
