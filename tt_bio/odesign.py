"""ODesign (AF3-family all-atom binder design) on Tenstorrent -- inference port.

ODesign (github.com/OTeam-AI4S/ODesign, Apache-2.0) is the Protenix-v2 / AF3 family:
trunk Pairformer (c_s=384, c_z=128, 48 blocks, 16 heads, c_s_inputs=453) + diffusion DiT
(c_token=768, 24 blocks, 16 heads) + 3-block atom enc/dec (c_atom=128, c_atompair=16),
EDM sampler (200 steps x N_sample, N_cycle=10, bf16). The genuinely-new pieces vs
Protenix-v2 are: (a) the constraint/hotspot distogram embedder (2 Pairformer-conditioned
blocks, c=64) in the trunk, (b) the OInvFold inverse-folding head, (c) the `design` job
path. The default `prot_binding_prot` config loads ProteinMPNN (already ported in
tt_bio.proteinmpnn) for the sequence leg, so OInvFold is a stretch goal.

Pass-3 scope = the DENOISER-PARITY leg only. The ODesign DiffusionModule is structurally
identical to Protenix-v2's (same c_token=768 / 24 blocks / 16 heads / head_dim=48 /
c_atom=128; only c_z=128 vs 256 and c_s_inputs=453 vs 449 differ, both absorbed by
weight-driven linears), so this module REUSES tt_bio.protenix.DiffusionModule +
AtomFeaturization loaded with ODesign weights. The single behavioral diff in the atom
featurization is ref_element being 129-dim (vs 128); the cond builder here applies that.
Trunk conditioning (s_trunk / s_inputs / z_trunk) is taken from the captured golden
intermediates, so the denoiser is validated independently of the (unported) trunk.

DEFERRED (not built this pass, not parity-claimed): the ODesign trunk (PairformerStack +
MSAModule + the constraint/hotspot distogram embedder), the OInvFold head, the
`tt-bio design --model odesign` CLI, per-model docs, and warm single-card throughput.
The constraint embedder is only on the trunk path; it is NOT exercised by the denoiser
parity replay (which consumes precomputed trunk conditioning), so its absence does not
affect the parity number reported here.
"""
import os
import random as _random
import numpy as np
import torch
import ttnn
from scipy.spatial.transform import Rotation

from .tenstorrent import get_device, CORE_GRID_MAIN
from .protenix import DiffusionModule, AtomFeaturization, AtomAttentionEncoder, _window_q, _window_kv


# ---------------------------------------------------------------------------
# ODesign sampler primitives (new port code). These mirror ODesign's
# src/utils/model/misc.py (centre_random_augmentation / reverse /
# uniform_random_rotation / rot_vec_mul) and src/model/modules/schedulers.py
# (InferenceNoiseEDMScheduler EDM math). Kept on host fp32 -- only the
# per-step denoise hits the device. The augmentation + EDM update + condition
# enforcement are the ODesign-specific pieces the denoiser-leg replay (pass 4)
# did not exercise.
# ---------------------------------------------------------------------------

def _append_dims(x, ndim):
    while x.ndim < ndim:
        x = x.unsqueeze(-1)
    return x


def _rot_vec_mul(r, t):
    """Apply rotation matrices r (...,3,3) to vectors t (...,3). Mirrors ODesign
    rot_vec_mul (fp32, hand-written to avoid AMP downcast)."""
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    if r.dtype != torch.float32:
        r = r.to(torch.float32)
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack([
        r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
        r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
        r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
    ], dim=-1)


def centre_random_augmentation(x_input_coords, n_sample=1, s_trans=1.0, dtype=torch.float32,
                               rot=None, trans=None):
    """Algorithm 19 (AF3) centre+SE(3) augmentation. Mirrors ODesign
    centre_random_augmentation with mask=None (inference passes no mask). Takes
    x_input_coords as (N_atom, 3) [no batch dim] and returns (n_sample, N_atom, 3).
    When `rot`/`trans` are supplied (precomputed draws), uses them instead of drawing
    new ones -- so the closed-loop replay shares an identical noise realization
    between the on-device and CPU-fp32 reference samplers. Returns
    (x_aug, trans, rot, x_center) on host fp32."""
    n_atom = x_input_coords.size(-2)
    x_center = x_input_coords.mean(dim=-2, keepdim=True)   # (1,3)
    x = x_input_coords - x_center                              # (N_atom,3)
    if rot is None:
        rot = uniform_random_rotation(n_sample)              # (n_sample,3,3)
    if trans is None:
        trans = s_trans * torch.randn(n_sample, 3)          # (n_sample,3)
    # rot (s,3,3) @ x (N_atom,3) -> (n_sample,N_atom,3); add per-sample trans
    x_aug = torch.einsum("sij,nj->sni", rot, x) + trans.unsqueeze(-2)
    return x_aug.to(dtype), trans.to(dtype), rot.to(dtype), x_center.to(dtype)


def reverse_centre_random_augmentation(x_aug, trans, rot, x_center):
    """Inverse of centre_random_augmentation. x_aug (n_sample,N_atom,3); returns
    (n_sample,N_atom,3) in the original frame. Mirrors ODesign reverse_centre_random_augmentation."""
    x = x_aug - trans.unsqueeze(-2)                        # (n_sample,N_atom,3)
    # apply rot^T: out[s,n,i] = sum_j rot[s,i,j] * x[s,n,j]
    x = torch.einsum("sij,snj->sni", rot, x)
    x = x + x_center.unsqueeze(-2)                          # (1,3) -> (1,1,3) broadcast
    return x.view_as(x_aug)


def uniform_random_rotation(n_sample=1):
    """Random rotation matrices via scipy Rotation.random (numpy RNG). Mirrors
    ODesign uniform_random_rotation. Returns (n_sample,3,3) fp32 torch tensor."""
    return torch.from_numpy(Rotation.random(num=n_sample).as_matrix()).float()


def edm_noise_schedule(n_step=200, s_max=160.0, s_min=4e-4, rho=7, sigma_data=16.0,
                       dtype=torch.float32):
    """InferenceNoiseEDMScheduler.set_noise_schedule: power-law sigma schedule,
    sigma(N_step)=0. Returns tensor [N_step+1]."""
    step = 1.0 / n_step
    idx = torch.arange(n_step + 1, dtype=dtype)
    t = sigma_data * (s_max ** (1.0 / rho) + idx * step * (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))) ** rho
    t[-1] = 0.0
    return t


def edm_step_params(schedule, step_idx, n_sample=1, use_pc=True, gamma0=0.8, gamma_min=1.0):
    """get_noise_level: returns (t_hat, c_tau, c_tau_last). t_hat = c_tau_last*(1+gamma),
    gamma=gamma0 if c_tau>gamma_min else 0."""
    c_tau_last = schedule[step_idx]
    c_tau = schedule[step_idx + 1]
    if use_pc:
        gamma = float(gamma0) if float(c_tau) > float(gamma_min) else 0.0
        t_hat = c_tau_last * (gamma + 1.0)
    else:
        t_hat = c_tau_last
    return t_hat.expand(n_sample), c_tau.expand(n_sample), c_tau_last.expand(n_sample)


def add_noise_with_condition(x_l, condition_mask, t_hat, sigma_data=16.0,
                             noise_scale_lambda=1.003, use_pc=True, c_tau_last=None,
                             noise=None):
    """InferenceNoiseEDMScheduler.add_noise_with_condition: corrector noise +
    c_in scaling. `noise` (precomputed draw) is used if supplied. Returns
    x_noisy = c_in*(x_l + lambda*delta_sigma*noise), with zero noise on cond atoms."""
    if use_pc:
        delta = torch.sqrt(t_hat ** 2 - c_tau_last ** 2)
        if noise is None:
            noise = torch.where(_append_dims(condition_mask, x_l.ndim),
                                torch.zeros_like(x_l), torch.randn_like(x_l))
        else:
            noise = torch.where(_append_dims(condition_mask, x_l.ndim),
                                torch.zeros_like(x_l), noise)
        x_noisy = x_l + noise_scale_lambda * delta * noise
    else:
        x_noisy = x_l
    c_in = 1.0 / torch.sqrt(sigma_data ** 2 + t_hat ** 2)
    return c_in * x_noisy


def update_with_condition(x_noisy, x_update, x_gt, condition_mask, t_hat, c_tau,
                          sigma_data=16.0, step_scale_eta=1.5):
    """InferenceNoiseEDMScheduler.update_with_condition: EDM denoise + Euler step +
    condition enforcement. Mirrors ODesign schedulers.py L514-592."""
    sigma = _append_dims(t_hat, x_noisy.ndim)
    s_ratio = sigma / sigma_data
    x_noisy_unscaled = x_noisy * torch.sqrt(sigma_data ** 2 + sigma ** 2)
    x_denoised = (1.0 / (1.0 + s_ratio ** 2)) * x_noisy_unscaled \
        + (sigma / torch.sqrt(1.0 + s_ratio ** 2)) * x_update
    delta = (x_noisy_unscaled - x_denoised) / sigma
    dt = c_tau - t_hat
    x_l = x_noisy_unscaled + step_scale_eta * dt * delta
    x_l = torch.where(_append_dims(condition_mask, x_l.ndim), x_gt, x_l)
    return x_l


def generate_sampler_draws(n_atom, n_step, seed=42, condition_mask=None,
                           s_max=160.0, sigma_data=16.0, dtype=torch.float32):
    """Generate the full set of random draws for one closed-loop sample, in the
    exact call order ODesign's sampler consumes them (seed_everything ->
    init randn -> per step: Rotation.random, trans randn, corrector randn_like).
    Seeding random + numpy + torch CPU. Returns a dict of host fp32 tensors so
    the on-device and CPU-fp32 reference samplers consume an IDENTICAL noise
    realization (isolating the device bf16 compounding). NOTE: the original
    golden was drawn from CUDA's Philox RNG, which is not reproducible on a
    CPU-only/Tenstorrent box -- so these draws match the golden's numpy
    (Rotation) stream but NOT its torch.randn (CUDA) stream; see pass-5 notes."""
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if condition_mask is None:
        condition_mask = torch.zeros(n_atom, dtype=torch.bool)
    condition_mask = condition_mask.unsqueeze(0)   # (1,N_atom) -- matches ODesign's (N_sample,N_atom)
    schedule = edm_noise_schedule(n_step, s_max=s_max, sigma_data=sigma_data, dtype=dtype)
    # init noise: s_max*randn for free atoms, x_gt (zeros here, no cond atoms) for cond
    init = schedule[0] * torch.randn(1, n_atom, 3, dtype=dtype)
    init = torch.where(_append_dims(condition_mask, init.ndim),
                       torch.zeros_like(init), init)
    rots, transes, noises = [], [], []
    for i in range(n_step):
        t_hat, c_tau, c_tau_last = edm_step_params(schedule, i, n_sample=1)
        rot = uniform_random_rotation(1)                       # numpy draw (matches golden)
        trans = 1.0 * torch.randn(1, 3, dtype=dtype)           # torch draw (CPU vs CUDA)
        noise = torch.randn_like(init)                         # torch draw (CPU vs CUDA)
        rots.append(rot); transes.append(trans); noises.append(noise)
    return {"seed": seed, "schedule": schedule, "init_noise": init,
            "rots": rots, "trans": transes, "noises": noises,
            "condition_mask": condition_mask}
from . import protenix_weights as PW


# ---------------------------------------------------------------------------
# ODesign trunk-input port (pass 6). The trunk is mostly Protenix-v2's
# (tt_bio.protenix.Trunk, c_z=128, no-template path) but with THREE genuinely
# new ODesign pieces this pass ports + parity-verifies:
#   (1) RelativePositionEncoding with the cyclic-peptide offset logic
#       (ODesign embedders.py L158-204; Protenix-v2's RPE has no cyclic branch).
#   (2) The trunk-init linears (s_init / z_init) -- weight-driven, c_s_inputs=453.
#   (3) ConstraintTemplateEmbedder FRONT-END: distogram binning (39 bins,
#       min_bin=3.25, max_bin=50.75) + v_ij = LinearNoBias_z(LN_z(z)) +
#       LinearNoBias_a(distogram). The 2-block PairformerStack INSIDE the
#       constraint embedder (triangle attention, 4 heads x head_dim=32) is
#       deferred to pass 7 -- it needs a dedicated pair-only Pairformer port
#       matching ODesign's head geometry.
# Parity vs ODesign's own modules run fresh on CPU (same methodology as
# passes 1-5): scripts/odesign_trunk_ref.py dumps the CPU reference,
# scripts/odesign_trunk_parity.py diffs the ttnn port against it.
# ---------------------------------------------------------------------------

class ODesignTrunkInput:
    """ODesign trunk-input construction on Tenstorrent (pass 6): the
    RelativePositionEncoding (with cyclic-peptide offset), the trunk-init
    linears (s_init / z_init), and the ConstraintTemplateEmbedder front-end
    (distogram binning + v_ij projection). Loaded from the full ODesign
    checkpoint state dict (relative_position_encoding.*, linear_no_bias_sinit,
    linear_no_bias_zinit1/2, linear_no_bias_token_bond, constraint_distogram_embedder.*).

    The 2-block PairformerStack inside the constraint embedder is deferred to
    pass 7; this class exposes the front-end (v_ij) only. The full trunk
    (48-block PairformerStack + MSAModule + the constraint embedder's
    PairformerStack) is also deferred -- pass 6 lands the clean, self-contained,
    parity-verified entry slice.
    """

    R_MAX, S_MAX, C_Z, C_S, C_S_INPUTS = 32, 2, 128, 384, 453
    CTE_C, CTE_NO_BINS, CTE_MIN_BIN, CTE_MAX_BIN = 64, 39, 3.25, 50.75

    def __init__(self, model_state_dict, compute_kernel_config, device=None):
        self._w = model_state_dict
        self.compute_kernel_config = compute_kernel_config
        self.dev = device or get_device()
        self._wc = {}  # device weight cache (uploaded once, reused)
        self._aae = None  # lazy: protenix.AtomAttentionEncoder (input_embedder weights)

    # --- upload helpers (mirror ODesign._tt/_lin/_ln; cached here) ---
    def _tt(self, x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)

    def _w_tt(self, key, transpose=True):
        v = self._wc.get((key, transpose))
        if v is None:
            w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)
            self._wc[(key, transpose)] = v
        return v

    @staticmethod
    def _to_host(t, shape=None):
        h = torch.Tensor(ttnn.to_torch(t)).float()
        return h.reshape(shape) if shape is not None else h

    def _lin(self, x, wkey, bkey=None):
        return ttnn.linear(x, self._w_tt(wkey), bias=(self._w_tt(bkey, False) if bkey else None),
                           compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)

    def _ln(self, x, wkey, bkey):
        return ttnn.layer_norm(x, weight=self._w_tt(wkey, False), bias=self._w_tt(bkey, False),
                               epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)

    # --- (1) RelativePositionEncoding with cyclic-peptide offset ---
    @staticmethod
    def relative_position_encoding(feat, r_max=32, s_max=2):
        """Host fp32 one-hot RPE feature (ODesign embedders.py L144-226),
        INCLUDING the cyclic-peptide offset branch (L158-197) that Protenix-v2's
        RPE lacks. feat needs asym_id, residue_index, entity_id, sym_id,
        token_index, is_cyclic_token. Returns (N, N, 4*r_max + 2*s_max + 7) =
        (N, N, 139) host fp32. The trailing linear is applied on device."""
        import torch.nn.functional as F
        asym = feat["asym_id"].long(); res = feat["residue_index"].long()
        ent = feat["entity_id"].long(); tok = feat["token_index"].long()
        sym = feat["sym_id"].long()
        b_same_chain = (asym[:, None] == asym[None, :]).long()
        b_same_residue = (res[:, None] == res[None, :]).long()
        b_same_entity = (ent[:, None] == ent[None, :]).long()
        offset = res[:, None] - res[None, :]                       # cyclic-peptide offset logic
        is_cyclic_token = feat.get("is_cyclic_token")
        if is_cyclic_token is not None and bool(is_cyclic_token.any()):
            for asym_id_val in torch.unique(asym):
                asym_mask = (asym == asym_id_val)
                if asym_mask.any() and bool(is_cyclic_token[asym_mask][0]):
                    len_chain = int(asym_mask.sum())
                    if len_chain > 0:
                        chain_mask = ((asym[:, None] == asym_id_val) &
                                      (asym[None, :] == asym_id_val))
                        offset_plus = offset + len_chain
                        offset_minus = offset - len_chain
                        abs_offset = offset.abs()
                        abs_offset_plus = offset_plus.abs()
                        abs_offset_minus = offset_minus.abs()
                        choice = torch.where(abs_offset_plus <= abs_offset_minus, offset_plus, offset_minus)
                        c_offset = torch.where(
                            (abs_offset <= abs_offset_plus) & (abs_offset <= abs_offset_minus),
                            offset, choice)
                        offset = torch.where(chain_mask, c_offset, offset)
        offset = (offset + r_max).clamp(0, 2 * r_max)
        d_residue = torch.where(b_same_chain.bool(), offset,
                                (2 * r_max + 1) * torch.ones_like(offset))
        a_rel_pos = F.one_hot(d_residue, 2 * (r_max + 1))
        d_token = torch.clip(tok[:, None] - tok[None, :] + r_max, 0, 2 * r_max) \
            * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (2 * r_max + 1)
        a_rel_token = F.one_hot(d_token, 2 * (r_max + 1))
        d_chain = torch.clip(sym[:, None] - sym[None, :] + s_max, 0, 2 * s_max) \
            * b_same_entity + (1 - b_same_entity) * (2 * s_max + 1)
        a_rel_chain = F.one_hot(d_chain, 2 * (s_max + 1))
        return torch.cat([a_rel_pos, a_rel_token, b_same_entity[..., None],
                          a_rel_chain], dim=-1).float()

    def rpe(self, feat):
        """Full RPE: one-hot feature (host) -> LinearNoBias (device). Returns
        (N, N, c_z) host fp32."""
        relp = self.relative_position_encoding(feat, self.R_MAX, self.S_MAX)
        out = self._lin(self._tt(relp), "relative_position_encoding.linear_no_bias.weight")
        return self._to_host(out, (relp.shape[0], relp.shape[1], self.C_Z))

    # --- (2) trunk-init linears (s_init, z_init) ---
    def trunk_init(self, s_inputs, relp, token_bonds):
        """s_init = LinearNoBias_sinit(s_inputs); z_init = zinit1(s_init)[:,None] +
        zinit2(s_init)[None,:] + relp + LinearNoBias_token_bond(token_bonds[...,None]).
        s_inputs (N,453) host, relp (N,N,128) host, token_bonds (N,N) host.
        Returns (s_init (N,384), z_init (N,N,128)) host fp32."""
        N = s_inputs.shape[0]
        s_init = self._lin(self._tt(s_inputs.float()), "linear_no_bias_sinit.weight")
        cz = self._w["linear_no_bias_zinit1.weight"].shape[0]
        z1 = ttnn.reshape(self._lin(s_init, "linear_no_bias_zinit1.weight"), (N, 1, cz))
        z2 = ttnn.reshape(self._lin(s_init, "linear_no_bias_zinit2.weight"), (1, N, cz))
        z = ttnn.add(z1, z2)
        z = ttnn.add(z, self._tt(relp.float()))
        z = ttnn.add(z, self._lin(self._tt(token_bonds.unsqueeze(-1).float()),
                                  "linear_no_bias_token_bond.weight"))
        return self._to_host(s_init, (N, self.C_S)), self._to_host(z, (N, N, cz))

    # --- (3) ConstraintTemplateEmbedder front-end ---
    def constraint_distogram(self, constraint_feature):
        """Bin a per-pair distance map (N,N,1) into a 39-bin one-hot distogram
        (N,N,39), matching ODesign ConstraintTemplateEmbedder.forward L1548-1558
        (boundaries = linspace(min_bin, max_bin, no_bins-1); true_bins =
        sum(cf > boundaries, dim=-1); one_hot(true_bins, no_bins)). Host fp32."""
        import torch.nn.functional as F
        boundaries = torch.linspace(self.CTE_MIN_BIN, self.CTE_MAX_BIN, self.CTE_NO_BINS - 1)
        cf = constraint_feature.float()
        true_bins = torch.sum(cf > boundaries, dim=-1)            # (N,N) in [0, no_bins-1]
        return F.one_hot(true_bins, self.CTE_NO_BINS).float()     # (N,N,39)

    def constraint_embedder_front(self, z, constraint_feature):
        """ConstraintTemplateEmbedder FRONT-END (the 2-block PairformerStack
        INSIDE it is deferred to pass 7). v_ij = LinearNoBias_z(LN_z(z)) +
        LinearNoBias_a(distogram). z (N,N,128) host, constraint_feature (N,N,1)
        host. Returns v_ij (N,N,64) host fp32 -- the input to the deferred
        2-block PairformerStack."""
        C = "constraint_distogram_embedder."
        distogram = self.constraint_distogram(constraint_feature)  # (N,N,39) host
        zc = self._ln(self._tt(z.float()), C + "layernorm_z.weight", C + "layernorm_z.bias")
        zt = self._lin(zc, C + "linear_no_bias_z.weight")          # (N,N,64)
        at = self._lin(self._tt(distogram), C + "linear_no_bias_a.weight")  # (N,N,64)
        v = ttnn.add(zt, at)
        return self._to_host(v, (z.shape[0], z.shape[1], self.CTE_C))

    # --- (4) InputFeatureEmbedder (atom encoder -> s_inputs) ---
    def input_feature_embedder(self, feat):
        """ODesign InputFeatureEmbedder: atom encoder -> `a` (N,384), then
        s_inputs = cat([a, restype(35), profile(32), deletion_mean(1), is_hotspot_residue(1)])
        -> (N, 453). REUSES tt_bio.protenix.AtomAttentionEncoder (has_coords=False) with
        ODesign's `input_embedder.*` weights -- the atom encoder is structurally
        identical to Protenix-v2's; the only ODesign diff is the 129-dim ref_element in
        f_in (vs 128; f_in = 1+129+256 = 386; the linear_no_bias_f weight shape (128, 386)
        absorbs it) and the +is_hotspot_residue(1) concat (453 vs 449). The protenix encoder
        returns cat([a, restype, profile, deletion_mean]) = 452 for ODesign's 35-dim
        restype; we append is_hotspot_residue(1) -> 453. feat needs: ref_pos, ref_charge,
        ref_mask, ref_element(129), ref_atom_name_chars, ref_space_uid, atom_to_token_idx,
        restype(35), profile(32), deletion_mean(1), is_hotspot_residue(1). Returns
        s_inputs (N, 453) host fp32."""
        if self._aae is None:
            self._aae = AtomAttentionEncoder(
                {k[len("input_embedder.atom_attention_encoder."):]: v
                 for k, v in self._w.items() if k.startswith("input_embedder.atom_attention_encoder.")},
                self.compute_kernel_config)
        N = feat["ref_pos"].shape[0]
        # f_in: cat([ref_mask(1), ref_element(129), ref_atom_name_chars(256)]) = 386 (ODesign)
        f_in = torch.cat([feat["ref_mask"].reshape(N, 1),
                          feat["ref_element"].reshape(N, 129),
                          feat["ref_atom_name_chars"].reshape(N, 256)], dim=-1).float()
        d_lm, v_lm, mt = ODesign._atom_pair_feats(feat["ref_pos"].float(), feat["ref_space_uid"])
        nb, nq, nk, _ = d_lm.shape
        M = nb * nq * nk
        d = d_lm.reshape(M, 3); v = v_lm.reshape(M, 1)
        invd = (1.0 / (1.0 + (d_lm ** 2).sum(-1, keepdim=True))).reshape(M, 1)
        a2t = feat["atom_to_token_idx"].long(); NT = int(a2t.max()) + 1
        S = torch.zeros(N, NT); S[torch.arange(N), a2t] = 1.0
        Mmat = (S.t() / (S.t().sum(-1, keepdim=True) + 1e-6))
        ref_charge_asinh = torch.arcsinh(feat["ref_charge"]).reshape(N, 1).float()
        tt = self._tt
        s452 = self._aae(
            tt(feat["ref_pos"].float()), tt(ref_charge_asinh),
            tt(feat["ref_mask"].reshape(N, 1).float()), tt(f_in),
            tt(d), tt(v), tt(invd), mt, tt(Mmat),
            tt(feat["restype"].float()), tt(feat["profile"].float()),
            tt(feat["deletion_mean"].reshape(-1, 1).float()))
        s452 = self._to_host(s452)[:NT]
        # append is_hotspot_residue (1) -> 453
        s_inputs = torch.cat([s452, feat["is_hotspot_residue"].reshape(NT, 1).float()], dim=-1)
        return s_inputs


# ---------------------------------------------------------------------------
# ODesign FULL TRUNK port (pass 7). Wires the 48-block PairformerStack (main
# trunk), the ConstraintTemplateEmbedder's 2-block pair-only PairformerStack, and
# the 4-block MSA module -- all reusing the already-ported protenix/tenstorrent
# primitives (Pairformer, PairformerLayer, Transition, OuterProductMean,
# PairWeightedAveraging) with ODesign weights (tt_bio.protenix_weights remaps).
# The front-end (RPE w/ cyclic offset, trunk-init linears, CTE front-end,
# InputFeatureEmbedder) reuses the pass-6 `ODesignTrunkInput`.
#
# ODesign's trunk cycle (src/model/odesign.py get_pairformer_output L315-382):
#   for cyc in range(N_cycle):
#     z = z_init + linear_no_bias_z_cycle(LN_z_cycle(z))
#     z += constraint_distogram_embedder(input_data, z)     # CTE (c=64, 2 pair-only blocks)
#     z  = msa_module(input_data, z, s_inputs)             # 4-block MSA (single-seq self-MSA)
#     s  = s_init + linear_no_bias_s(LN_s(s))
#     s, z = pairformer_stack(s, z)                        # 48-block with-single
# Parity vs the CPU-fp32 reference (ODesign's own trunk run fresh): see
# scripts/odesign_trunk_full_parity.py.
# ---------------------------------------------------------------------------

class ODesignTrunk:
    """ODesign full trunk on Tenstorrent (pass 7): InputFeatureEmbedder -> trunk
    cycle (48-block Pairformer + CTE 2-block pair-only Pairformer + 4-block MSA)
    -> (s_trunk, z_trunk). Reuses the pass-6 `ODesignTrunkInput` for the front-end
    (RPE w/ cyclic offset, trunk-init linears, CTE front-end, InputFeatureEmbedder)
    and the protenix/tenstorrent Pairformer / PairformerLayer / MSA primitives
    with ODesign weights.

    Cycle (matching ODesign get_pairformer_output, N_cycle=10, inplace_safe):
      z = z_init + linear_no_bias_z_cycle(LN_z_cycle(z))
      z += CTE(input_data, z)            # distogram binning + 2 pair-only Pairformer blocks
      z  = MSA(input_data, z, s_inputs)  # single-sequence self-MSA (use_msa=false still
                                        #   produces a 1-row MSA via set_default_msa_features,
                                        #   so MSAModule.forward RUNS -- NOT a no-op; confirmed pass 7)
      s  = s_init + linear_no_bias_s(LN_s(s))
      s, z = PF(s, z)                   # 48-block with-single Pairformer

    No template (template_embedder.n_blocks=0 -> nt=0, skipped). MSA uses the
    inference-mode path (no msa_token_mask zeroing). Loaded from the full ODesign
    checkpoint state dict (module. prefix stripped)."""

    N_CYCLES = 10
    C_Z, C_S, C_S_INPUTS = 128, 384, 453
    TRI_HEAD_DIM = 32
    CTE_C, CTE_NO_BINS, CTE_MIN_BIN, CTE_MAX_BIN = 64, 39, 3.25, 50.75

    def __init__(self, model_state_dict, compute_kernel_config, device=None):
        import re
        from .tenstorrent import (get_device, Pairformer, PairformerLayer,
                                   OuterProductMean, PairWeightedAveraging, Transition)
        self._w = model_state_dict
        self.compute_kernel_config = compute_kernel_config
        self.dev = device or get_device()
        self._wc = {}  # device weight cache (uploaded once, reused across cycles)
        # Pass-8 precision levers (release-gated; env-gated for A/B measurement).
        #   ODESIGN_FP32_TRANSITION=1 -> fp32 pair/single Transition MLP in the 48-block
        #   PF + CTE pairformer + MSA pair_stack (the 4x inner expansion, 512-wide bf16
        #   matmul, is the most likely 10-cycle z-precision bottleneck per pass-7).
        self.fp32_transition = os.environ.get("ODESIGN_FP32_TRANSITION", "1") == "1"
        # Lever 3 (extended): keep z in fp32 across the 48-block PF residual adds.
        # DEFAULT OFF: on the small Wormhole grid the per-op bf16 typecast of the
        # fp32 accumulator lands in a memory config that sends the triangle-attention
        # SDPA into a pathological multi-minute recompile/compute per block (pass-8
        # measured: 25+ min, no completion). Revisit in pass 9 with a layout-preserving
        # typecast or by making the triangle ops natively fp32.
        self.fp32_residual = os.environ.get("ODESIGN_FP32_RESIDUAL", "0") == "1"
        # front-end (pass-6, parity-verified): RPE w/ cyclic offset, trunk-init linears,
        # CTE front-end (v_ij), InputFeatureEmbedder (atom encoder -> s_inputs).
        self.ti = ODesignTrunkInput(model_state_dict, compute_kernel_config, self.dev)

        n_tri_heads = self.C_Z // self.TRI_HEAD_DIM  # 128 // 32 = 4
        # 48-block with-single Pairformer (c_z=128, c_s=384, att 16 heads x head_dim=24)
        nb_pf = 1 + max(int(re.search(r"pairformer_stack\.blocks\.(\d+)\.", k).group(1))
                        for k in self._w if "pairformer_stack.blocks." in k)
        comb = {}
        for i in range(nb_pf):
            blk = {k[len(f"pairformer_stack.blocks.{i}."):]: v for k, v in self._w.items()
                   if k.startswith(f"pairformer_stack.blocks.{i}.")}
            for k, v in PW.remap_pairformer_block(blk).items():
                comb[f"layers.{i}.{k}"] = v
        self.PF = Pairformer(nb_pf, self.TRI_HEAD_DIM, n_tri_heads, self.C_S // 16, 16,
                             True, comb, compute_kernel_config,
                             fp32_transition=self.fp32_transition,
                             fp32_residual=self.fp32_residual)

        # ConstraintTemplateEmbedder: 2 pair-only PairformerLayers (c=64, 4 heads x head_dim=32)
        self.CTE_PF = []
        for b in range(2):
            p = f"constraint_distogram_embedder.pairformer_stack.blocks.{b}."
            sub = {k[len(p):]: v for k, v in self._w.items() if k.startswith(p)}
            self.CTE_PF.append(PairformerLayer(self.TRI_HEAD_DIM, n_tri_heads, None, None,
                                False, PW.remap_msa_pair_stack(sub), compute_kernel_config,
                                fp32_transition=self.fp32_transition))

        # 4-block MSA module (reuse protenix.Trunk's MSA wiring verbatim)
        self.MSA = []
        for i in range(4):
            P = f"msa_module.blocks.{i}."
            sub = lambda pp: {k[len(pp):]: v for k, v in self._w.items() if k.startswith(pp)}
            opm = OuterProductMean(PW.remap_outer_product_mean(sub(P + "outer_product_mean_msa.")),
                                   compute_kernel_config)
            pl = PairformerLayer(self.TRI_HEAD_DIM, n_tri_heads, None, None, False,
                                  PW.remap_msa_pair_stack(sub(P + "pair_stack.")),
                                  compute_kernel_config,
                                  fp32_transition=self.fp32_transition)
            has = any(k.startswith(P + "msa_stack.") for k in self._w)
            pwa = tm = None
            if has:
                pwa = PairWeightedAveraging(8, 8,
                        PW.remap_pair_weighted_averaging(sub(P + "msa_stack.msa_pair_weighted_averaging.")),
                        compute_kernel_config)
                tm = Transition(PW.remap_transition(sub(P + "msa_stack.transition_m.")),
                                compute_kernel_config, fp32=self.fp32_transition)
            self.MSA.append((opm, pwa, tm, pl))
        self._msa_update_first = False        # ODesign/protenix-v2 order: OPM before MSA update

    # --- _KeyedWeights-style helpers (this class owns its own weight cache so it
    #     can stay self-contained without inheriting the mixin) ---
    def _w_tt(self, key, transpose=True):
        v = self._wc.get((key, transpose))
        if v is None:
            w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)
            self._wc[(key, transpose)] = v
        return v

    def _up(self, t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)

    def _lin(self, x, wkey, bkey=None):
        return ttnn.linear(x, self._w_tt(wkey), bias=(self._w_tt(bkey, False) if bkey else None),
                            compute_kernel_config=self.compute_kernel_config,
                            core_grid=CORE_GRID_MAIN)

    def _ln(self, x, wkey, bkey):
        return ttnn.layer_norm(x, weight=self._w_tt(wkey, False), bias=self._w_tt(bkey, False),
                               epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)

    @staticmethod
    def _to_host(t, shape=None):
        h = torch.Tensor(ttnn.to_torch(t)).float()
        return h.reshape(shape) if shape is not None else h

    # --- ConstraintTemplateEmbedder (full: front-end + 2-block pairformer + LN_v + relu + linear_u) ---
    def _ctet_a_term(self, constraint_feature):
        """Precompute the distogram's linear_a term ONCE (constraint_feature is
        cycle-invariant): distogram = one_hot(sum(cf > boundaries, -1), 39);
        a_term = linear_no_bias_a(distogram) -> (N,N,64) device. Returned device
        tensor is added to linear_z(LN_z(z)) each cycle."""
        import torch.nn.functional as F
        boundaries = torch.linspace(self.CTE_MIN_BIN, self.CTE_MAX_BIN, self.CTE_NO_BINS - 1)
        cf = constraint_feature.float()
        true_bins = torch.sum(cf > boundaries, dim=-1)            # (N,N)
        distogram = F.one_hot(true_bins, self.CTE_NO_BINS).float()  # (N,N,39)
        return self._lin(self._up(distogram), "constraint_distogram_embedder.linear_no_bias_a.weight")

    def _cte(self, z_dev, a_term_dev):
        """One CTE cycle update on the device-resident z (N,N,128) -> u (N,N,128) added to z.
        v_ij = linear_no_bias_z(LN_z(z)) + a_term; v_ij = 2 pair-only PairformerLayers;
        u = linear_no_bias_u(relu(LN_v(v_ij)))."""
        C = "constraint_distogram_embedder."
        zc = self._ln(z_dev, C + "layernorm_z.weight", C + "layernorm_z.bias")
        v = self._lin(zc, C + "linear_no_bias_z.weight")          # (N,N,64)
        v = ttnn.add(v, a_term_dev)
        for pl in self.CTE_PF:
            v = pl(None, v)[1]
        v = self._ln(v, C + "layernorm_v.weight", C + "layernorm_v.bias")
        v = ttnn.relu(v)
        return self._lin(v, C + "linear_no_bias_u.weight")        # (N,N,128)

    # --- MSA module (reuse protenix.Trunk._msa logic; _msa_update_first=False matches ODesign).
    # ODesign's MSAModule.forward recomputes m_feat fresh each cycle from input_feature
    # (cycle-invariant) and discards the block-updated msa_sample -- only z is returned.
    # So _msa returns ONLY z3; the caller passes the same initial m_feat every cycle. ---
    def _msa(self, z3, m_feat):
        def update_msa(m, z, pwa, transition):
            if pwa is None:
                return m
            m = ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
            return ttnn.add(m, ttnn.reshape(transition(m), tuple(m.shape)))
        for (opm, pwa, tm, pl) in self.MSA:
            if self._msa_update_first:
                m_feat = update_msa(m_feat, z3, pwa, tm)
            z3 = ttnn.add(z3, opm(m_feat, None, None))
            if not self._msa_update_first:
                m_feat = update_msa(m_feat, z3, pwa, tm)
            z3 = pl(None, z3)[1]
        return z3

    def __call__(self, feat, n_cycles=None, progress_fn=None):
        """Run the full ODesign trunk. feat needs (host tensors): ref_pos, ref_space_uid,
        ref_element(129), ref_mask, ref_atom_name_chars, ref_charge, atom_to_token_idx,
        restype(35), profile(32), deletion_mean(1), is_hotspot_residue(1), token_bonds,
        msa(1,N), has_deletion(1,N), deletion_value(1,N), constraint_feature(N,N,1),
        and the token meta (asym_id, residue_index, entity_id, sym_id, token_index,
        is_cyclic_token). Returns (s_trunk (N,384), z_trunk (N,N,128)) host fp32."""
        import torch.nn.functional as F
        N = feat["residue_index"].shape[0]          # N_token (trunk operates at token level)
        # 1) s_inputs (InputFeatureEmbedder) + RPE + trunk-init linears (pass-6 front-end)
        s_inputs = self.ti.input_feature_embedder(feat)            # (N,453) host
        relp = self.ti.rpe(feat)                                  # (N,N,128) host
        token_bonds = feat["token_bonds"].float()
        s_init, z_init = self.ti.trunk_init(s_inputs, relp, token_bonds)  # host (N,384),(N,N,128)
        # 2) CTE cycle-invariant a_term (precompute once)
        a_term = self._ctet_a_term(feat["constraint_feature"])     # device (N,N,64)
        # 3) MSA m_feat (precompute once): linear_no_bias_m(ms) + linear_no_bias_s(s_inputs)
        msa = F.one_hot(feat["msa"].long(), 32).float()           # (1,N,32)
        ms = torch.cat([msa, feat["has_deletion"].unsqueeze(-1).float(),
                        feat["deletion_value"].unsqueeze(-1).float()], dim=-1).unsqueeze(0)  # (1,1,N,34)
        m_feat = ttnn.add(self._lin(self._up(ms), "msa_module.linear_no_bias_m.weight"),
                          self._lin(self._up(s_inputs), "msa_module.linear_no_bias_s.weight"))
        # 4) trunk cycle (N_cycle=10)
        z3 = ttnn.mul(self._up(z_init.float().reshape(1, N, N, self.C_Z)), 0.0)
        s = ttnn.mul(self._up(s_init.float().reshape(1, N, self.C_S)), 0.0)
        s_init_dev = self._up(s_init.float().reshape(1, N, self.C_S))
        z_init_dev = self._up(z_init.float().reshape(1, N, N, self.C_Z))
        n_cycles = self.N_CYCLES if n_cycles is None else n_cycles
        for cyc in range(n_cycles):
            if progress_fn:
                progress_fn("trunk", step=cyc, total=n_cycles)
            zc = self._lin(self._ln(z3, "layernorm_z_cycle.weight", "layernorm_z_cycle.bias"),
                            "linear_no_bias_z_cycle.weight")
            z3 = ttnn.add(z_init_dev, zc)
            z3 = ttnn.add(z3, self._cte(z3, a_term))              # CTE (2 pair-only blocks)
            z3 = self._msa(z3, m_feat)                          # 4-block MSA (m_feat cycle-invariant)
            sc = self._lin(self._ln(s, "layernorm_s.weight", "layernorm_s.bias"),
                            "linear_no_bias_s.weight")
            s = ttnn.add(s_init_dev, sc)
            s, z3 = self.PF(s, z3)
            s = ttnn.reshape(s, (N, self.C_S))
            if progress_fn:
                progress_fn("trunk_cycle_end", step=cyc, s=s, z=z3)
        return self._to_host(s, (N, self.C_S)), self._to_host(z3, (N, N, self.C_Z))


class ODesign:
    """ODesign denoiser-parity harness on Tenstorrent (pass-3 scope: diffusion leg only).

    Reuses tt_bio.protenix.DiffusionModule (loaded with ODesign weights) for the per-step
    denoise, and tt_bio.protenix.AtomFeaturization for the t-independent atom single (c_l)
    and atom-pair (p_lm) conditioning. The cond builder mirrors Protenix.fold's assembly but
    reads the trunk conditioning (s_trunk / s_inputs / z_trunk) from the captured golden
    intermediates instead of running the (unported) trunk, and uses ODesign's 129-dim
    ref_element in the atom featurization f_in.

    build_cond(pre) -> cond dict; denoise_step(x_noisy, t_hat, cond) -> denoised coords;
    replay_trajectory(pre, traj) -> per-step PCC list. See scripts/odesign_traj_replay.py.
    """

    C_S, C_Z, C_S_INPUTS = 384, 128, 453
    C_ATOM, C_ATOMPAIR, C_TOKEN = 128, 16, 768
    NQ, NK, PAD_LEFT = 32, 128, 48
    R_MAX, S_MAX = 32, 2

    def __init__(self, model_state_dict, compute_kernel_config, device=None):
        self._w = model_state_dict
        self.compute_kernel_config = compute_kernel_config
        self.dev = device or get_device()

        def under(pfx):
            return {k[len(pfx):]: v for k, v in self._w.items() if k.startswith(pfx)}
        self.diffusion = DiffusionModule(under("diffusion_module."), self.dev, compute_kernel_config)
        self.diff_feat = AtomFeaturization(under("diffusion_module.atom_attention_encoder."),
                                          compute_kernel_config)

    @classmethod
    def load_from_checkpoint(cls, path, compute_kernel_config=None, device=None):
        """Load an ODesign .pt checkpoint and build the denoiser. weights_only=True."""
        dev = device or get_device()
        ckc = compute_kernel_config or ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        ck = ck.get("model", ck)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
        return cls(sd, ckc, dev)

    # --- upload helpers (mirror protenix._KeyedWeights; the diffusion module owns its own
    #     weight cache, these are for the cond-building linears on this object) ---
    def _tt(self, x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)

    @staticmethod
    def _to_host(t, shape=None):
        h = torch.Tensor(ttnn.to_torch(t)).float()
        return h.reshape(shape) if shape is not None else h

    def _lin(self, x, wkey):
        w = self._w[wkey]
        wtt = ttnn.from_torch(w.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                              device=self.dev, dtype=ttnn.bfloat16)
        return ttnn.linear(x, wtt, compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)

    def _ln(self, x, wkey):
        w = ttnn.from_torch(self._w[wkey], layout=ttnn.TILE_LAYOUT,
                            device=self.dev, dtype=ttnn.bfloat16)
        return ttnn.layer_norm(x, weight=w, epsilon=1e-5,
                              compute_kernel_config=self.compute_kernel_config)

    @staticmethod
    def _generate_relp(feats, r_max=32, s_max=2):
        """RelativePositionEncoder feature (reference embedders.generate_relp): one-hot of
        clipped residue/token/chain offsets + same-entity. dims 2(r_max+1)+2(r_max+1)+1+
        2(s_max+1) = 139. Identical to Protenix-v2 (same r_max/s_max)."""
        import torch.nn.functional as F
        asym = feats["asym_id"].long(); res = feats["residue_index"].long()
        ent = feats["entity_id"].long(); tok = feats["token_index"].long(); sym = feats["sym_id"].long()
        sc = (asym[:, None] == asym[None, :]).long()
        sr = (res[:, None] == res[None, :]).long()
        se = (ent[:, None] == ent[None, :]).long()
        d_res = torch.clip(res[:, None] - res[None, :] + r_max, 0, 2 * r_max) * sc + (1 - sc) * (2 * r_max + 1)
        d_tok = torch.clip(tok[:, None] - tok[None, :] + r_max, 0, 2 * r_max) * sc * sr + (1 - sc * sr) * (2 * r_max + 1)
        d_ch = torch.clip(sym[:, None] - sym[None, :] + s_max, 0, 2 * s_max) * se + (1 - se) * (2 * s_max + 1)
        return torch.cat([F.one_hot(d_res, 2 * (r_max + 1)), F.one_hot(d_tok, 2 * (r_max + 1)),
                          se[..., None], F.one_hot(d_ch, 2 * (s_max + 1))], dim=-1).float()

    @staticmethod
    def _atom_pair_feats(ref_pos, ref_space_uid):
        """Algorithm 5 lines 1-3 (reference update_input_feature_dict): windowed atom-pair
        feats from ref_pos + ref_space_uid (NQ=32, NK=128, pad_left=48). Identical to
        Protenix-v2. Returns d_lm (nb,NQ,NK,3), v_lm (nb,NQ,NK,1), mask_trunked (nb,NQ,NK)."""
        import torch.nn.functional as F
        N = ref_pos.shape[0]; NQ, NK, PADL = 32, 128, 48
        nb = (N + NQ - 1) // NQ; NP = nb * NQ; qpad = NP - N
        ruid = ref_space_uid.long()
        qpos = F.pad(ref_pos.float(), (0, 0, 0, qpad)).reshape(nb, NQ, 3)
        quid = F.pad(ruid, (0, qpad), value=0).reshape(nb, NQ)
        pad_right = int((nb - 0.5) * NQ + NK / 2 - N + 0.5)
        kpos_p = F.pad(ref_pos.float(), (0, 0, PADL, pad_right))
        kuid_p = F.pad(ruid, (PADL, pad_right), value=0)
        kpos = torch.stack([kpos_p[b * NQ:b * NQ + NK] for b in range(nb)], 0)
        kuid = torch.stack([kuid_p[b * NQ:b * NQ + NK] for b in range(nb)], 0)
        d_lm = qpos[:, :, None, :] - kpos[:, None, :, :]
        v_lm = (quid[:, :, None] == kuid[:, None, :]).float().unsqueeze(-1)
        qidx = torch.arange(NP).reshape(nb, NQ); qval = (qidx < N).float()
        kglob = torch.stack([torch.arange(b * NQ - PADL, b * NQ - PADL + NK) for b in range(nb)], 0)
        kval = ((kglob >= 0) & (kglob < N)).float()
        mask_trunked = qval[:, :, None] * kval[:, None, :]
        return d_lm, v_lm, mask_trunked

    def _diffusion_pair_cond(self, z_trunk_tt, relp):
        """DiffusionConditioning pair branch (reference diffusion_module.diffusion_conditioning):
        zc = LN(concat[z_trunk, relpe(relp)]); pz = linear_z(zc); pz += transition_z1 +
        transition_z2. Identical to Protenix-v2 (no z_trunk compression -- ODesign keeps
        c_z_pair_diffusion == c_z == 128). Returns conditioned pair_z host (NT,NT,c_z)."""
        from .tenstorrent import Transition
        C = "diffusion_module.diffusion_conditioning."
        relpe = ttnn.linear(self._tt(relp), self._tt(self._w[C + "relpe.linear_no_bias.weight"].t().contiguous()),
                            compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        z_trunk_tt = ttnn.reshape(z_trunk_tt, (relpe.shape[0], relpe.shape[1], -1))
        zc = ttnn.concat([z_trunk_tt, relpe], dim=-1)
        zc = ttnn.layer_norm(zc, weight=self._tt(self._w[C + "layernorm_z.weight"]), epsilon=1e-5,
                             compute_kernel_config=self.compute_kernel_config)
        pz = ttnn.linear(zc, self._tt(self._w[C + "linear_no_bias_z.weight"].t().contiguous()),
                         compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        N = relpe.shape[0]
        pz = ttnn.reshape(pz, (1, N, N, pz.shape[-1]))
        for nm in ("transition_z1", "transition_z2"):
            sub = {k[len(C + nm + "."):]: v for k, v in self._w.items() if k.startswith(C + nm + ".")}
            pz = ttnn.add(pz, Transition(PW.remap_transition(sub), self.compute_kernel_config)(pz))
        return self._to_host(pz)

    def _plm_z_term(self, pair_z, a2t, nb, nq, nk):
        """broadcast_token_to_local_atom_pair: W_z(LN_z(z_trunk)) gathered into windowed
        atom-pair blocks (nb,nq,nk,16). The diffusion atom-encoder's p_lm cache adds this
        trunk-pair-z term (reference transformer.py prepare_cache, r_l path). Identical to
        Protenix-v2."""
        import torch.nn.functional as F
        E = "diffusion_module.atom_attention_encoder."
        lnz = F.layer_norm(pair_z, (pair_z.shape[-1],)) * self._w[E + "layernorm_z.weight"]
        ztok = F.linear(lnz, self._w[E + "linear_no_bias_z.weight"])
        N = a2t.shape[0]; NQ, NK, PADL = 32, 128, 48; NP = nb * NQ
        aq = torch.cat([a2t, torch.zeros(NP - N, dtype=torch.long)]).reshape(nb, NQ)
        ak_src = torch.cat([torch.zeros(PADL, dtype=torch.long), a2t,
                            torch.zeros(PADL + NP + NK, dtype=torch.long)])
        ak = torch.stack([ak_src[b * NQ:b * NQ + NK] for b in range(nb)], 0)
        return torch.stack([ztok[aq[b][:, None].expand(NQ, NK), ak[b][None, :].expand(NQ, NK)]
                            for b in range(nb)], 0)

    def build_cond(self, pre):
        """Assemble the denoiser cond dict from the captured golden trunk conditioning.

        pre: dict with s_trunk (NT,c_s), s_inputs (NT,c_s_inputs), z_trunk (NT,NT,c_z),
        input_data (atom feats: ref_pos, ref_space_uid, ref_element, ref_mask,
        ref_atom_name_chars, ref_charge, atom_to_token_idx, ...).
        Returns cond = {s_trunk, s_inputs, pair_z, c_l, p_lm, S, mask_trunked} with host
        tensors, matching the contract of protenix.DiffusionModule.denoise. No dit_z is
        set, so denoise uses the host fp32 DiT path (strict per-step parity, matching
        scripts/protenix_traj_replay.py)."""
        import torch
        s_inputs = pre["s_inputs"].float(); s_trunk = pre["s_trunk"].float()
        z_trunk = pre["z_trunk"].float()
        feat = pre["input_data"]
        N = feat["ref_pos"].shape[0]; NT = s_inputs.shape[0]
        a2t = feat["atom_to_token_idx"].long()
        # atom single feat f_in: cat([ref_mask(1), ref_element(129), ref_atom_name_chars(256)]) = 386
        # (ODesign ref_element is 129-dim vs Protenix-v2's 128 -- the only atom-feat diff.)
        f_in = torch.cat([feat["ref_mask"].reshape(N, 1),
                         feat["ref_element"].reshape(N, 129),
                         feat["ref_atom_name_chars"].reshape(N, 256)], dim=-1).float()
        d_lm, v_lm, mt = self._atom_pair_feats(feat["ref_pos"], feat["ref_space_uid"])
        nb, nq, nk, _ = d_lm.shape
        M = nb * nq * nk
        d = d_lm.reshape(M, 3); v = v_lm.reshape(M, 1)
        invd = (1.0 / (1.0 + (d_lm ** 2).sum(-1, keepdim=True))).reshape(M, 1)
        S = torch.zeros(N, NT); S[torch.arange(N), a2t] = 1.0
        ref_charge_asinh = torch.arcsinh(feat["ref_charge"]).reshape(N, 1).float()
        tt = self._tt
        # c_l (atom single conditioning) and p_lm (windowed atom-pair) via the diffusion
        # atom encoder's AtomFeaturization, then augment with the c_l_q/c_l_k + small_mlp
        # and the trunk-pair-z broadcast (mirrors protenix.fold's p_lm assembly).
        c_l = self._to_host(self.diff_feat.c_l(
            tt(feat["ref_pos"].float()), tt(ref_charge_asinh),
            tt(feat["ref_mask"].reshape(N, 1).float()), tt(f_in)), (N, self.C_ATOM))
        p_lm = self._to_host(self.diff_feat.p_lm(tt(d), tt(v), tt(invd), tt(mt.reshape(-1, 1).float())),
                             (nb, nq, nk, self.C_ATOMPAIR))
        pair_z = self._diffusion_pair_cond(tt(z_trunk), self._generate_relp(feat)).reshape(NT, NT, self.C_Z)
        p_lm = p_lm + self._plm_z_term(pair_z, a2t, nb, nq, nk)
        return {"s_trunk": s_trunk, "s_inputs": s_inputs, "pair_z": pair_z, "c_l": c_l,
                "p_lm": p_lm, "S": S, "mask_trunked": mt.float()}

    def denoise_step(self, x_noisy, t_hat, cond):
        """One denoise network step, matching ODesign's DiffusionModule.forward. x_noisy
        (1,N,3) is the ODesign checkpoint's ALREADY-c_in-scaled noisy coords
        (c_in = 1/sqrt(sd^2+t^2); see schedulers.add_noise_with_condition -- magnitude ~1,
        NOT raw coords). t_hat (1,), cond from build_cond. Returns the raw network output
        x_update (1,N,3) host tensor -- NOT EDM-preconditioned.

        Two conventions differ from protenix.DiffusionModule.denoise, both absorbed here
        by calling _denoise_net directly (no c_in re-scaling, no EDM preconditioning):
          (1) c_in: ODesign stores x_noisy already c_in-scaled and feeds it straight to the
              atom encoder (r_l = x_noisy); protenix.denoise expects RAW x_noisy and re-applies
              c_in internally. Feeding ODesign's scaled x_noisy to protenix.denoise would
              double-scale the encoder's coordinate input (~sigma/sigma_data too small).
          (2) EDM: ODesign's DiffusionModule.forward returns the raw network output x_update
              (the sampler's update_with_condition applies EDM later); protenix.denoise applies
              EDM itself and returns c_skip*x + c_out*x_update. The golden trajectory's
              `denoised` field is the hooked diffusion_module.forward return = x_update, so we
              compare x_update-to-x_update (no EDM)."""
        self.diffusion._atom_cond(cond)
        return self.diffusion._denoise_net(x_noisy[0].float(), t_hat, cond)

    def replay_trajectory(self, pre, traj, n_steps=None, verbose=True):
        """Replay the golden denoiser trajectory step-by-step and report REAL per-step PCC
        vs the reference `denoised`. Mirrors scripts/protenix_traj_replay.py. n_steps=None
        replays all 200; an int caps it (e.g. 1 for a single-step smoke). Returns
        (pccs, maxerrs) lists."""
        cond = self.build_cond(pre)
        N = pre["input_data"]["ref_pos"].shape[0]
        steps = traj["steps"][:n_steps] if n_steps is not None else traj["steps"]

        def pcc(u, v):
            u = u.flatten().double(); v = v.flatten().double()
            return float(((u - u.mean()) * (v - v.mean())).sum()
                         / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))
        pccs, maxerrs = [], []
        for i, st in enumerate(steps):
            xn = st["x_noisy"].float(); th = st["t_hat"].float(); ref = st["denoised"].float()
            out = self.denoise_step(xn, th, cond)
            p = pcc(out, ref[:, :N]); m = float((out - ref[:, :N]).abs().max())
            pccs.append(p); maxerrs.append(m)
            if verbose:
                print("step %3d  t_hat=%9.4g  denoised PCC %.5f  maxerr %.3e"
                      % (i, float(th.max()), p, m), flush=True)
        if verbose and pccs:
            print("\nALL-STEP denoiser PCC: min %.5f  mean %.5f  (across t_hat %.3g..%.3g)"
                  % (min(pccs), sum(pccs) / len(pccs),
                     float(steps[-1]["t_hat"].max()), float(steps[0]["t_hat"].max())), flush=True)
        return pccs, maxerrs

    def closed_loop_sample(self, pre, draws, cond=None, verbose=True):
        """Run the FULL ODesign sampler loop (Algorithm 18) around the on-device
        denoise_step, consuming the precomputed `draws` (init noise + per-step
        rotation/translation/corrector-noise) so the result is directly comparable
        to a CPU-fp32 reference sampler fed the SAME draws. This is the closed-loop
        parity test the pass-4 per-step replay did not exercise: augmentation,
        condition enforcement, and the EDM Euler update compound the per-step
        network error over all 200 steps.

        pre: golden trunk conditioning (s_inputs/s_trunk/z_trunk/input_data/...).
        draws: dict from generate_sampler_draws (identical for device + CPU-ref).
        cond: prebuilt cond dict (build_cond); built from pre if None.
        Returns final coords (1,N,3) host fp32 tensor.

        The loop mirrors ODesign src/model/modules/generator.sample_diffusion exactly:
          x_l = init_noise (free atoms) / x_gt (cond atoms)
          for step: x_l,trans,rot,center = centre_random_augmentation(x_l)
                    t_hat,c_tau = get_noise_level(step)
                    x_noisy = add_noise_with_condition(x_l, cond_mask, noise=noises[step])
                    x_update = denoise_step(x_noisy, t_hat, cond)   # on-device network
                    x_l = update_with_condition(x_noisy, x_update, x_gt=x_l_augment, cond_mask)
                    x_l = reverse_centre_random_augmentation(x_l, trans, rot, center)
        No condition atoms in this golden (is_condition_atom sum 0) -> unconditional
        generation; condition enforcement is a no-op here but implemented for fidelity.
        """
        if cond is None:
            cond = self.build_cond(pre)
        feat = pre["input_data"]
        N = feat["ref_pos"].shape[0]
        schedule = draws["schedule"]
        cond_mask = draws["condition_mask"]
        x_gt = pre["gt_coordinate"].float()                      # (N,3); unused when no cond atoms
        x_l = draws["init_noise"].clone()                        # (1,N,3) s_max*randn
        n_step = len(draws["rots"])
        for i in range(n_step):
            x_l_aug, trans, rot, x_center = centre_random_augmentation(
                x_l.squeeze(0), n_sample=1, rot=draws["rots"][i], trans=draws["trans"][i])
            # x_l_aug is (1,N,3); keep the N_sample dim -- the reference DiffusionModule
            # and our denoise_step both expect (N_sample,N_atom,3).
            x_l_augment = x_l_aug.clone()
            t_hat, c_tau, c_tau_last = edm_step_params(schedule, i, n_sample=1)
            x_noisy = add_noise_with_condition(
                x_l_aug, cond_mask, t_hat, c_tau_last=c_tau_last, noise=draws["noises"][i])
            x_update = self.denoise_step(x_noisy, t_hat, cond)     # on-device (1,N,3)
            x_l = update_with_condition(x_noisy, x_update, x_gt=x_l_augment,
                                        condition_mask=cond_mask, t_hat=t_hat, c_tau=c_tau)
            x_l = reverse_centre_random_augmentation(x_l, trans, rot, x_center)
            if verbose and (i % 20 == 0 or i == n_step - 1):
                print("  closed-loop step %3d  t_hat=%9.4g  |x_l|=%.4f"
                      % (i, float(t_hat.max()), float(x_l.norm() / (x_l.numel() ** 0.5))), flush=True)
        return x_l
