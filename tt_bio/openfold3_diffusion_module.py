"""OpenFold3 ``DiffusionModule`` device port (P8 assembly + P9 merge gate).

Contains two device modules:

  * ``OF3NoisyPositionEmbedder`` -- the entry leg of the ``DiffusionModule`` atom
    encoder (AF3 Algorithm 5, lines 8-12), PCC-gated on device in P8 tick 16.
  * ``OF3DiffusionModule`` -- the full post-conditioning ``DiffusionModule`` forward
    (AF3 Algorithm 20): conditioning -> atom encoder -> DiT -> atom decoder -> EDM
    output scaling -> ``xl_out``, PCC-gated against the full-module golden.

``OF3NoisyPositionEmbedder`` fuses the trunk single/pair representations into the
reference-conformer atom conditioning and seeds the atom single rep ``ql`` from the
noisy coordinates. Reference topology
(``openfold3.core.model.layers.sequence_local_atom_attention.NoisyPositionEmbedder``):

    cl = cl0 + broadcast_to_atoms(linear_s(LN_s(si_trunk)))      # single broadcast
    plm = plm0 + to_blocks(linear_z(LN_z(zij_trunk)))            # pair broadcast (blocked)
    ql = cl + linear_r(rl_noisy)                                 # noisy-coordinate projection

``cl0``/``plm0`` are the ``RefAtomFeatureEmbedder`` outputs (gated separately); ``zij``
is the *conditioned* pair (diffusion-conditioning output). All five linears are
bias-free; both LNs are weight-only (``create_offset=False``, eps=1e-5).

The two broadcasts are mask-derived gathers (token -> atom, token-pair -> atom-block)
precomputed on host in the golden and replayed on device via ``ttnn.embedding`` -- the
same isolation discipline as the P7 atom-transformer key gather:
  * single: ``atom_to_token_index`` [NP] gathers ``linear_s(LN_s(si_trunk))`` [N_tok, c]
    to [NP, c] (padded atoms -> 0, zeroed by ``atom_mask_col``);
  * pair: ``zij_flat_idx`` [nb*nq*nk] gathers ``linear_z(LN_z(zij))`` [N_tok*N_tok, c]
    (flattened with stride = N_tok_pad) to [nb, nq, nk, c], masked by ``zij_mask``
    (``(1-invalid_key) * atom_pair_mask``).

Padded atom positions (n_atom -> NP = nb*N_QUERY) are zeroed via ``atom_mask_col`` so
the additive single broadcast does not leak into real atoms; the pair broadcast is
zeroed at padded query positions by ``zij_mask``.
"""
import math

import torch
import ttnn

from .tenstorrent import Module, CORE_GRID_MAIN
from .openfold3_atom_transformer import OF3AtomTransformer
from .openfold3_diffusion_transformer import OF3DiffusionTransformer
from .openfold3_diffusion_decoder import OF3AtomAttentionDecoder


def _sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


class OF3NoisyPositionEmbedder(Module):
    """Device port of OF3 ``NoisyPositionEmbedder`` (Algorithm 5 L8-12).

    Inputs (device bf16):
        cl0:  [1, NP, 128]         RefAtomFeatureEmbedder single out (atom-padded, 0 at pad).
        plm0: [1, nb, 32, 128, 16] RefAtomFeatureEmbedder pair out (blocked).
        si_trunk: [1, N_tok_pad, 384]  raw trunk single (seq-padded to a tile).
        zij:  [1, N_tok_pad, N_tok_pad, 128]  conditioned pair (seq-padded).
        rl:   [1, NP, 3]           noisy atom positions (atom-padded, 0 at pad).
        atom_mask_col:       [1, NP, 1] device.
        atom_to_token_idx:   [1, NP] uint32 device (real atoms -> token idx; pad -> 0).
        zij_flat_idx:        [1, nb*32*128] uint32 device
                                 (= q_token*N_tok_pad + k_token per (b,q,k)).
        zij_mask:            [1, nb, 32, 128, 1] device.
        n_atom, NP: ints.

    Outputs (device bf16):
        cl:  [1, n_atom, 128] ; plm: [1, nb, 32, 128, 16] ; ql: [1, n_atom, 128]
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w_ls = self.torch_to_tt("linear_s.weight")         # (384, 128) tiled
        self.w_lz = self.torch_to_tt("linear_z.weight")         # (128, 16) tiled
        self.w_lr = self.torch_to_tt("linear_r.weight")         # (3, 128) tiled
        self.ln_s_w = self.torch_to_tt("layer_norm_s.weight", transform=lambda x: x)
        self.ln_z_w = self.torch_to_tt("layer_norm_z.weight", transform=lambda x: x)

    def __call__(self, cl0, plm0, si_trunk, zij, rl, atom_mask_col,
                 atom_to_token_idx_tt, zij_flat_idx_tt, zij_mask, n_atom, NP):
        ckc = self.compute_kernel_config

        # --- single broadcast: cl = cl0 + broadcast_to_atoms(linear_s(LN_s(si_trunk))) ---
        si_ln = ttnn.layer_norm(si_trunk, weight=self.ln_s_w, epsilon=1e-5,
                                compute_kernel_config=ckc)                      # [1, Ntk, 384]
        si_tok = ttnn.linear(si_ln, self.w_ls, compute_kernel_config=ckc,
                             core_grid=CORE_GRID_MAIN)                          # [1, Ntk, 128]
        si_tok_2d = ttnn.reshape(ttnn.to_layout(si_tok, ttnn.ROW_MAJOR_LAYOUT),
                                 (si_tok.shape[1], 128))
        si_atoms = ttnn.embedding(atom_to_token_idx_tt, si_tok_2d,
                                  layout=ttnn.ROW_MAJOR_LAYOUT,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)        # [NP, 128]
        si_atoms = ttnn.to_layout(ttnn.reshape(si_atoms, (1, NP, 128)), ttnn.TILE_LAYOUT)
        si_atoms = ttnn.multiply(si_atoms, atom_mask_col)                       # zero padded
        ttnn.deallocate(si_tok)
        cl = ttnn.add(cl0, si_atoms)
        ttnn.deallocate(si_atoms)

        # --- pair broadcast: plm = plm0 + to_blocks(linear_z(LN_z(zij))) ---
        zij_ln = ttnn.layer_norm(zij, weight=self.ln_z_w, epsilon=1e-5,
                                 compute_kernel_config=ckc)                     # [1, Ntk, Ntk, 128]
        zij_tok = ttnn.linear(zij_ln, self.w_lz, compute_kernel_config=ckc,
                              core_grid=CORE_GRID_MAIN)                         # [1, Ntk, Ntk, 16]
        ntk = zij_tok.shape[1]
        zij_tok_2d = ttnn.reshape(ttnn.to_layout(zij_tok, ttnn.ROW_MAJOR_LAYOUT),
                                  (ntk * ntk, 16))
        zij_blocks = ttnn.embedding(zij_flat_idx_tt, zij_tok_2d,
                                    layout=ttnn.ROW_MAJOR_LAYOUT,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)      # [1, nb*nq*nk, 16]
        nb = zij_mask.shape[1]
        zij_blocks = ttnn.to_layout(
            ttnn.reshape(zij_blocks, (1, nb, 32, 128, 16)), ttnn.TILE_LAYOUT)
        zij_blocks = ttnn.multiply(zij_blocks, zij_mask)
        ttnn.deallocate(zij_tok)
        plm = ttnn.add(plm0, zij_blocks)
        ttnn.deallocate(zij_blocks)

        # --- noisy-coordinate projection: ql = cl + linear_r(rl) ---
        rl_proj = ttnn.linear(rl, self.w_lr, compute_kernel_config=ckc,
                              core_grid=CORE_GRID_MAIN)                         # [1, NP, 128]
        ql = ttnn.add(cl, rl_proj)
        ttnn.deallocate(rl_proj)

        # slice atom-padded outputs back to n_atom.
        cl = ttnn.to_layout(cl, ttnn.ROW_MAJOR_LAYOUT)
        cl = ttnn.slice(cl, [0, 0, 0], [1, n_atom, 128])
        cl = ttnn.to_layout(cl, ttnn.TILE_LAYOUT)
        ql = ttnn.to_layout(ql, ttnn.ROW_MAJOR_LAYOUT)
        ql = ttnn.slice(ql, [0, 0, 0], [1, n_atom, 128])
        ql = ttnn.to_layout(ql, ttnn.TILE_LAYOUT)
        return cl, plm, ql


class OF3DiffusionModule(Module):
    """Full OF3 ``DiffusionModule`` (AF3 Algorithm 20) device assembly -- the post-
    conditioning forward that turns the conditioned ``(si, zij)`` plus the
    reference-conformer atom conditioning ``(cl0, plm0)`` and a noisy sample
    ``rl_noisy`` into the denoised atom positions ``xl_out``.

    Topology (reference ``openfold3.core.model.structure.diffusion_module``):

        cl, plm, ql = NoisyPositionEmbedder(cl0, plm0, si_trunk, zij, rl_noisy)  # gated
        plm = AtomAttentionEncoder.pair_update(cl, plm)                          # fresh
        ql = AtomAttentionEncoder.atom_transformer(ql, cl, plm)                  # gated
        ai = aggregate_to_tokens(linear_q(ql))                                   # fresh
        ai = ai + linear_s(LN_s(si))                                             # fresh glue
        ai = DiffusionTransformer(ai, si, zij, token_mask)                       # gated
        ai = layer_norm_a(ai)                                                    # fresh glue
        rl_update = AtomAttentionDecoder(ai, ql, cl, plm)                        # gated
        xl_out = EDM_scaling(xl_noisy, rl_update, t, sigma_data)                 # fresh

    The conditioned ``(si, zij)`` come from ``OF3DiffusionConditioning`` (gated
    separately, PCC 1.00000/0.99999) and ``(cl0, plm0)`` from ``RefAtomFeatureEmbedder``
    (gated P7) -- both are fed from their goldens here, the same bisect discipline the
    NPE / decoder gates use, so this gate isolates the post-conditioning assembly
    (encoder pair update, linear_q aggregation, the two glue linears/LNs, EDM output
    scaling) plus the already-gated NPE / DiT / atom-transformer / decoder wiring.
    The encoder ``atom_transformer`` is the same ``OF3AtomTransformer`` topology as the
    decoder's (3-block windowed cross-attention), with the encoder's own weights.

    The fresh device work is small: ``linear_l``/``linear_m`` (128->16, ReLU-gated
    outer-sum into the blocked pair) + ``pair_mlp`` (3x 16->16 ReLU MLP) + ``linear_q``
    (128->768 + ReLU, atom->token mean aggregation via the host ``atom_to_token_mean``
    matrix) + the top-level ``linear_s``/``layer_norm_s`` glue + ``layer_norm_a`` +
    the EDM output scaling (two host scalars applied as multiplies). The mask-derived
    atom windowing and broadcasts are precomputed on host (golden) and replayed on
    device via ``ttnn.embedding`` -- identical isolation to the other OF3 legs.
    """
    N_QUERY = 32
    N_KEY = 128

    def __init__(self, state_dict, compute_kernel_config):
        # state_dict = _sub(sd, "diffusion_module")
        super().__init__({}, compute_kernel_config)
        self._w = dict(state_dict)
        self._wc = {}

        enc = _sub(self._w, "atom_attn_enc")
        self.npe = OF3NoisyPositionEmbedder(
            _sub(enc, "noisy_position_embedder"), compute_kernel_config)
        self.enc_at = OF3AtomTransformer(_sub(enc, "atom_transformer"),
                                         compute_kernel_config)
        self.dit = OF3DiffusionTransformer(_sub(self._w, "diffusion_transformer"),
                                           compute_kernel_config)
        self.dec = OF3AtomAttentionDecoder(_sub(self._w, "atom_attn_dec"),
                                           compute_kernel_config)

        # Fresh: encoder pair update (linear_l/linear_m + pair_mlp).
        self.w_ll = self._w_tt(enc["linear_l.weight"])           # (16, 128) tiled
        self.w_lm = self._w_tt(enc["linear_m.weight"])           # (16, 128) tiled
        self.w_pm1 = self._w_tt(enc["pair_mlp.1.weight"])        # (16, 16) tiled
        self.w_pm3 = self._w_tt(enc["pair_mlp.3.weight"])
        self.w_pm5 = self._w_tt(enc["pair_mlp.5.weight"])
        # Fresh: linear_q (128 -> 768) aggregation head.
        self.w_lq = self._w_tt(enc["linear_q.0.weight"])         # (768, 128) tiled
        # Fresh: top-level glue linear_s (384 -> 768) + weight-only layer_norm_s/a.
        self.w_ls = self._w_tt(self._w["linear_s.weight"])       # (768, 384) tiled
        self.ln_s_w = self._w_tt(self._w["layer_norm_s.weight"], False)  # (384,)
        self.ln_a_w = self._w_tt(self._w["layer_norm_a.weight"], False)  # (768,)

    def _w_tt(self, w, transpose=True):
        key = id(w)
        v = self._wc.get(key)
        if v is None:
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.device,
                                dtype=ttnn.bfloat16)
            self._wc[key] = v
        return v

    def _lin(self, x, w, activation=None):
        return ttnn.linear(x, w, activation=activation,
                           compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)

    @staticmethod
    def _pad_atoms(x, n_atom, NP):
        th = ttnn.to_torch(x).float()
        if NP > n_atom:
            th = torch.nn.functional.pad(th, (0, 0, 0, NP - n_atom))
        return ttnn.from_torch(th, layout=ttnn.TILE_LAYOUT,
                               device=x.device(), dtype=ttnn.bfloat16)

    @staticmethod
    def _pad_tokens(x, n_token, n_tok_pad):
        th = ttnn.to_torch(x).float()
        if n_tok_pad > n_token:
            th = torch.nn.functional.pad(th, (0, 0, 0, n_tok_pad - n_token))
        return ttnn.from_torch(th, layout=ttnn.TILE_LAYOUT,
                               device=x.device(), dtype=ttnn.bfloat16)

    def __call__(self, si_trunk, si, zij, cl0, plm0, rl_noisy, xl_noisy,
                 atom_mask_col, atom_mask_col_na, atom_to_token_idx_tt,
                 npe_flat_idx_tt, npe_zij_mask,
                 enc_key_block_idxs_tt, enc_valid_mask, enc_mask_bias, enc_pair_mask,
                 atom_to_token_mean_tt, token_mask_pad_tt, tok_mask_col_pad_tt,
                 n_atom, NP, nb, n_token, n_tok_pad, t, sigma_data,
                 _return_intermediates=False):
        """Run the post-conditioning ``DiffusionModule`` forward -> ``xl_out``.

        See the class docstring for the topology. Device bf16 inputs:
          si_trunk: [1, n_tok_pad, 384]  raw trunk single (NPE input).
          si, zij:            conditioned single/pair (zij tile-padded to n_tok_pad).
          cl0: [1, NP, 128]   RefAtom single (atom-padded, 0 at pad).
          plm0:[1, nb, 32, 128, 16]  RefAtom pair (blocked).
          rl_noisy: [1, NP, 3]  noisy atom positions (atom-padded, 0 at pad).
          xl_noisy: [1, n_atom, 3]  noisy sample (atom-masked) for EDM.
          atom_mask_col: [1, NP, 1]; atom_mask_col_na: [1, n_atom, 1].
          atom_to_token_idx_tt: [1, NP] uint32 (NPE single broadcast).
          npe_flat_idx_tt: [1, nb*32*128] uint32 (NPE pair broadcast).
          npe_zij_mask: [1, nb, 32, 128, 1].
          enc_key_block_idxs_tt: [1, nb*128] uint32 (encoder pair-update + atom-transformer).
          enc_valid_mask: [1, nb, 128, 1]; enc_mask_bias: [1, nb, 1, 32, 128].
          enc_pair_mask: [1, nb, 32, 128, 1] (multiplicative mask_trunked).
          atom_to_token_mean_tt: [1, n_token, n_atom] (mean aggregation matrix).
          token_mask_pad_tt: [1, n_tok_pad]; tok_mask_col_pad_tt: [1, n_tok_pad, 1].
        Returns [1, n_atom, 3] device bf16.
        """
        lin = self._lin

        # --- NoisyPositionEmbedder (gated) -> cl, plm, ql (sliced to n_atom). ---
        cl, plm, ql = self.npe(cl0, plm0, si_trunk, zij, rl_noisy, atom_mask_col,
                               atom_to_token_idx_tt, npe_flat_idx_tt, npe_zij_mask,
                               n_atom, NP)

        # --- Encoder pair update (fresh): cl_lm + pair_mlp, update plm. ---
        cl_pad = self._pad_atoms(cl, n_atom, NP)                # [1, NP, 128]
        ttnn.deallocate(cl)
        # cl_l = reshape to query blocks [1, nb, NQ, 128].
        cl_l = ttnn.to_layout(cl_pad, ttnn.ROW_MAJOR_LAYOUT)
        cl_l = ttnn.reshape(cl_l, (1, nb, self.N_QUERY, 128))
        cl_l = ttnn.to_layout(cl_l, ttnn.TILE_LAYOUT)
        # cl_m = gather cl by key_block_idxs -> [1, nb, NK, 128], invalid -> 0.
        cl2d = ttnn.reshape(ttnn.to_layout(cl_pad, ttnn.ROW_MAJOR_LAYOUT), (NP, 128))
        cl_m = ttnn.embedding(enc_key_block_idxs_tt, cl2d, layout=ttnn.ROW_MAJOR_LAYOUT,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cl_m = ttnn.reshape(cl_m, (1, nb, self.N_KEY, 128))
        cl_m = ttnn.to_layout(cl_m, ttnn.TILE_LAYOUT)
        cl_m = ttnn.multiply(cl_m, enc_valid_mask)              # [1, nb, NK, 128]
        # relu -> linear_l/m outer sum -> [1, nb, NQ, NK, 16], * pair_mask.
        # (reference: linear_l(relu(cl_l)) + linear_m(relu(cl_m)) -- relu is applied to
        #  the INPUT, before the linear, NOT as a post-activation.)
        ll = lin(ttnn.relu(cl_l), self.w_ll)                  # [1, nb, NQ, 16]
        ll = ttnn.unsqueeze(ll, -2)                            # [1, nb, NQ, 1, 16]
        lm = lin(ttnn.relu(cl_m), self.w_lm)                  # [1, nb, NK, 16]
        lm = ttnn.unsqueeze(lm, -3)                            # [1, nb, 1, NK, 16]
        ttnn.deallocate(cl_l); ttnn.deallocate(cl_m)
        cl_lm = ttnn.add(ll, lm)                               # [1, nb, NQ, NK, 16]
        ttnn.deallocate(ll); ttnn.deallocate(lm)
        cl_lm = ttnn.multiply(cl_lm, enc_pair_mask)            # * mask_trunked
        plm = ttnn.add(plm, cl_lm)
        ttnn.deallocate(cl_lm)
        # pair_mlp = Sequential(ReLU, Linear, ReLU, Linear, ReLU, Linear): relu BEFORE
        # each linear (no trailing relu). out = L5(relu(L3(relu(L1(relu(plm)))))).
        pm = lin(ttnn.relu(plm), self.w_pm1)
        pm = lin(ttnn.relu(pm), self.w_pm3)
        pm = lin(ttnn.relu(pm), self.w_pm5)
        plm = ttnn.add(plm, pm)
        ttnn.deallocate(pm)
        plm = ttnn.multiply(plm, enc_pair_mask)
        plm_postpu = ttnn.clone(plm) if _return_intermediates else None

        # --- Encoder atom_transformer (gated OF3AtomTransformer) -> ql [1, n_atom, 128]. ---
        ql_pad = self._pad_atoms(ql, n_atom, NP)               # [1, NP, 128]
        ttnn.deallocate(ql)
        ql_enc = self.enc_at(ql_pad, cl_pad, plm, atom_mask_col,
                             enc_key_block_idxs_tt, enc_valid_mask, enc_mask_bias,
                             n_atom, NP, nb)                  # [1, n_atom, 128]
        ttnn.deallocate(ql_pad)
        ql_enc_cp = ttnn.clone(ql_enc) if _return_intermediates else None

        # --- linear_q aggregation (fresh): ai = atom_to_token_mean @ relu(linear_q(ql)). ---
        qproj = lin(ql_enc, self.w_lq, activation="relu")      # [1, n_atom, 768]
        ai = ttnn.matmul(atom_to_token_mean_tt, qproj,
                         compute_kernel_config=self.compute_kernel_config)  # [1, n_token, 768]
        ttnn.deallocate(qproj)
        # Glue: ai += linear_s(LN_s(si)) (si tile-padded; slice the result to n_token).
        si_ln = ttnn.layer_norm(si, weight=self.ln_s_w, epsilon=1e-5,
                                compute_kernel_config=self.compute_kernel_config)
        si_proj = lin(si_ln, self.w_ls)                        # [1, n_tok_pad, 768]
        ttnn.deallocate(si_ln)
        si_proj = ttnn.to_layout(si_proj, ttnn.ROW_MAJOR_LAYOUT)
        si_proj = ttnn.slice(si_proj, [0, 0, 0], [1, n_token, 768])
        si_proj = ttnn.to_layout(si_proj, ttnn.TILE_LAYOUT)
        ai = ttnn.add(ai, si_proj)
        ttnn.deallocate(si_proj)
        ai_postglue = ttnn.clone(ai) if _return_intermediates else None

        # --- DiT (gated) -> ai. Feed padded (n_tok_pad) with the padded token mask. ---
        ai_pad = self._pad_tokens(ai, n_token, n_tok_pad)      # [1, n_tok_pad, 768]
        ttnn.deallocate(ai)
        ai_pad = self.dit(ai_pad, si, zij, token_mask_pad_tt, tok_mask_col_pad_tt)
        ai_postdit = ttnn.clone(ai_pad) if _return_intermediates else None

        # --- layer_norm_a (fresh glue), keep padded for the decoder. ---
        ai_ln = ttnn.layer_norm(ai_pad, weight=self.ln_a_w, epsilon=1e-5,
                                compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(ai_pad)

        # --- AtomAttentionDecoder (gated) -> rl_update [1, n_atom, 3]. ---
        # Decoder reuses the encoder's ql (atom_transformer output), cl, and the
        # post-pair-update plm -- the same (ql, cl, plm) the reference decoder consumes.
        ql_dec = self._pad_atoms(ql_enc, n_atom, NP)           # [1, NP, 128]
        ttnn.deallocate(ql_enc)
        rl_update = self.dec(ai_ln, ql_dec, cl_pad, plm, atom_mask_col,
                             atom_to_token_idx_tt, enc_key_block_idxs_tt,
                             enc_valid_mask, enc_mask_bias, n_atom, NP, nb)
        ttnn.deallocate(ai_ln); ttnn.deallocate(ql_dec)
        rl_update_cp = ttnn.clone(rl_update) if _return_intermediates else None

        # --- EDM output scaling (fresh) -> xl_out. ---
        sd2 = sigma_data * sigma_data
        t2 = t * t
        c_skip = sd2 / (sd2 + t2)
        c_out = sigma_data * t / math.sqrt(sd2 + t2)
        xl_out = ttnn.add(ttnn.multiply(xl_noisy, c_skip),
                          ttnn.multiply(rl_update, c_out))
        ttnn.deallocate(rl_update)
        xl_out = ttnn.multiply(xl_out, atom_mask_col_na)
        if _return_intermediates:
            return xl_out, ai_postglue, ai_postdit, rl_update_cp, plm_postpu, ql_enc_cp
        return xl_out
