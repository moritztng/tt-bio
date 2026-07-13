"""OpenFold3 ``NoisyPositionEmbedder`` (AF3 Algorithm 5, lines 8-12) on device.

The entry leg of the ``DiffusionModule`` atom encoder: fuses the trunk single/pair
representations into the reference-conformer atom conditioning and seeds the atom
single rep ``ql`` from the noisy coordinates. Reference topology
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
import torch
import ttnn

from .tenstorrent import Module, CORE_GRID_MAIN


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
        self.ln_s_w = self.torch_to_tt("layer_norm_s.weight", transform=lambda x: x)  # (384,)
        self.ln_z_w = self.torch_to_tt("layer_norm_z.weight", transform=lambda x: x)  # (128,)

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
