"""OpenFold3 ``AtomAttentionDecoder`` (AF3 Algorithm 6) on device.

The exit leg of ``DiffusionModule.forward``: maps the token-level DiT output ``ai`` back
to a per-atom coordinate update ``rl_update`` [N_atom, 3]. Topology (reference
``openfold3.core.model.layers.sequence_local_atom_attention.AtomAttentionDecoder``):

    ql'  = ql + broadcast_to_atoms(linear_q_in(ai))   # token -> atom broadcast
    ql'' = AtomTransformer(ql', cl, plm, atom_mask)    # 3-block cross-attention (Alg 5 L15)
    rl_update = linear_q_out(layer_norm(ql''))         # weight-only LN + c_atom->3

The 3-block cross-attention is the *same* ``DiffusionTransformer`` (non-cross, windowed)
topology as the encoder's and is reused verbatim via ``OF3AtomTransformer`` (separate
decoder weights ``diffusion_module.atom_attn_dec.atom_transformer.*``). The fresh device
work here is therefore just:

  * ``linear_q_in`` (c_token=768 -> c_atom=128, bias-free) + the token->atom broadcast
    (host gather index ``atom_to_token_index`` replayed on device via ``ttnn.embedding``,
    same discipline as the P7 atom-transformer key gather);
  * a weight-only ``layer_norm`` (c_atom=128, eps=1e-5, no offset);
  * ``linear_q_out`` (c_atom=128 -> 3, bias-free).

The mask-derived atom windowing (``convert_single_rep_to_blocks`` ->
``key_block_idxs`` / ``invalid_mask`` / ``mask_trunked``) is precomputed on host in the
golden and replayed on device -- identical isolation to the encoder leg. Padded atom
positions (n_atom -> NP = nb*N_QUERY) are zeroed via ``atom_mask_col`` so the additive
broadcast and the per-row layer_norm do not leak into real atoms.
"""
import torch
import ttnn

from .tenstorrent import Module, CORE_GRID_MAIN
from .openfold3_atom_transformer import OF3AtomTransformer
from .openfold3_weights import _sub


class OF3AtomAttentionDecoder(Module):
    """Device port of OF3 ``AtomAttentionDecoder`` (Algorithm 6).

    Inputs (device bf16):
        ai:  [1, N_tok_pad, 768]  token rep (LN_a(DiT(ai))), seq-padded to a tile.
        ql:  [1, NP, 128]         encoder atom single rep, atom-padded (zeros at pad).
        cl:  [1, NP, 128]         atom conditioning, atom-padded (zeros at pad).
        plm: [1, nb, 32, 128, 16] blocked atom pair (fixed across decoder blocks).
        atom_mask_col:        [1, NP, 1] device.
        atom_to_token_index:  [1, NP] uint32 device (real atoms -> real token idx;
                                   padded atoms -> 0, zeroed by atom_mask_col).
        key_block_idxs_tt:    [1, nb*128] uint32 device.
        valid_mask:           [1, nb, 128, 1] device.
        mask_bias:            [1, nb, 1, 32, 128] device.
        n_atom, NP, nb: ints.

    Output (device bf16):
        rl_update: [1, n_atom, 3]
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w_q_in = self.torch_to_tt("linear_q_in.weight")    # (128, 768) tiled
        self.w_q_out = self.torch_to_tt("linear_q_out.weight")  # (3, 128) tiled
        self.ln_w = self.torch_to_tt("layer_norm.weight", transform=lambda x: x)
        self.at = OF3AtomTransformer(_sub(state_dict, "atom_transformer"),
                                     compute_kernel_config)

    def __call__(self, ai, ql, cl, plm, atom_mask_col, atom_to_token_index_tt,
                 key_block_idxs_tt, valid_mask, mask_bias, n_atom, NP, nb):
        # token -> atom broadcast: linear_q_in(ai) then gather by atom_to_token_index.
        q_in_tok = ttnn.linear(ai, self.w_q_in,
                               compute_kernel_config=self.compute_kernel_config,
                               core_grid=CORE_GRID_MAIN)  # [1, N_tok_pad, 128]
        q_in_2d = ttnn.reshape(ttnn.to_layout(q_in_tok, ttnn.ROW_MAJOR_LAYOUT),
                               (q_in_tok.shape[1], 128))
        bcast = ttnn.embedding(atom_to_token_index_tt, q_in_2d,
                               layout=ttnn.ROW_MAJOR_LAYOUT,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [NP, 128]
        bcast = ttnn.to_layout(ttnn.reshape(bcast, (1, NP, 128)), ttnn.TILE_LAYOUT)
        bcast = ttnn.multiply(bcast, atom_mask_col)  # zero padded atoms
        ttnn.deallocate(q_in_tok)

        ql_aug = ttnn.add(ql, bcast)
        ttnn.deallocate(bcast)

        # 3-block windowed cross-attention (reuses the gated OF3AtomTransformer).
        ql_out = self.at(ql_aug, cl, plm, atom_mask_col, key_block_idxs_tt,
                         valid_mask, mask_bias, n_atom, NP, nb)  # [1, n_atom, 128]
        ttnn.deallocate(ql_aug)

        # weight-only layer_norm + linear_q_out (c_atom -> 3); pad to NP for tiling,
        # slice back to n_atom. LN is per-row so padded (zero) rows do not affect reals.
        ql_out_pad = self._pad_atoms(ql_out, n_atom, NP)
        ttnn.deallocate(ql_out)
        ln = ttnn.layer_norm(ql_out_pad, weight=self.ln_w, epsilon=1e-5,
                             compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(ql_out_pad)
        rl = ttnn.linear(ln, self.w_q_out,
                         compute_kernel_config=self.compute_kernel_config,
                         core_grid=CORE_GRID_MAIN)  # [1, NP, 3]
        ttnn.deallocate(ln)
        rl = ttnn.to_layout(rl, ttnn.ROW_MAJOR_LAYOUT)
        rl = ttnn.slice(rl, [0, 0, 0], [1, n_atom, 3])
        return ttnn.to_layout(rl, ttnn.TILE_LAYOUT)

    @staticmethod
    def _pad_atoms(x, n_atom, NP):
        # x: [1, n_atom, 128] device -> [1, NP, 128] device (zero-padded).
        th = ttnn.to_torch(x).float()  # [1, n_atom, 128]
        if NP > n_atom:
            th = torch.nn.functional.pad(th, (0, 0, 0, NP - n_atom))
        return ttnn.from_torch(th, layout=ttnn.TILE_LAYOUT,
                               device=x.device(), dtype=ttnn.bfloat16)
