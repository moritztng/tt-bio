"""OF3 AtomTransformer device port (P7).

OF3's ``atom_transformer`` is a ``DiffusionTransformer(cross_attention_mode=True)``:
3 blocks of windowed sequence-local atom attention (n_query=32, n_key=128, 4 heads,
head_dim=32). Each block: AdaLN-conditioned q/kv -> pair-biased windowed attention
with a query gate (``mha.linear_g``) -> ``sigmoid(linear_ada_out(s))`` output gate ->
residual -> AdaLN-conditioned SwiGLU transition (``swiglu.linear_a/b`` + ``linear_out``,
``sigmoid(linear_g(s))`` zero gate) -> residual. The pair ``plm`` is layer-normed once
at the top (``layer_norm_z``) and a per-block ``linear_z`` projects the pair bias.

The block topology is OF3-specific (``linear_ada_out`` output gate, ``mha.linear_g``
query gate, ``swiglu.linear_a/b`` + ``linear_out`` transition) and is NOT a key-remap
onto ``protenix.AtomTransformer`` -- this is a fresh device port. The shared ``AdaLN``
math (a = LN(a); a = sigmoid(Wg(LN_s(s))) * a + Ws(LN_s(s))) is identical, so the nine
conditioning submodules reuse ``tenstorrent.AdaLN`` via ``remap_of3_adaln``.

The mask-derived block gather (OF3 ``convert_single_rep_to_blocks``: centered key
windows with underflow/overflow shift) is precomputed on host (``key_block_idxs``,
``invalid_mask``, ``mask_trunked``) and replayed on device via ``ttnn.embedding`` --
the device re-blocks the evolving single ``a`` every block with the fixed gather
indices, so the port is reusable: the diffusion atom encoder/decoder use the same class
with ``a != s``. See ``scripts/of3_atom_transformer_golden.py`` for the golden and
``tests/test_openfold3_atom_transformer.py`` for the PCC gate.
"""
from __future__ import annotations

import torch
import ttnn

from .tenstorrent import Module, AdaLN, CORE_GRID_MAIN


def remap_of3_adaln(sd: dict) -> dict:
    """OF3 AdaLN keys -> tenstorrent.AdaLN keys.

    OF3 AdaLN: a = LayerNorm(a); g = sigmoid(linear_g(LayerNorm_s(s)));
    a = g * a + linear_s(LayerNorm_s(s)). ``LayerNorm(c_s, create_offset=False)`` is
    weight-only, matching tenstorrent.AdaLN's bias-free s layer norm; the a layer norm
    is weightless on both sides. ``linear_s`` is bias-free in the OF3 checkpoint, so
    s_bias (the shift term) carries no bias -- exactly tenstorrent.AdaLN's layout.
    """
    return {
        "s_norm.weight": sd["layer_norm_s.weight"],
        "s_scale.weight": sd["linear_g.weight"],
        "s_scale.bias": sd["linear_g.bias"],
        "s_bias.weight": sd["linear_s.weight"],
    }


def _sub(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class OF3AtomTransformer(Module):
    N_HEADS = 4
    HEAD_DIM = 32
    N_QUERY = 32
    N_KEY = 128

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self._w = {k: v for k, v in self.weights.data.items()}
        self._wc: dict = {}
        self.ln_z_w = self._w_tt("layer_norm_z.weight", False)
        self.adaln_q = [
            AdaLN(False, remap_of3_adaln(_sub(self._w, f"blocks.{b}.attention_pair_bias.layer_norm_a_q.")),
                  compute_kernel_config) for b in range(3)]
        self.adaln_k = [
            AdaLN(False, remap_of3_adaln(_sub(self._w, f"blocks.{b}.attention_pair_bias.layer_norm_a_k.")),
                  compute_kernel_config) for b in range(3)]
        self.adaln_t = [
            AdaLN(False, remap_of3_adaln(_sub(self._w, f"blocks.{b}.conditioned_transition.layer_norm.")),
                  compute_kernel_config) for b in range(3)]

    def _w_tt(self, key, transpose=True):
        v = self._wc.get((key, transpose))
        if v is None:
            w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
            self._wc[(key, transpose)] = v
        return v

    def _lin(self, x, wkey, bkey=None, activation=None):
        return ttnn.linear(x, self._w_tt(wkey), bias=(self._w_tt(bkey, False) if bkey else None),
                           activation=activation, compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)

    def _heads(self, x, n_blk, n_seq):
        # x: [1, n_blk, n_seq, 128] -> [1, n_blk, H, n_seq, dh]
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (1, n_blk, n_seq, self.N_HEADS, self.HEAD_DIM))
        x = ttnn.permute(x, (0, 1, 3, 2, 4))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _gather_keys(self, x, key_block_idxs_tt, valid_mask, nb):
        # x: [1, NP, 128] -> [1, nb, N_KEY, 128] via fixed host gather indices.
        x2d = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (x.shape[1], 128))
        xk = ttnn.embedding(key_block_idxs_tt, x2d, layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xk = ttnn.reshape(xk, (1, nb, self.N_KEY, 128))
        xk = ttnn.to_layout(xk, ttnn.TILE_LAYOUT)
        return ttnn.multiply(xk, valid_mask)

    def __call__(self, a, s, z, atom_mask_col, key_block_idxs_tt, valid_mask,
                 mask_bias, n_atom, NP, nb):
        """a, s: [1, NP, 128] device (padded to a multiple of N_QUERY; a_init = cl when
        rl=None); z: [1, nb, N_QUERY, N_KEY, 16] device (blocked pair = plm);
        atom_mask_col: [1, NP, 1] device; key_block_idxs_tt: [1, nb*N_KEY] uint32 device;
        valid_mask: [1, nb, N_KEY, 1] device; mask_bias: [1, nb, 1, N_QUERY, N_KEY] device.
        Returns [1, n_atom, 128]."""
        scale = self.HEAD_DIM ** -0.5
        nq, nk, H, dh = self.N_QUERY, self.N_KEY, self.N_HEADS, self.HEAD_DIM

        z_ln = ttnn.layer_norm(z, weight=self.ln_z_w, epsilon=1e-5,
                               compute_kernel_config=self.compute_kernel_config)
        z_bias = []
        for b in range(3):
            zb = self._lin(z_ln, f"blocks.{b}.attention_pair_bias.linear_z.weight")
            zb = ttnn.to_layout(zb, ttnn.ROW_MAJOR_LAYOUT)
            zb = ttnn.permute(ttnn.reshape(zb, (1, nb, nq, nk, H)), (0, 1, 4, 2, 3))
            z_bias.append(ttnn.to_layout(zb, ttnn.TILE_LAYOUT))
        ttnn.deallocate(z_ln)

        # s blocks are fixed across blocks (s = cl does not evolve).
        s_q = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
        s_q = ttnn.reshape(s_q, (1, nb, nq, 128))
        s_q = ttnn.to_layout(s_q, ttnn.TILE_LAYOUT)
        s_k = self._gather_keys(s, key_block_idxs_tt, valid_mask, nb)

        x = a
        for b in range(3):
            P = f"blocks.{b}."
            apb = P + "attention_pair_bias."
            x_q = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x_q = ttnn.reshape(x_q, (1, nb, nq, 128))
            x_q = ttnn.to_layout(x_q, ttnn.TILE_LAYOUT)
            x_k = self._gather_keys(x, key_block_idxs_tt, valid_mask, nb)

            a_qn = self.adaln_q[b](x_q, s_q)
            a_kn = self.adaln_k[b](x_k, s_k)
            Q = self._heads(self._lin(a_qn, apb + "mha.linear_q.weight", apb + "mha.linear_q.bias"), nb, nq)
            K = self._heads(self._lin(a_kn, apb + "mha.linear_k.weight"), nb, nk)
            V = self._heads(self._lin(a_kn, apb + "mha.linear_v.weight"), nb, nk)
            sc = ttnn.matmul(Q, ttnn.permute(K, (0, 1, 2, 4, 3)),
                             compute_kernel_config=self.compute_kernel_config)
            sc = ttnn.multiply(sc, scale)
            sc = ttnn.add(sc, mask_bias)
            sc = ttnn.add(sc, z_bias[b])
            attn = ttnn.softmax(sc, dim=-1)
            o = ttnn.matmul(attn, V, compute_kernel_config=self.compute_kernel_config)
            ttnn.deallocate(sc); ttnn.deallocate(attn)
            # o: [1,nb,H,Q,dh] -> [1,nb,Q,H,dh]; gate with sigmoid(linear_g(a_qn)).
            o = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
            o = ttnn.permute(o, (0, 1, 3, 2, 4))
            o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
            g_raw = self._lin(a_qn, apb + "mha.linear_g.weight")
            g_raw = ttnn.to_layout(g_raw, ttnn.ROW_MAJOR_LAYOUT)
            g_raw = ttnn.reshape(g_raw, (1, nb, nq, H, dh))
            g_raw = ttnn.to_layout(g_raw, ttnn.TILE_LAYOUT)
            o = ttnn.multiply(o, g_raw, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            o = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
            o = ttnn.reshape(o, (1, nb, nq, 128))
            o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
            o = self._lin(o, apb + "mha.linear_o.weight")
            o = ttnn.reshape(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), (1, NP, 128))
            o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
            ada_raw = self._lin(s, apb + "linear_ada_out.weight", apb + "linear_ada_out.bias")
            o = ttnn.multiply(o, ada_raw, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(ada_raw)
            x = ttnn.add(x, o)
            ttnn.deallocate(o)

            ct = P + "conditioned_transition."
            an = self.adaln_t[b](x, s)
            b1 = self._lin(an, ct + "swiglu.linear_a.weight", activation="silu")
            b2 = self._lin(an, ct + "swiglu.linear_b.weight")
            bb = ttnn.multiply(b1, b2)
            ttnn.deallocate(b1); ttnn.deallocate(b2)
            out = self._lin(bb, ct + "linear_out.weight")
            ttnn.deallocate(bb)
            cg_raw = self._lin(s, ct + "linear_g.weight", ct + "linear_g.bias")
            out = ttnn.multiply(out, cg_raw, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(cg_raw)
            out = ttnn.multiply(out, atom_mask_col)
            x = ttnn.add(x, out)
            ttnn.deallocate(out); ttnn.deallocate(a_qn); ttnn.deallocate(a_kn)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.slice(x, [0, 0, 0], [1, n_atom, 128])
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
