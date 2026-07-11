"""OpenFold (AlphaFold2) — Tenstorrent port.

Net-new AF2-specific device blocks live here; the O(L²)/O(L³) pair-track heavy ops
(TriangleMultiplication, TriangleAttention, OuterProductMean) are reused directly from
tt_bio.tenstorrent (PCC-verified — see docs/openfold-port.md). Weight key names follow
the vendored reference (tt_bio/_vendor/openfold), so most blocks need no remap.
"""
from __future__ import annotations

import torch
import ttnn

from tt_bio.tenstorrent import (
    Module, get_device, CORE_GRID_MAIN,
    TriangleMultiplication, TriangleAttention, OuterProductMean,
)


class ReluTransition(Module):
    """AF2 PairTransition / MSATransition (Algorithm 9/15 style feed-forward):

        LayerNorm -> Linear(c -> n*c) -> ReLU -> Linear(n*c -> c)

    A plain ReLU MLP — distinct from the AF3 gated-SwiGLU tt_bio.tenstorrent.Transition,
    so it cannot reuse that block. Weight keys match the reference module directly
    (layer_norm.{weight,bias}, linear_1.{weight,bias}, linear_2.{weight,bias}).
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_w = self.torch_to_tt("layer_norm.weight")
        self.norm_b = self.torch_to_tt("layer_norm.bias")
        self.w1 = self.torch_to_tt("linear_1.weight")
        self.b1 = self.torch_to_tt("linear_1.bias", lambda x: x.reshape(1, -1))
        self.w2 = self.torch_to_tt("linear_2.weight")
        self.b2 = self.torch_to_tt("linear_2.bias", lambda x: x.reshape(1, -1))

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.layer_norm(
            x, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        h = ttnn.linear(
            x, self.w1, bias=self.b1, activation="relu",
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16,
        )
        out = ttnn.linear(
            h, self.w2, bias=self.b2,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        return out


class _MSAGatedAttention(Module):
    """Shared core for AF2 MSA gated attention. Given m2d [B, L, C_m] (B = batch axis,
    L = attention axis) and an optional per-head additive bias [1, H, L, L], applies:
    LayerNorm -> gated multi-head self-attention over L -> output projection. Mirrors
    the shared TriangleAttention mechanics; when a pair bias is used it is pre-scaled by
    head_dim**0.5 so ttnn sdpa's scale(=head_dim**-0.5) over (QK+mask) yields AF2's
    QK/sqrt(hd) + linear_z(z). q/k/v are bias-free (matches AF2); o/g biases dropped
    (gated o/g bias is the tracked real-weight follow-up)."""

    def __init__(self, head_dim, n_heads, state_dict, compute_kernel_config, pair_bias):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.scale = head_dim ** 0.5
        self.pair_bias = pair_bias
        self.norm_m_w = self.torch_to_tt("layer_norm_m.weight")
        self.norm_m_b = self.torch_to_tt("layer_norm_m.bias")
        self.qkv_weight = ttnn.from_torch(
            torch.cat(
                [self.weights["mha.linear_q.weight"],
                 self.weights["mha.linear_k.weight"],
                 self.weights["mha.linear_v.weight"]], dim=0,
            ).t(),
            layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16,
        )
        self.g_weight = self.torch_to_tt("mha.linear_g.weight")
        self.o_weight = self.torch_to_tt("mha.linear_o.weight")
        if pair_bias:
            self.norm_z_w = self.torch_to_tt("layer_norm_z.weight")
            self.norm_z_b = self.torch_to_tt("layer_norm_z.bias")
            self.z_weight = ttnn.multiply_(self.torch_to_tt("linear_z.weight"), self.scale)

    def _lnorm(self, x, w, b):
        return ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5,
                               compute_kernel_config=self.compute_kernel_config)

    def _proj(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16)

    def _bias(self, z):  # z: [L, L, C_z] -> [1, H, L, L]
        b = self._proj(self._lnorm(z, self.norm_z_w, self.norm_z_b), self.z_weight)
        return ttnn.unsqueeze(ttnn.permute(b, (2, 0, 1)), 0)

    def _attend(self, m2d, bias):  # m2d: [B, L, C_m] -> [B, L, C_m]
        mn = self._lnorm(m2d, self.norm_m_w, self.norm_m_b)
        qkv = ttnn.unsqueeze(self._proj(mn, self.qkv_weight), 1)  # [B,1,L,3*H*hd]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.n_heads, num_kv_heads=self.n_heads,
            transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=self.scale ** -1)
        for t in (q, k, v):
            ttnn.deallocate(t)
        if bias is not None:
            ttnn.deallocate(bias)
        o = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG), 1)
        g = self._proj(mn, self.g_weight)
        ttnn.deallocate(mn)
        o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        out = self._proj(o, self.o_weight)
        ttnn.deallocate(o)
        return out


class MSARowAttentionWithPairBias(_MSAGatedAttention):
    """AF2 Algorithm 7: per MSA row (N_seq = batch), gated attention over residues
    (N_res) with additive per-head bias from the pair tensor z. Keys direct
    (layer_norm_m, layer_norm_z, linear_z, mha.linear_{q,k,v,o,g})."""

    def __init__(self, head_dim, n_heads, state_dict, compute_kernel_config):
        super().__init__(head_dim, n_heads, state_dict, compute_kernel_config, pair_bias=True)

    def __call__(self, m: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        m = ttnn.reshape(m, tuple(m.shape)[1:])   # [N_seq, N_res, C_m]
        z = ttnn.reshape(z, tuple(z.shape)[1:])   # [N_res, N_res, C_z]
        return self._attend(m, self._bias(z))


class MSAColumnAttention(_MSAGatedAttention):
    """AF2 Algorithm 8: per residue column (N_res = batch), gated attention over
    sequences (N_seq), no pair bias. Reference wraps its attention in `_msa_att`, so the
    remap strips that prefix onto the shared core's flat keys."""

    def __init__(self, head_dim, n_heads, state_dict, compute_kernel_config):
        sd = {k[len("_msa_att."):]: v for k, v in state_dict.items() if k.startswith("_msa_att.")}
        super().__init__(head_dim, n_heads, sd, compute_kernel_config, pair_bias=False)

    def __call__(self, m: ttnn.Tensor) -> ttnn.Tensor:
        m = ttnn.reshape(m, tuple(m.shape)[1:])        # [N_seq, N_res, C_m]
        m = ttnn.permute(m, (1, 0, 2))                 # [N_res, N_seq, C_m] (attend over N_seq)
        out = self._attend(m, None)
        return ttnn.permute(out, (1, 0, 2))            # back to [N_seq, N_res, C_m]


class EvoformerBlock:
    """One AF2 Evoformer block on device: the verified sub-blocks composed in the
    reference EvoformerBlock.forward residual order (inference, opm_first=False, no
    dropout/mask). Sub-block state_dicts are supplied pre-remapped by the caller
    (`row`, `col`, `msa_transition`, `opm`, `tri_mul_out`, `tri_mul_in`,
    `tri_att_start`, `tri_att_end`, `pair_transition`)."""

    def __init__(self, sub, hd_pair, h_pair, hd_msa, h_msa, cfg):
        self.row = MSARowAttentionWithPairBias(hd_msa, h_msa, sub["row"], cfg)
        self.col = MSAColumnAttention(hd_msa, h_msa, sub["col"], cfg)
        self.msa_tr = ReluTransition(sub["msa_transition"], cfg)
        self.opm = OuterProductMean(sub["opm"], cfg)
        self.tmo = TriangleMultiplication(False, sub["tri_mul_out"], cfg)
        self.tmi = TriangleMultiplication(True, sub["tri_mul_in"], cfg)
        self.tas = TriangleAttention(hd_pair, h_pair, False, sub["tri_att_start"], cfg)
        self.tae = TriangleAttention(hd_pair, h_pair, True, sub["tri_att_end"], cfg)
        self.pair_tr = ReluTransition(sub["pair_transition"], cfg)

    def __call__(self, m, z):
        _, S, I, cm = m.shape
        cz = z.shape[-1]
        ms, zs = (1, S, I, cm), (1, I, I, cz)
        add = lambda base, out, shp: ttnn.add(base, ttnn.reshape(out, shp))
        m = add(m, self.row(m, z), ms)
        m = add(m, self.col(m), ms)
        m = add(m, self.msa_tr(m), ms)
        z = add(z, self.opm(m, msa_mask=None, n_msa=S), zs)
        z = add(z, self.tmo(z), zs)
        z = add(z, self.tmi(z), zs)
        z = add(z, self.tas(z), zs)
        z = add(z, self.tae(z), zs)
        z = add(z, self.pair_tr(z), zs)
        return m, z


class EvoformerStack:
    """AF2 Evoformer trunk on device: N EvoformerBlocks then the single-rep projection
    s = Linear(c_m -> c_s)(m[..., 0, :, :]). Returns (m, z, s). This is the only heavy
    (O(L^3)) part placed on device; embedders / structure module / heads run as the
    vendored host reference (trunk dominates compute — ESMFold2/Protenix precedent)."""

    def __init__(self, block_subs, s_linear_sd, hd_pair, h_pair, hd_msa, h_msa, cfg):
        self.cfg = cfg
        self.blocks = [EvoformerBlock(sub, hd_pair, h_pair, hd_msa, h_msa, cfg)
                       for sub in block_subs]
        dev = get_device()
        ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        self.s_w = ft(s_linear_sd["weight"].t())
        self.s_b = ft(s_linear_sd["bias"].reshape(1, -1))

    def __call__(self, m, z):
        for blk in self.blocks:
            m, z = blk(m, z)
        _, S, I, cm = m.shape
        first = ttnn.reshape(ttnn.slice(m, [0, 0, 0, 0], [1, 1, I, cm]), (1, I, cm))
        s = ttnn.linear(first, self.s_w, bias=self.s_b, compute_kernel_config=self.cfg,
                        core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16)
        return m, z, s
