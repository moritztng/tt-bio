"""OF3 token-level DiffusionTransformer device port (P8, leg 3).

OF3 ``DiffusionTransformer`` (AF3 Algorithm 23, non-cross-attention path): a 24-block
stack of ``DiffusionTransformerBlock``, each block = ``AttentionPairBias`` (with
``use_ada_layer_norm=True``) + ``ConditionedTransitionBlock``.

Per block (``a`` evolves, ``s``/``z``/``mask`` are fixed across the stack):

    a_ln = AdaLN(a, s)                                   # c_a=768, c_s=384
    z_b  = permute(linear_z(LN_z(z)), [0,3,1,2])         # [1,16,N,N] per-block pair bias
    o    = MHA(a_ln, a_ln, biases=[mask_bias, z_b])       # 16 heads, head_dim=48
    o    = linear_o(o * sigmoid(linear_g(a_ln)))         # query gate (mha), output proj
    o    = o * sigmoid(linear_ada_out(s))                 # APB output gate
    a    = a + o
    a_t  = AdaLN_t(a, s)
    tr   = linear_out(silu(linear_a(a_t)) * linear_b(a_t)) * sigmoid(linear_g_t(s))
    a    = a + (tr * tok_mask)

``AttentionPairBias`` here is the non-cross variant: full N-token self-attention (no
windowing, no block gather), with a per-block weight-only ``layer_norm_z`` +
``linear_z`` pair-bias projection (the cross-attention ``AtomTransformer`` instead
applies one top-level ``layer_norm_z`` shared across blocks). The shared ``AdaLN``
math (a = sigmoid(Wg(LN_s(s))) * LN(a) + Ws(LN_s(s))) is identical to the
``AtomTransformer``'s, so the conditioning submodules reuse ``tenstorrent.AdaLN`` via
``remap_of3_adaln`` -- no primitive is reimplemented. The SwiGLU transition is the
same math the trunk transition and the ``AtomTransformer`` conditioned transition use.

head_dim=48 is not tile-aligned, so the q/k/v projections are fused and padded to
head_dim=64 (16 heads -> 1024/head-group, fused qkv -> 3072), then
``nlp_create_qkv_heads`` splits the padded heads. Attention itself is MANUAL (matmul
QK^T + scale + mask, fp32 numerically-stable softmax, matmul attn@V), NOT the fused
``scaled_dot_product_attention``: the fused SDPA does its softmax in bf16, and its
per-block error (~0.998) compounds to ~0.967 over the 24-block stack; a CPU bf16
control with an fp32 softmax holds 0.99996 over the same stack, isolating the softmax
precision as the lever. Computing the softmax in fp32 (scores cast up, softmax, cast
back) closes the gap -- the device stack tracks the reference at >=0.9998 at every
block. ``linear_z`` is a plain linear (no head_dim scaling, unlike the protenix pair
bias).

Two correctness details the port had to get right, both invisible at the block level
and fatal at the stack level: (1) the additive attention mask must cover the
tile-padded key positions -- ``from_torch`` pads *storage* with 0 (="unmasked" for an
additive mask), so a logical-N mask leaves the tile-padded keys unmasked and they leak
garbage into valid queries, compounding to a ~0.30 stack PCC collapse; the stack pads
its inputs to the tile width and marks padded keys -1e9; (2) the fp32 softmax above.

Golden: ``scripts/of3_diffusion_transformer_golden.py`` instantiates the full
``DiffusionModule``, forward-hooks ``diffusion_transformer`` to capture its exact
``(a, s, z, mask)`` input and ``a`` output (block 0 + full 24-block stack), and adds
``diffusion_transformer_real`` to ``~/of3_ref_out.pkl`` -- so the device port is
PCC-gated against the exact reference artifacts, isolating the device block precision
from the atom-encoder/conditioning host math.
"""
from __future__ import annotations

import torch
import ttnn

from .tenstorrent import Module, AdaLN, CORE_GRID_MAIN
from .openfold3_atom_transformer import remap_of3_adaln

C_A = 768
C_S = 384
C_Z = 128
N_HEADS = 16
HEAD_DIM = 48
PADDED_HEAD_DIM = 64  # -48 % 32 = 16 -> 64 (tile-aligned)
N_BLOCKS = 24


def _sub(sd: dict, prefix: str) -> dict:
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def _pad_single(t: ttnn.Tensor, padded_N: int) -> ttnn.Tensor:
    """Device [1, N, C] -> [1, padded_N, C] (zero-padded), logical width padded_N."""
    N = t.shape[1]
    if padded_N == N:
        return t
    x = ttnn.to_torch(t)  # [1, N, C] (logical, to_torch strips tile padding)
    x = torch.nn.functional.pad(x, (0, 0, 0, padded_N - N), value=0.0)
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=t.device(), dtype=ttnn.bfloat16)


def _pad_pair(t: ttnn.Tensor, padded_N: int) -> ttnn.Tensor:
    """Device [1, N, N, C] -> [1, padded_N, padded_N, C] (zero-padded)."""
    N = t.shape[1]
    if padded_N == N:
        return t
    x = ttnn.to_torch(t)
    x = torch.nn.functional.pad(x, (0, 0, 0, padded_N - N, 0, padded_N - N), value=0.0)
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=t.device(), dtype=ttnn.bfloat16)


class _DiTBlock(Module):
    """One OF3 DiffusionTransformerBlock (Algorithm 23, non-cross path)."""

    def __init__(self, sd_block: dict, compute_kernel_config):
        super().__init__({}, compute_kernel_config)  # weights loaded via _w_tt, not torch_to_tt
        self._w = sd_block
        self._wc: dict = {}

        apb = "attention_pair_bias."
        self.adaln_a = AdaLN(False, remap_of3_adaln(_sub(self._w, apb + "layer_norm_a")),
                             compute_kernel_config)
        self.ln_z_w = self._w_tt(apb + "layer_norm_z.weight", False)
        self.w_lin_z = self._w_tt(apb + "linear_z.weight")
        self.w_ada_out = self._w_tt(apb + "linear_ada_out.weight")
        self.b_ada_out = self._w_tt(apb + "linear_ada_out.bias", False)

        # Fused padded qkv: cat([linear_q, linear_k, linear_v]) with head_dim 48->64.
        wq = self._w[apb + "mha.linear_q.weight"]
        bq = self._w[apb + "mha.linear_q.bias"]
        wk = self._w[apb + "mha.linear_k.weight"]
        wv = self._w[apb + "mha.linear_v.weight"]
        qkv_w = torch.cat([wq, wk, wv], dim=0)  # [3*768, 768]
        qkv_w = qkv_w.reshape(3 * N_HEADS, HEAD_DIM, -1)
        qkv_w = torch.nn.functional.pad(qkv_w, (0, 0, 0, PADDED_HEAD_DIM - HEAD_DIM), value=0)
        qkv_w = qkv_w.reshape(3 * N_HEADS * PADDED_HEAD_DIM, -1)  # [3072, 768]
        self.qkv_w = ttnn.from_torch(qkv_w.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                                     device=self.device, dtype=ttnn.bfloat16)
        qb = bq.reshape(N_HEADS, HEAD_DIM)
        qb = torch.nn.functional.pad(qb, (0, PADDED_HEAD_DIM - HEAD_DIM), value=0)
        qb = qb.reshape(N_HEADS * PADDED_HEAD_DIM)
        qkv_b = torch.cat([qb, torch.zeros(2 * N_HEADS * PADDED_HEAD_DIM, dtype=qb.dtype)])
        self.qkv_b = ttnn.from_torch(qkv_b, layout=ttnn.TILE_LAYOUT, device=self.device,
                                     dtype=ttnn.bfloat16)

        self.w_g = self._w_tt(apb + "mha.linear_g.weight")
        self.w_o = self._w_tt(apb + "mha.linear_o.weight")

        ct = "conditioned_transition."
        self.adaln_t = AdaLN(False, remap_of3_adaln(_sub(self._w, ct + "layer_norm")),
                             compute_kernel_config)
        self.w_la = self._w_tt(ct + "swiglu.linear_a.weight")
        self.w_lb = self._w_tt(ct + "swiglu.linear_b.weight")
        self.w_lout = self._w_tt(ct + "linear_out.weight")
        self.w_lg = self._w_tt(ct + "linear_g.weight")
        self.b_lg = self._w_tt(ct + "linear_g.bias", False)

    def _w_tt(self, key, transpose=True):
        v = self._wc.get((key, transpose))
        if v is None:
            w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
            self._wc[(key, transpose)] = v
        return v

    def _lin(self, x, w, bias=None, activation=None):
        return ttnn.linear(x, w, bias=bias, activation=activation,
                           compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)

    def __call__(self, a, s, z, mask_bias, tok_mask_col):
        lin = self._lin
        # AdaLN-conditioned a.
        a_ln = self.adaln_a(a, s)

        # Per-block pair bias: LN_z(z) -> linear_z -> [1,16,N,N].
        z = ttnn.layer_norm(z, weight=self.ln_z_w, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        zb = lin(z, self.w_lin_z)                       # [1, N, N, 16]
        zb = ttnn.to_layout(zb, ttnn.ROW_MAJOR_LAYOUT)
        zb = ttnn.permute(zb, (0, 3, 1, 2))             # [1, 16, N, N]
        zb = ttnn.to_layout(zb, ttnn.TILE_LAYOUT)
        ttnn.deallocate(z)
        zb = ttnn.add_(zb, mask_bias)                   # + mask_bias [1,1,1,N]

        # Fused padded qkv -> heads.
        qkv = lin(a_ln, self.qkv_w, bias=self.qkv_b)   # [1, N, 3072]
        qkv = ttnn.unsqueeze(qkv, 1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=N_HEADS, num_kv_heads=N_HEADS, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        # Manual attention with an fp32 softmax: the fused SDPA does softmax in bf16,
        # and its per-block error (~0.998) compounds to ~0.967 over 24 blocks. A CPU
        # bf16 control with an fp32 softmax holds 0.99996 over the same stack, so the
        # softmax precision is the lever. Compute scores in bf16 (fp32 dest acc), cast
        # up for a numerically-stable fp32 softmax, cast back for attn@V.
        scale = HEAD_DIM ** -0.5
        sc = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)),
                         compute_kernel_config=self.compute_kernel_config)
        sc = ttnn.multiply(sc, scale)
        sc = ttnn.add(sc, zb)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(zb)
        sc = ttnn.typecast(sc, ttnn.float32)
        attn = ttnn.softmax(sc, dim=-1, numeric_stable=True)
        ttnn.deallocate(sc)
        attn = ttnn.typecast(attn, ttnn.bfloat16)
        o = ttnn.matmul(attn, v, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(attn); ttnn.deallocate(v)
        # Slice padded head_dim 64->48, merge heads -> [1, N, 768].
        o = o[:, :, :, :HEAD_DIM]
        o = ttnn.permute(o, (0, 1, 3, 2))               # [1, 16, 48, N]
        o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))  # [1, 768, N]
        o = ttnn.permute(o, (0, 2, 1))                  # [1, N, 768]
        # Query gate (flat == per-head: g.view(N,H,d) * o(H,N,d) == flat multiply).
        g = lin(a_ln, self.w_g)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        o = lin(o, self.w_o)                            # [1, N, 768]
        # APB output gate from s.
        og = lin(s, self.w_ada_out, bias=self.b_ada_out)
        o = ttnn.multiply(o, og, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(og); ttnn.deallocate(a_ln)
        a = ttnn.add(a, o)
        ttnn.deallocate(o)

        # Conditioned SwiGLU transition.
        a_t = self.adaln_t(a, s)
        b1 = lin(a_t, self.w_la, activation="silu")
        b2 = lin(a_t, self.w_lb)
        ttnn.deallocate(a_t)
        bb = ttnn.multiply(b1, b2)
        ttnn.deallocate(b1); ttnn.deallocate(b2)
        out = lin(bb, self.w_lout)
        ttnn.deallocate(bb)
        lg = lin(s, self.w_lg, bias=self.b_lg)
        out = ttnn.multiply(out, lg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(lg)
        out = ttnn.multiply(out, tok_mask_col)
        a = ttnn.add(a, out)
        ttnn.deallocate(out)
        return a


class OF3DiffusionTransformer(Module):
    """OF3 ``DiffusionTransformer`` (Algorithm 23, non-cross path) on device.

    Inputs (device bf16):
        a:        [1, N, 768]   token single (evolving)
        s:        [1, N, 384]   conditioning single (si, fixed)
        z:        [1, N, N, 128] conditioning pair (zij, fixed)
        mask_bias:[1, 1, 1, N]  additive attention mask (inf*(token_mask-1))
        tok_mask_col: [1, N, 1] token mask for transition masking
    Returns [1, N, 768].
    """

    def __init__(self, state_dict, compute_kernel_config, n_blocks=N_BLOCKS):
        super().__init__(state_dict, compute_kernel_config)
        self._w = {k: v for k, v in self.weights.data.items()}
        self.blocks = [_DiTBlock(_sub(self._w, f"blocks.{b}"), compute_kernel_config)
                       for b in range(n_blocks)]
        self.n_blocks = n_blocks

    def __call__(self, a, s, z, token_mask, tok_mask_col):
        # Pad to the tile-aligned logical width so the SDPA's tiled key extent is
        # fully masked. from_torch pads *storage* with 0 (="unmasked" for an additive
        # mask), so a logical-N mask leaves the tile-padded keys (N -> ceil(N/32)*32)
        # unmasked and they leak garbage into valid queries, compounding across the
        # 24 blocks (stack PCC collapses to ~0.3). Padding the inputs to the tile
        # width here and marking padded keys -1e9 in mask_bias closes the leak; valid
        # positions are unaffected (padded queries' outputs are stripped at readout,
        # padded keys are masked out of every valid query's softmax).
        N = token_mask.shape[-1]
        padded_N = ((N + 31) // 32) * 32
        tok = ttnn.to_torch(token_mask).float().reshape(-1)  # [N]
        if padded_N == N:
            a_d, s_d, z_d, tmc_d = a, s, z, tok_mask_col
            mb_t = (tok - 1.0) * 1e9
        else:
            a_d = _pad_single(a, padded_N)
            s_d = _pad_single(s, padded_N)
            z_d = _pad_pair(z, padded_N)
            tok_p = torch.zeros(padded_N, dtype=torch.float32)
            tok_p[:N] = tok
            tmc_p = tok_p.reshape(1, padded_N, 1)
            tmc_d = ttnn.from_torch(tmc_p, layout=ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=ttnn.bfloat16)
            mb_t = torch.full((padded_N,), -1e9, dtype=torch.float32)
            mb_t[:N] = (tok - 1.0) * 1e9
        mb = mb_t.reshape(1, 1, 1, padded_N)
        mask_bias = ttnn.from_torch(mb, layout=ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=ttnn.bfloat16)
        x = a_d
        for blk in self.blocks:
            x = blk(x, s_d, z_d, mask_bias, tmc_d)
        if padded_N == N:
            return x
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.slice(x, [0, 0, 0], [1, N, C_A])
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
