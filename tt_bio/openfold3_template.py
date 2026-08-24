"""OF3 TemplateEmbedderAllAtom device port (P8).

OF3 ``TemplateEmbedderAllAtom`` (AF3 Algorithm 16) composes three legs:

  1. ``TemplatePairEmbedderAllAtom`` (feature embedder): eight bias-free linears summed
     into ``a`` [N_templ, N, N, c_t=64] (dgram, pseudo_beta_mask, aatype_1/2, x/y/z unit
     vectors, backbone_mask), plus ``z_bias = linear_z(layer_norm_z(z))`` [1, N, N, c_t]
     shared across templates. ``t_embed = z_bias + a`` [N_templ, N, N, c_t].
  2. ``TemplatePairStack``: two AF2 PairBlocks (tri_mul_out/in + tri_att_start/end +
     swiglu pair_transition, tri_mul_first=True) + a final affine stack layer_norm. The
     block is structurally identical to the MSA module's ``pair_stack`` subtree, so it
     reuses ``PairformerLayer(transform_s=False)`` via ``remap_template_pair_stack``.
     Templates do not interact, so the stack runs per-template (the reference's
     per-template loop is mathematically identical to a batched pass, but the device
     ``TriangleAttention`` reshape assumes a singleton batch dim, so the loop is kept).
  3. Aggregate: ``z_template = linear_t(relu(sum_t(t_stack) / n_templ))`` [1, N, N, c_z].

Reuses the existing ``TriangleMultiplication``/``TriangleAttention``/``Transition``
primitives (the same ones the MSA pair_stack and Protenix-v2 trunk reuse) -- no new
pair primitive. The mask-derived feature products (multichain / pseudo-beta /
backbone-frame pair masks) are precomputed on host and captured in the golden
(``scripts/of3_template_embedder_golden.py``), so the device feature linears are gated
against the exact reference masks, isolating the device linear precision from the mask
logic (same discipline as the RefAtomFeatureEmbedder ``dlm``/``vlm``/``inv_sq_dists``
and the InputEmbedder glue's ``relpos``).

The trunk feeds the template embedder the cycle's ``z = z_init + linear_z(layer_norm_z(z))``
(z starts at zeros); the golden captures the cycle-0 z (a constant shift of ``z_init``)
so the device port is gated against the real trunk input, not a re-computation.
"""
from __future__ import annotations

import ttnn

from .tenstorrent import Module, PairformerLayer, accurate_softmax_site
from .openfold3_weights import remap_template_pair_stack, _sub

# OF3 template_pair_stack dims: c_hidden_tri_att=16, no_heads=4 (config.model_config).
_TRI_DIMS = (16, 4)


class TemplatePairFeatureEmbedder(Module):
    """OF3 ``TemplatePairEmbedderAllAtom`` feature embedder (leg 1).

    Inputs (device bf16):
        feat: dict of per-template feature tensors, each [N_templ, N, N, c_in] (the
              mask products precomputed on host): ``distogram`` (39),
              ``pseudo_beta_pair_mask`` (1), ``restype_ti``/``restype_tj`` (32),
              ``unit_vec_x``/``unit_vec_y``/``unit_vec_z`` (1),
              ``backbone_frame_pair_mask`` (1).
        z:    [1, N, N, c_z=128] -- the cycle trunk pair embedding.

    Outputs (device bf16):
        t_embed: [N_templ, N, N, c_t=64] = z_bias + a
        z_bias:  [1, N, N, c_t=64]       (returned for reuse/debug)
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w_dgram = self.torch_to_tt("dgram_linear.weight")
        self.w_pbm = self.torch_to_tt("pseudo_beta_mask_linear.weight")
        self.w_aa1 = self.torch_to_tt("aatype_linear_1.weight")
        self.w_aa2 = self.torch_to_tt("aatype_linear_2.weight")
        self.w_x = self.torch_to_tt("x_linear.weight")
        self.w_y = self.torch_to_tt("y_linear.weight")
        self.w_z = self.torch_to_tt("z_linear.weight")
        self.w_bb = self.torch_to_tt("backbone_mask_linear.weight")
        self.w_lnz = self.torch_to_tt("layer_norm_z.weight")
        self.b_lnz = self.torch_to_tt("layer_norm_z.bias")
        self.w_linz = self.torch_to_tt("linear_z.weight")

    def features(self, feat):
        """``a``: the eight feature linears summed. A function of ``feat`` alone, so the
        trunk computes it once and hands it back on every later recycle."""
        lin = self._lin
        a = lin(feat["distogram"], self.w_dgram)
        a = ttnn.add(a, lin(feat["pseudo_beta_pair_mask"], self.w_pbm))
        a = ttnn.add(a, lin(feat["restype_ti"], self.w_aa1))
        a = ttnn.add(a, lin(feat["restype_tj"], self.w_aa2))
        a = ttnn.add(a, lin(feat["unit_vec_x"], self.w_x))
        a = ttnn.add(a, lin(feat["unit_vec_y"], self.w_y))
        a = ttnn.add(a, lin(feat["unit_vec_z"], self.w_z))
        return ttnn.add(a, lin(feat["backbone_frame_pair_mask"], self.w_bb))

    def __call__(self, feat, z, a=None):
        lin = self._lin
        zln = ttnn.layer_norm(
            z, weight=self.w_lnz, bias=self.b_lnz, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config)
        z_bias = lin(zln, self.w_linz)                       # [1, N, N, 64]
        if a is None:
            a = self.features(feat)
        ttnn.deallocate(zln)

        t_embed = ttnn.add(a, z_bias)                        # broadcast over N_templ
        return t_embed, z_bias


class TemplatePairStack(Module):
    """OF3 ``TemplatePairStack`` (leg 2): 2 AF2 PairBlocks + final stack layer_norm.

    Input (device bf16): ``t_embed`` [N_templ, N, N, c_t=64].
    Output: ``t_stack`` [N_templ, N, N, c_t=64] (final-LN'd, per template).

    ``pair_mask``/``attn_mask`` are PairformerLayer's ``mask`` and ``attn_mask_start``/``_end``.
    They are None for a target whose token axis is already a tile multiple, and carry the token
    bucket's pad mask otherwise (OF3Fold.fold). The single-chain golden passes None either way.
    """

    def __init__(self, state_dict, compute_kernel_config, transpose_bias: bool = True):
        super().__init__(state_dict, compute_kernel_config)
        remap = remap_template_pair_stack(state_dict, prefix="template_pair_stack")
        self.blocks = [
            PairformerLayer(*_TRI_DIMS, None, None, False, b, compute_kernel_config,
                            scale_pair_bias=False, fp32_softmax=True,
                            transpose_bias=transpose_bias,
                            accurate_softmax=accurate_softmax_site("openfold3.template"))
            for b in remap["blocks"]
        ]
        self.ln_w = self.torch_to_tt("template_pair_stack.layer_norm.weight")
        self.ln_b = self.torch_to_tt("template_pair_stack.layer_norm.bias")

    def __call__(self, t_embed, pair_mask=None, attn_mask=None):
        nt = t_embed.shape[0]
        outs = []
        for t in range(nt):
            v = t_embed[t:t + 1]                            # [1, N, N, c_t]
            for blk in self.blocks:
                v = blk(None, v, pair_mask, attn_mask, attn_mask)[1]
            v = ttnn.layer_norm(
                v, weight=self.ln_w, bias=self.ln_b, epsilon=1e-5,
                compute_kernel_config=self.compute_kernel_config)
            outs.append(v)
        if len(outs) == 1:
            return outs[0]
        return ttnn.concat(outs, dim=0)                     # [N_templ, N, N, c_t]


class TemplateEmbedder(Module):
    """OF3 ``TemplateEmbedderAllAtom`` full device port (legs 1+2+3).

    Input (device bf16): ``feat`` (per-template feature dict, see
    TemplatePairFeatureEmbedder) + ``z`` [1, N, N, c_z=128].
    Output: ``z_template`` [1, N, N, c_z=128] = linear_t(relu(mean_t(t_stack))).
    """

    def __init__(self, state_dict, compute_kernel_config, transpose_bias: bool = True):
        super().__init__(state_dict, compute_kernel_config)
        self.fe = TemplatePairFeatureEmbedder(
            _sub(state_dict, "template_pair_embedder"), compute_kernel_config)
        self.ps = TemplatePairStack(state_dict, compute_kernel_config,
                                    transpose_bias=transpose_bias)
        self.w_lt = self.torch_to_tt("linear_t.weight")

    def features(self, feat):
        """The z-independent half of leg 1, hoistable out of the recycle loop."""
        return self.fe.features(feat)

    def __call__(self, feat, z, slots=None, a=None, pair_mask=None, attn_mask=None):
        """``slots`` maps every original template slot to its row in ``feat`` after
        ``dedup_template_slots``. The aggregation still sums one term per ORIGINAL slot
        and divides by their count, in the original order, so a deduplicated stack is
        bit-identical to the full one: ``((v+v)+v)+v`` packs to bf16 at every step and
        ``3v`` does not fit in 8 mantissa bits, so the sum cannot be collapsed."""
        t_embed, _ = self.fe(feat, z, a)                    # [n_distinct, N, N, 64]
        t_stack = self.ps(t_embed, pair_mask, attn_mask)    # [n_distinct, N, N, 64]
        if slots is None:
            slots = list(range(int(t_stack.shape[0])))
        u = t_stack[slots[0]:slots[0] + 1]
        for t in slots[1:]:
            u = ttnn.add(u, t_stack[t:t + 1])
        u = ttnn.multiply(u, 1.0 / len(slots))
        u = ttnn.relu(u)
        return self._lin(u, self.w_lt)                      # [1, N, N, 128]
