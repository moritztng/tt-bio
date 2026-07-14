"""OF3 DiffusionConditioning device port (P8, leg 2).

OF3 ``DiffusionConditioning`` (AF3 Algorithm 21): produces the conditioned single
``si`` [N, c_s=384] and pair ``zij`` [N, N, c_z=128] that drive the diffusion
transformer, from the trunk outputs plus a noise level ``t``.

    zij = linear_z(LN_z(cat([zij_trunk, relpos])))            # 267 -> 128
    zij = zij + SwiGLUTransition_z(zij) x2                     # masked by pair_token_mask
    si  = linear_s(LN_s(cat([si_trunk, si_input])))            # 833 -> 384
    n   = fourier_emb(0.25 * log(t / sigma_data))              # host, captured in golden
    si  = si + linear_n(LN_n(n)).unsqueeze(-2)                 # 256 -> 384, broadcast
    si  = si + SwiGLUTransition_s(si) x2                       # masked by token_mask

All linears are bias-free in the OF3 checkpoint; the three top LNs (``layer_norm_z``/
``layer_norm_s``/``layer_norm_n``) are weight-only (``create_offset=False``); the four
transition LNs carry weight+bias. The SwiGLU transition is ``silu(linear_a(x)) *
linear_b(x) -> linear_out``, masked, added as a residual -- the same SwiGLU math the
trunk transition and the P7 AtomTransformer conditioned transition use.

The mask-derived relpos (``relpos_complex``, 139-dim) and the Fourier noise embedding
``n`` (256-dim) are host-computed and captured in the golden
(``scripts/of3_diffusion_conditioning_golden.py``), so this module is PCC-gated against
the exact reference artifacts -- isolating the device linear/LN/SwiGLU precision from
the relpos/Fourier host math, the same discipline as the other OF3 golden legs.

This is a fresh OF3 port, NOT a key-remap onto ``protenix.DiffusionConditioning``: OF3's
diffusion conditioning carries its own relpos bin concat and weight-only top LNs, and
feeds the OF3 DiffusionTransformer (ported separately) -- the dims (c_s=384, c_z=128,
c_fourier_emb=256, relpos=139) match but the topology is OF3's.
"""
from __future__ import annotations

import torch
import ttnn

from .tenstorrent import Module, CORE_GRID_MAIN


class _SwiGLUTransition(Module):
    """OF3 ``SwiGLUTransition`` (AF3 Algorithm 11) on device: LN -> SwiGLU -> linear_out,
    masked. Residual add is the caller's responsibility (this returns the update only)."""

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.ln_w = self.torch_to_tt("layer_norm.weight")
        self.ln_b = self.torch_to_tt("layer_norm.bias")
        self.la = self.torch_to_tt("swiglu.linear_a.weight")
        self.lb = self.torch_to_tt("swiglu.linear_b.weight")
        self.lo = self.torch_to_tt("linear_out.weight")

    def __call__(self, x, mask_col):
        x = ttnn.layer_norm(x, weight=self.ln_w, bias=self.ln_b, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        a = self._lin(x, self.la, activation="silu")
        b = self._lin(x, self.lb)
        ttnn.deallocate(x)
        h = ttnn.multiply(a, b)
        ttnn.deallocate(a); ttnn.deallocate(b)
        out = self._lin(h, self.lo)
        ttnn.deallocate(h)
        out = ttnn.multiply(out, mask_col)
        return out


class OF3DiffusionConditioning(Module):
    """OF3 ``DiffusionConditioning`` (Algorithm 21) on device.

    Inputs (device bf16):
        zij_trunk: [1, N, N, 128]   trunk pair representation
        relpos:    [1, N, N, 139]   reference relpos_complex (captured)
        si_trunk:  [1, N, 384]      trunk single representation
        si_input:  [1, N, 449]      InputEmbedder s_input
        n_emb:     [1, 1, 256]      post-Fourier noise embedding (captured)
        pair_mask: [1, N, N, 1]     pair token mask (token_mask * token_mask)
        tok_mask:  [1, N, 1]        token mask

    Outputs (device bf16):
        si:  [1, N, 384]
        zij: [1, N, N, 128]
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.ln_z = self.torch_to_tt("layer_norm_z.weight")
        self.w_lin_z = self.torch_to_tt("linear_z.weight")
        self.ln_s = self.torch_to_tt("layer_norm_s.weight")
        self.w_lin_s = self.torch_to_tt("linear_s.weight")
        self.ln_n = self.torch_to_tt("layer_norm_n.weight")
        self.w_lin_n = self.torch_to_tt("linear_n.weight")
        self.tr_z = [_SwiGLUTransition(self.scope(f"transition_z.{i}"), compute_kernel_config)
                     for i in range(2)]
        self.tr_s = [_SwiGLUTransition(self.scope(f"transition_s.{i}"), compute_kernel_config)
                     for i in range(2)]

    def __call__(self, zij_trunk, relpos, si_trunk, si_input, n_emb, pair_mask, tok_mask):
        lin = self._lin
        # Pair conditioning: cat([zij_trunk, relpos]) -> LN_z -> linear_z -> 2x SwiGLU.
        zc = ttnn.concat([zij_trunk, relpos], dim=-1)        # [1, N, N, 267]
        z = ttnn.layer_norm(zc, weight=self.ln_z, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(zc)
        z = lin(z, self.w_lin_z)                              # [1, N, N, 128]
        for tr in self.tr_z:
            z = ttnn.add(z, tr(z, pair_mask))

        # Single conditioning: cat([si_trunk, si_input]) -> LN_s -> linear_s.
        sc = ttnn.concat([si_trunk, si_input], dim=-1)       # [1, N, 833]
        s = ttnn.layer_norm(sc, weight=self.ln_s, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(sc)
        s = lin(s, self.w_lin_s)                              # [1, N, 384]
        # Noise embedding: LN_n(n) -> linear_n, broadcast-added over the token dim.
        n_ln = ttnn.layer_norm(n_emb, weight=self.ln_n, epsilon=1e-5,
                               compute_kernel_config=self.compute_kernel_config)
        n_proj = lin(n_ln, self.w_lin_n)                      # [1, 1, 384]
        ttnn.deallocate(n_ln)
        s = ttnn.add(s, n_proj)
        ttnn.deallocate(n_proj)
        for tr in self.tr_s:
            s = ttnn.add(s, tr(s, tok_mask))

        return s, z
