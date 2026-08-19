"""RF3's diffusion atom attention decoder on ttnn.

    Ql = linear_1(Ai[..., tok_idx, :]) + Ql_skip
    Ql = atom_transformer(Ql, Cl_skip, Plm_skip)
    Rl_update = to_r_update(Ql)                      # LayerNorm + linear(c_atom, 3)

Same 3-block atom transformer and the same three flags as every other RF3 atom stack,
so the windowed bias is the shared `windowed_bias` helper.

The reference gathers A_I to atoms and then applies linear_1. This applies linear_1 on
the token track first and gathers after: the linear is per-position so the two agree,
and doing it token-side is [I, 768] -> [I, 128] instead of [L, 768] -> [L, 128], which
for a real target is the difference between one matmul per token and one per atom.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.rf3.atom_encoder import ATOM_WINDOW, C_ATOM, N_HEADS, windowed_bias
from tt_bio.rf3.remap_encoder import (atom_transformer_bias_weights,
                                      remap_atom_transformer)
from tt_bio.tenstorrent import (CORE_GRID_MAIN, DiffusionTransformer, Module,
                                _dtype)


class DiffusionAtomDecoder(Module):
    def __init__(self, state_dict, compute_kernel_config, n_block: int = 3):
        super().__init__(state_dict, compute_kernel_config)
        raw = self.weights.as_dict()
        self.n_block = n_block
        self.linear_1 = self.torch_to_tt("linear_1.weight")
        self.r_norm_w = self.torch_to_tt("to_r_update.0.weight")
        self.r_norm_b = self.torch_to_tt("to_r_update.0.bias")
        self.r_w = self.torch_to_tt("to_r_update.1.weight")

        self.ln0_w, self.ln0_b, self.to_b = [], [], []
        for i in range(n_block):
            bw = atom_transformer_bias_weights(raw, i)
            self.ln0_w.append(ttnn.from_torch(
                bw["ln_0.weight"].float(), layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))
            self.ln0_b.append(ttnn.from_torch(
                bw["ln_0.bias"].float(), layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))
            self.to_b.append(ttnn.from_torch(
                bw["to_b.weight"].t().contiguous(), layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))

        self.transformer = DiffusionTransformer(
            n_layers=n_block, dim=C_ATOM, n_heads=N_HEADS, atom_level=True,
            state_dict=remap_atom_transformer(raw, n_block),
            compute_kernel_config=compute_kernel_config,
            no_residual=True, a_to_b_gate=False, fp32_softmax=True,
        )

    def __call__(self, a_i, q_skip, c_skip, p_skip, a2t_onehot, keys_indexing,
                 mask, n_pad):
        q = ttnn.linear(a_i, self.linear_1,
                        compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)
        q = ttnn.matmul(a2t_onehot, q,
                        compute_kernel_config=self.compute_kernel_config)
        q = ttnn.add(q, q_skip)

        biases = [windowed_bias(p_skip, self.ln0_w[i], self.ln0_b[i], self.to_b[i],
                                mask, n_pad, self.compute_kernel_config, self.device)
                  for i in range(self.n_block)]
        k = n_pad // ATOM_WINDOW
        out = self.transformer(ttnn.reshape(q, (1, k, ATOM_WINDOW, C_ATOM)),
                               ttnn.reshape(c_skip, (1, k, ATOM_WINDOW, C_ATOM)),
                               biases, keys_indexing)
        out = ttnn.reshape(out, (1, n_pad, C_ATOM))
        out = ttnn.layer_norm(out, weight=self.r_norm_w, bias=self.r_norm_b,
                              epsilon=1e-5,
                              compute_kernel_config=self.compute_kernel_config)
        return ttnn.linear(out, self.r_w,
                           compute_kernel_config=self.compute_kernel_config)
