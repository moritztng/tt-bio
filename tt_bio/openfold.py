"""OpenFold (AlphaFold2) — Tenstorrent port.

Net-new AF2-specific device blocks live here; the O(L²)/O(L³) pair-track heavy ops
(TriangleMultiplication, TriangleAttention, OuterProductMean) are reused directly from
tt_bio.tenstorrent (PCC-verified — see docs/openfold-port.md). Weight key names follow
the vendored reference (tt_bio/_vendor/openfold), so most blocks need no remap.
"""
from __future__ import annotations

import ttnn

from tt_bio.tenstorrent import Module, get_device, CORE_GRID_MAIN


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
