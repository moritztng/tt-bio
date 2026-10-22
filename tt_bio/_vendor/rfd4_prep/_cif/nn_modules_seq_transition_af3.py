# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
import torch

from .nn_modules_adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
)
from .nn_modules_swiglu import SwiGLU


# Code from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class Transition(torch.nn.Module):
    """Transition layer."""

    def __init__(self, dim: int, expansion_factor: int = 4, layer_norm: bool = False) -> None:
        super().__init__()

        dim_inner = int(dim * expansion_factor)

        self.use_layer_norm = layer_norm
        if self.use_layer_norm:
            self.ln = torch.nn.LayerNorm(dim)

        self.swish_linear = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_inner * 2, bias=False),
            SwiGLU(),
        )
        self.linear_out = torch.nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            mask: binary, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        if self.use_layer_norm:
            x = self.ln(x)
        x = self.linear_out(self.swish_linear(x))
        return x * mask[..., None]


class TransitionADALN(torch.nn.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling aplied to output."""

    def __init__(self, *, dim: int, dim_cond: int, expansion_factor: int = 4) -> None:
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.transition = Transition(dim=dim, expansion_factor=expansion_factor, layer_norm=False)
        self.scale_output = AdaptiveOutputScale(dim=dim, dim_cond=dim_cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        x = self.adaln(x, cond, mask)  # [b, n, dim]
        x = self.transition(x, mask)  # [b, n, dim]
        x = self.scale_output(x, cond, mask)  # [b, n, dim]
        return x * mask[..., None]  # [b, n, dim]
