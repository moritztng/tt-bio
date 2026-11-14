# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
import torch
from torch.nn import functional as F


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class SwiGLU(torch.nn.Module):
    """SwiGLU layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor, shape [..., d]

        Returns:
            Tensor of shape [..., d//2].
        """
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x
