"""RF3's distogram head on ttnn.

    logits = predictor(Z_II + Z_II.transpose(-2, -3))

The symmetrisation is a SUM, not a mean. Adding the transpose doubles the magnitude of
the symmetric part, and `predictor` is trained against that -- halving it (or applying
the transpose twice) is the double-symmetrize class of bug that has already cost this
repo an ESMFold2 defect. Kept as written.

`predictor` carries a bias, unlike most of RF3's linears.
"""

from __future__ import annotations

import ttnn

from tt_bio.tenstorrent import CORE_GRID_MAIN, Module


class DistogramHead(Module):
    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w = self.torch_to_tt("predictor.weight")
        self.b = self.torch_to_tt("predictor.bias")

    def __call__(self, z: ttnn.Tensor) -> ttnn.Tensor:
        # [1, I, I, c_z]: the reference transposes -2/-3, i.e. the two token axes
        zs = ttnn.add(z, ttnn.permute(z, (0, 2, 1, 3)))
        return ttnn.linear(zs, self.w, bias=self.b,
                           compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)
