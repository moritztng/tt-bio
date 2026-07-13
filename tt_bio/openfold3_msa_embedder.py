"""OF3 MSAModuleEmbedder device port (P7).

OF3 ``MSAModuleEmbedder`` (AF3 Algorithm 8 lines 1-4): subsample the MSA, then

    m = linear_m(msa_feat) + linear_s_input(s_input).unsqueeze(-3)

Two bias-free linears (``linear_m`` 34->c_m=64 over ``cat([msa, has_deletion,
deletion_value])``; ``linear_s_input`` c_s_input=449->c_m=64) and a broadcast add over
the MSA-sequence dim. The MSA subsampling (stochastic, AF3 SI 2.2) is host-side and
captured in the golden (``scripts/of3_msa_embedder_golden.py`` records the post-subsample
``msa_feat`` via a ``linear_m`` input hook), so this module is PCC-gated against the exact
reference subsample -- isolating the device linear precision from the subsample logic, the
same discipline as the other OF3 golden legs.

This extends the trunk validation past the InputEmbedder: ``s_input -> m`` here, complementing
the already-gated MSA stack (``m, z -> z`` in tests/test_openfold3_msa.py).
"""
from __future__ import annotations

import ttnn

from .tenstorrent import Module


class MSAModuleEmbedder(Module):
    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w_m = self.torch_to_tt("linear_m.weight")
        self.w_s = self.torch_to_tt("linear_s_input.weight")

    def __call__(self, msa_feat, s_input):
        """msa_feat: [1, N_seq, N_token, 34] device; s_input: [1, N_token, 449] device.
        Returns m: [1, N_seq, N_token, c_m=64]."""
        lin = self._lin
        m = lin(msa_feat, self.w_m)
        s = lin(s_input, self.w_s)            # [1, N_token, 64]
        s = ttnn.unsqueeze(s, -3)             # [1, 1, N_token, 64] -- broadcast over N_seq
        m = ttnn.add(m, s)
        ttnn.deallocate(s)
        return m
