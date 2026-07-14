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

from .tenstorrent import (
    Module, OuterProductMean, PairWeightedAveraging, Transition, PairformerLayer,
)
from .openfold3_weights import remap_msa_module


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


# OF3 msa_module dims (config.model_config): c_hidden_msa_att=8, no_heads_msa=8;
# c_hidden_pair_att=32, no_heads_pair=4. The pair_stack runs pair-only (transform_s=False).
_MSA_AVG_DIMS = (8, 8)
_MSA_TRI_DIMS = (32, 4)


class MSAModuleBlock:
    """One OF3 ``MSAModuleBlock`` (AF3 Algorithm 10), ``opm_first=True`` ordering
    (OuterProductMean runs BEFORE the msa update -- the reverse of
    ``tt_bio.tenstorrent.MSALayer``'s Boltz-2 ``opm_first=False`` order, so the block
    is composed from the raw primitives directly, not via ``MSALayer``).

    m, z -> m, z per block:

        z = z + outer_product_mean(m)
        if not skip_msa_update:                  # all blocks except the last
            m = m + pair_weighted_averaging(m, z)
            m = m + msa_transition(m)
        z = pair_stack(z)                        # tri_mul + tri_att + pair_transition

    The last block (``skip_msa_update=True``) has no PWA/transition; ``has_msa_update``
    is inferred from which keys ``remap_msa_block`` returned. Composes the same
    primitives, in the same order, as ``tests/test_openfold3_msa.py::_run_block`` -- the
    single source of truth for the OF3 block ordering.
    """

    def __init__(self, block_remap, compute_kernel_config):
        ckc = compute_kernel_config
        self.opm = OuterProductMean(block_remap["outer_product_mean"], ckc)
        self.has_msa_update = "pair_weighted_averaging" in block_remap
        if self.has_msa_update:
            self.pwa = PairWeightedAveraging(
                *_MSA_AVG_DIMS, block_remap["pair_weighted_averaging"], ckc)
            self.msa_transition = Transition(block_remap["msa_transition"], ckc)
        self.pair_stack = PairformerLayer(
            *_MSA_TRI_DIMS, None, None, False, block_remap["pair_stack"], ckc)

    def __call__(self, m, z):
        z = ttnn.add(z, self.opm(m, None, None))
        if self.has_msa_update:
            m = ttnn.add(m, ttnn.reshape(self.pwa(m, ttnn.clone(z)), tuple(m.shape)))
            m = ttnn.add(m, ttnn.reshape(self.msa_transition(m), tuple(m.shape)))
        z = self.pair_stack(None, z)[1]
        return m, z


class MSAModule:
    """OF3 ``MSAModuleStack`` (4-block, ``opm_first=True``) device port. ``m, z -> z``
    (the reference discards ``m`` after the last block; the device path returns both so
    the trunk can reuse the constant ``m`` across cycles without re-running the
    embedder).

    Device-precision note (see docs/openfold3-port.md P8 tick 17 / tests/test_openfold3_msa.py,
    both xfail): the z-track is an intrinsic bf16 ill-conditioning limit at OF3's
    activation magnitude (z_pcc ~0.75 over 4 blocks), NOT a kernel bug and NOT the
    softmax lever; the fp32-z-path fix is release-gated. The trunk accepts this
    degradation as a known, quantified real number and measures its propagation into
    s_trunk/z_trunk rather than chasing it further.
    """

    def __init__(self, state_dict, compute_kernel_config):
        self.blocks = [
            MSAModuleBlock(b, compute_kernel_config)
            for b in remap_msa_module(state_dict, prefix="msa_module")
        ]

    def __call__(self, m, z):
        for block in self.blocks:
            m, z = block(m, z)
        return m, z
