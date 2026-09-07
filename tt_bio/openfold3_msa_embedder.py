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
    accurate_softmax_site, pwa_single_shot_bytes,
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

    def __init__(self, block_remap, compute_kernel_config, transpose_bias: bool = True):
        ckc = compute_kernel_config
        self.opm = OuterProductMean(block_remap["outer_product_mean"], ckc,
                                  scale_bias=True)
        self.has_msa_update = "pair_weighted_averaging" in block_remap
        if self.has_msa_update:
            self.pwa = PairWeightedAveraging(
                *_MSA_AVG_DIMS, block_remap["pair_weighted_averaging"], ckc)
            self.msa_transition = Transition(block_remap["msa_transition"], ckc)
        # scale_pair_bias=False: openfold3 adds the tri_att pair bias UNSCALED (q is
        # pre-scaled by 1/sqrt(d)); the shared primitive's default sqrt(d) fold is the
        # Boltz convention and was root-caused as the OF3 MSA z-track degradation.
        self.pair_stack = PairformerLayer(
            *_MSA_TRI_DIMS, None, None, False, block_remap["pair_stack"], ckc,
            scale_pair_bias=False, fp32_softmax=True, transpose_bias=transpose_bias,
            accurate_softmax=accurate_softmax_site("openfold3.msa"))

    def __call__(self, m, z, pair_mask=None, attn_mask=None, own_m=False):
        # OuterProductMean is deliberately left unmasked: it reduces over MSA DEPTH, so a padded
        # token can only reach a padded pair through it. PairWeightedAveraging is not -- its
        # softmax runs over the token axis, so a padded key would take real weight without the
        # additive -1e9 (protenix.py Trunk.update_msa passes the same tensor for the same reason).
        # Each residual writes its sum into the UPDATE's buffer, not into a third one. `m` is
        # [depth, tokens, c_m] -- 1 860 042 752 B at 1024 tokens x 14191 MSA rows -- and
        # `ttnn.add(m, upd)` holds the old `m`, the update and the result live at once, so every
        # residual here carried one whole redundant copy of the MSA representation. The operands
        # and their order are unchanged and elementwise addition is commutative, so the sum is the
        # same bits; only which buffer receives it moves. `m` itself is never written in place:
        # the trunk re-feeds the embedder's `m` on every recycle, so mutating it would corrupt the
        # next cycle. Same lever as PairWeightedAveraging's in-place head accumulate.
        upd = self.opm(m, None, None)
        z = ttnn.add_(upd, z)
        if self.has_msa_update and isinstance(m, list):
            # `m` is a list of depth chunks and stays one: both residuals are per depth row, so
            # each chunk's new value depends only on that chunk's old value and on `z`. Nothing
            # here ever holds the representation contiguously, which is the point -- at 1024
            # tokens x 14191 rows the contiguous form is 1.86 GB and the allocator had 3.79 GB
            # free with 1.256 GB as its largest run.
            #
            # `own_m` says whether this block may free the chunks it was handed. The FIRST block
            # may not: the trunk re-feeds the embedder's `m` on every recycle, so its chunks have
            # to survive the whole trunk. Every later block consumes its predecessor's output and
            # frees each chunk as soon as its replacement exists, so the peak is the trunk's list
            # plus one block's list plus ~2 chunks, never a second contiguous copy.
            pwa_out = self.pwa(m, ttnn.clone(z), attn_mask)
            out = []
            for mc, pc in zip(m, pwa_out):
                t1 = ttnn.add_(pc, mc)
                if own_m:
                    ttnn.deallocate(mc)
                t2 = ttnn.reshape(self.msa_transition(t1), tuple(t1.shape))
                out.append(ttnn.add_(t2, t1))
                ttnn.deallocate(t1)
            m = out
        elif self.has_msa_update:
            upd = ttnn.reshape(self.pwa(m, ttnn.clone(z), attn_mask), tuple(m.shape))
            m = ttnn.add_(upd, m)
            # Each residual leaves the previous `m` buffer free somewhere in the middle of the
            # heap, and the next one needs its whole width contiguous: at 960 tokens x 14191 rows
            # the transition's 1 743 790 080 B was refused with 4.4 GB free and 1.35 GB as the
            # largest run. Compacting between the two residuals coalesces that hole, the same call
            # OuterProductMean already makes before its own matmuls. Pure data movement, and only
            # where the residual is wide enough to be at risk -- below the budget this is the
            # single-shot path's untouched sequence of allocations.
            if m.logical_volume() * 2 > pwa_single_shot_bytes():
                m = ttnn.reallocate(m)
            upd = ttnn.reshape(self.msa_transition(m), tuple(m.shape))
            m = ttnn.add_(upd, m)
        z = self.pair_stack(None, z, pair_mask, attn_mask, attn_mask)[1]
        return m, z


class MSAModule:
    """OF3 ``MSAModuleStack`` (4-block, ``opm_first=True``) device port. ``m, z -> z``
    (the reference discards ``m`` after the last block; the device path returns both so
    the trunk can reuse the constant ``m`` across cycles without re-running the
    embedder).

    Device-precision note (see docs/openfold3-port.md, "Precision"): the residual
    z-track gap is an intrinsic bf16 ill-conditioning limit at OF3's
    activation magnitude, NOT a kernel bug and NOT the
    softmax lever; the fp32-z-path fix is release-gated. The trunk accepts this
    degradation as a known, quantified real number and measures its propagation into
    s_trunk/z_trunk rather than chasing it further.
    """

    def __init__(self, state_dict, compute_kernel_config, transpose_bias: bool = True):
        self.blocks = [
            MSAModuleBlock(b, compute_kernel_config, transpose_bias=transpose_bias)
            for b in remap_msa_module(state_dict, prefix="msa_module")
        ]

    def __call__(self, m, z, pair_mask=None, attn_mask=None):
        # Only a block that updates `m` produces a new one, and only then does the next block own
        # what it is handed. The trunk's own `m` is never freed here: it is re-fed every recycle.
        own = False
        for block in self.blocks:
            m, z = block(m, z, pair_mask, attn_mask, own_m=own)
            own = own or block.has_msa_update
        return m, z
