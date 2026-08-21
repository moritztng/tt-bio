"""RF3 MSA module on ttnn.

Composed from tt-bio's shared `OuterProductMean`, `PairWeightedAveraging`,
`Transition` and `PairformerLayer` rather than reusing `MSALayer`, because RF3
differs from it in two ways that a weight remap cannot see:

- **Weight-shared.** The checkpoint holds one set of weights under
  ``recycler.msa_module`` with no block index at all; upstream constructs its
  sub-modules once and loops ``n_block`` times over them. `MSALayer` is built per
  block, so reusing it would mean four blocks fed from one block's weights.
- **Outer product first.** RF3 updates the pair representation from the previous
  iteration's MSA before pair-weighted averaging reads it. `MSALayer` runs
  pwa -> transition -> outer_product -> pairformer, which is a different
  computation, and the two orders are not rotations of each other because the
  pairformer sits differently relative to the outer product.

Conventions taken from the reference rather than guessed
(`rf3/model/layers/outer_product.py`, `model/RF3_blocks.py`):

- The outer product divides its right projection by the MSA depth before
  contracting and adds ``proj_out``'s bias at full strength afterwards, which is
  tt-bio's default (``scale_bias=False``).
- The inner pairformer uses the same RF3 settings as the trunk one:
  ``scale_pair_bias=False`` (q is pre-scaled, the pair bias added unscaled) and
  ``transpose_bias=False`` (the ending pair bias is built before the transpose).
"""

from __future__ import annotations

import ttnn

from tt_bio.tenstorrent import (
    Module,
    OuterProductMean,
    PairformerLayer,
    PairWeightedAveraging,
    Transition,
    Weights,
)

#: MSA-module channel dims, from configs/model/components/rf3_net.yaml.
C_M = 64
PWA_HEAD_DIM, PWA_N_HEADS = 32, 8
TRI_ATT_HEAD_DIM, TRI_ATT_N_HEADS = 32, 4


class MSASubsampleEmbedder(Module):
    """``emb_msa(msa) + emb_S_inputs(S_inputs)``. Both projections are bias-free."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.msa_weight = self.torch_to_tt("emb_msa.weight")
        self.s_weight = self.torch_to_tt("emb_S_inputs.weight")

    def __call__(self, msa: ttnn.Tensor, s_inputs: ttnn.Tensor) -> ttnn.Tensor:
        m = ttnn.linear(
            msa, self.msa_weight, compute_kernel_config=self.compute_kernel_config
        )
        s = ttnn.linear(
            s_inputs, self.s_weight, compute_kernel_config=self.compute_kernel_config
        )
        # s is per-token [I, c_m]; it broadcasts over the MSA depth axis of m.
        return ttnn.add_(m, s)


class MSAModule(Module):
    """RF3's MSA module: one weight-shared block applied ``n_block`` times."""

    def __init__(
        self,
        n_block: int,
        state_dict: Weights,
        compute_kernel_config,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.n_block = n_block
        self.subsampler = MSASubsampleEmbedder(
            self.scope("msa_subsampler"), compute_kernel_config
        )
        # small_depth reassociates the outer product per MSA row instead of materialising
        # [I, J, C, D]. It declines above OPM_SMALL_DEPTH_MAX = 8 rows, where looping rows
        # would cost more than the materialised path (measured: 3.03 ms/row against 27.43 ms
        # materialised per block at 512 aa, so break-even is ~9 rows). RF3-scoped because the
        # reassociated path is a different reduction order; at <= 8 rows its error against
        # fp64 is 1.014-1.018x the materialised path's own, every rung.
        self.outer_product = OuterProductMean(
            self.scope("outer_product"), compute_kernel_config, small_depth=True
        )
        self.pair_weighted_averaging = PairWeightedAveraging(
            PWA_HEAD_DIM,
            PWA_N_HEADS,
            self.scope("msa_pair_weighted_averaging"),
            compute_kernel_config,
        )
        self.msa_transition = Transition(
            self.scope("msa_transition"), compute_kernel_config
        )
        # transform_s=False: the block updates only the pair representation.
        self.pairformer_layer = PairformerLayer(
            TRI_ATT_HEAD_DIM,
            TRI_ATT_N_HEADS,
            None,
            None,
            False,
            self.scope("pairformer_layer"),
            compute_kernel_config,
            scale_pair_bias=False,
            transpose_bias=False,
            # See template.py: the reference's softmax stays fp32 under autocast.
            fp32_softmax=True,
        )

    def __call__(
        self,
        msa: ttnn.Tensor,
        z: ttnn.Tensor,
        s_inputs: ttnn.Tensor,
        mask: ttnn.Tensor | None = None,
        attn_mask: ttnn.Tensor | None = None,
        msa_mask: ttnn.Tensor | None = None,
        n_msa: int | None = None,
    ) -> ttnn.Tensor:
        m = self.subsampler(msa, s_inputs)
        for _ in range(self.n_block):
            # Outer product FIRST: z carries the previous iteration's MSA before
            # pair-weighted averaging reads it. This ordering is the whole reason
            # this class exists instead of reusing MSALayer.
            z_update = self.outer_product(m, msa_mask, n_msa)
            z = ttnn.add_(z, z_update)
            ttnn.deallocate(z_update)

            m_update = self.pair_weighted_averaging(m, z, attn_mask)
            m = ttnn.add_(m, m_update)
            ttnn.deallocate(m_update)

            m_update = self.msa_transition(m)
            m = ttnn.add_(m, m_update)
            ttnn.deallocate(m_update)

            z = self.pairformer_layer(
                None, z, mask=mask,
                attn_mask_start=attn_mask, attn_mask_end=attn_mask,
            )[1]
        ttnn.deallocate(m)
        return z
