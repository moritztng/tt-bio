# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
import torch

from .nn_feature_factory import FeatureFactory
from .nn_modules_adaptive_ln_scale import AdaptiveLayerNorm


class PairReprBuilder(torch.nn.Module):
    """Build pair features, broadcasting per-example conditioning from [b, dim_cond]."""

    def __init__(
        self,
        feats_repr: list,
        feats_cond: list | None,
        dim_feats_out: int,
        dim_cond_pair: int,
        **kwargs: object,
    ) -> None:
        super().__init__()

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None
        if feats_cond is not None and len(feats_cond) > 0:
            self.cond_factory = FeatureFactory(
                feats=feats_cond,
                dim_feats_out=dim_cond_pair,
                use_ln_out=True,
                mode="pair",
                compact=True,
                **kwargs,
            )
            self.adaln = AdaptiveLayerNorm(dim=dim_feats_out, dim_cond=dim_cond_pair, preserve_dtype=True)

    def forward(self, batch_nn: dict) -> torch.Tensor:
        mask = batch_nn["padding_mask"]  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn)  # [b, dim_cond] for per-example conditioning
            repr = self.adaln(repr, cond, pair_mask)
        return repr
