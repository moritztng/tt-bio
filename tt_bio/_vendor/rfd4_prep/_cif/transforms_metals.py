# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Transforms for metal atoms and their coordination bonds."""

import numpy as np
from tt_bio._vendor.atomworks.ml.transforms.base import Transform


class DropoutCoordinationBonds(Transform):
    """Drop out coordination bonds (bond_order=4) with two-level stochastic masking.

    Level 1: With probability ``p_all``, hide ALL coordination bonds in the example.
    Level 2: If kept, each individual coordination bond is independently dropped
    with probability ``p_individual``.

    Modifies ``bond_mask``, ``bond_order``, and ``is_aromatic`` in-place so all
    downstream features stay consistent. Only added to pipeline during training.

    Reads / writes:
      ``data["feats"]["bond_order"]`` (``int16 [n_tokens, n_tokens]``),
      ``data["feats"]["bond_mask"]`` (``bool  [n_tokens, n_tokens]``),
      ``data["feats"]["is_aromatic"]`` (``bool  [n_tokens, n_tokens]``).
      All three matrices must be symmetric; the transform preserves that.
    """

    COORDINATION_ORDER = 4

    def __init__(self, p_all: float = 0.5, p_individual: float = 0.1):
        self.p_all = p_all
        self.p_individual = p_individual

    def forward(self, data: dict) -> dict:
        bond_order = data["feats"]["bond_order"]  # [n_tokens, n_tokens] int16
        coord_mask = bond_order == self.COORDINATION_ORDER  # [n_tokens, n_tokens] bool

        if not coord_mask.any():
            return data

        # Level 1: drop ALL coordination bonds with probability p_all
        if np.random.rand() < self.p_all:
            drop = coord_mask
        else:
            # Level 2: drop individual coordination bonds with probability p_individual
            drop = coord_mask & (np.random.rand(*coord_mask.shape) < self.p_individual)
            # Symmetrize: if (i,j) is dropped, (j,i) must also be dropped
            drop = drop | drop.T

        data["feats"]["bond_order"][drop] = 0
        data["feats"]["bond_mask"][drop] = False
        data["feats"]["is_aromatic"][drop] = False  # coordination bonds shouldn't be aromatic, but stay safe

        return data
