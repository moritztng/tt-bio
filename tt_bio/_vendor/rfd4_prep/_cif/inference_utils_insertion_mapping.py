# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Read the insertion head's predicted token mapping via per-chain Hungarian assignment."""

import logging
import numpy as np
import torch
from tt_bio._vendor.atomworks.ml.conditions import C_IDX
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger(__name__)


def build_predicted_insertion_mapping(
    insertion_logits: torch.Tensor,
    full_array_with_unindexed: AtomArray,
) -> dict[int, int]:
    """Match each unindexed token to a distinct generated token on the same chain.

    Args:
      insertion_logits: ``[n_valid, n_valid]`` logits, ``n_valid = n_gen + n_unindexed``.
      full_array_with_unindexed: Atom array of generated + unindexed in-context tokens.

    Returns:
      ``{unindexed_token_id: target_generated_token_id}``. Empty when there are no
      unindexed tokens or the logits shape is inconsistent with the token counts.
    """
    all_tokens = full_array_with_unindexed[get_token_starts(full_array_with_unindexed)]
    is_indexed_token = C_IDX.mask(all_tokens, default="generate")
    generated_tokens = all_tokens[is_indexed_token]
    unindexed_tokens = all_tokens[~is_indexed_token]
    n_gen, n_unindexed = len(generated_tokens), len(unindexed_tokens)
    if n_unindexed == 0:
        return {}

    if n_gen + n_unindexed != insertion_logits.shape[0]:
        logger.warning(
            f"Insertion logits shape {tuple(insertion_logits.shape)} inconsistent with token "
            f"counts (gen={n_gen}, unindexed={n_unindexed}); no insertion mapping produced."
        )
        return {}

    logits = insertion_logits.float().cpu().numpy()  # [n_valid, n_valid]
    gen_chain = np.asarray(generated_tokens.chain_id)  # [n_gen]
    unindexed_chain = np.asarray(unindexed_tokens.chain_id)  # [n_unindexed]
    gen_token_id = np.asarray(generated_tokens.token_id)  # [n_gen]
    unindexed_token_id = np.asarray(unindexed_tokens.token_id)  # [n_unindexed]

    # Per-chain Hungarian assignment: each unindexed token -> a DISTINCT generated token on the same chain
    mapping: dict[int, int] = {}
    for chain in np.unique(unindexed_chain):
        u_idx = np.flatnonzero(unindexed_chain == chain)  # rows, in [0, n_unindexed)
        g_idx = np.flatnonzero(gen_chain == chain)  # cols, in [0, n_gen)

        if len(g_idx) == 0:
            continue

        # Solve linear assignment problem - maximize the sum of logits for the assigned pairs
        score = logits[np.ix_(n_gen + u_idx, g_idx)]  # [n_u_chain, n_g_chain]
        rows, cols = linear_sum_assignment(score, maximize=True)
        for r, c in zip(rows, cols, strict=True):
            mapping[int(unindexed_token_id[u_idx[r]])] = int(gen_token_id[g_idx[c]])

    return mapping
