# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Atom-level boolean predicates for loss-weight upweighting.

Each predicate takes an :py:class:`AtomArray` and returns a boolean ``np.ndarray`` of
shape ``[n_atoms]``. Predicates compose ADDITIVELY into the per-atom ``loss_weight``
annotation via :py:class:`UpweightLossWeightOnSubset` in the data pipeline:
``w = 1 + Σ αₖ · is_k(atom_array)``.

The Upweight transform spreads each predicate token-wise (``np.max``), so a
predicate that flags even a single atom in a residue will upweight that whole token's loss.

To add a new predicate, define the function here and register it in :py:data:`PREDICATES`.
"""

from collections.abc import Callable, Iterator

import numpy as np
from tt_bio._vendor.atomworks.constants import STANDARD_POLYMER_RESIDUES
from biotite.structure import AtomArray


def is_non_standard(atom_array: AtomArray) -> np.ndarray:
    """True for atoms outside the 20 standard AAs / 4 RNA / 4 DNA nucleotides."""
    return ~np.isin(atom_array.res_name, list(STANDARD_POLYMER_RESIDUES))


def is_unindexed(atom_array: AtomArray) -> np.ndarray:
    """True for atoms in the original ground-truth tokens whose atoms were
    copied to form the in-context (``C_IDX=False``) conditioning slice.
    """
    if "is_unindexed_source" not in atom_array.get_annotation_categories():
        return np.zeros(atom_array.array_length(), dtype=bool)

    return atom_array.get_annotation("is_unindexed_source").astype(bool)


def is_sampled_tip_atom(atom_array: AtomArray) -> np.ndarray:
    """True for atoms in the protein-side conditioning region of a tip-atom
    design task (the sampled tip-atom seed plus its residue-bounded bond-hop
    growth — i.e. what was actually shown to the model as conditioning).
    """
    if "is_sampled_tip_atom" not in atom_array.get_annotation_categories():
        return np.zeros(atom_array.array_length(), dtype=bool)

    return atom_array.get_annotation("is_sampled_tip_atom").astype(bool)


PREDICATES: dict[str, Callable[[AtomArray], np.ndarray]] = {
    "non_standard": is_non_standard,
    "unindexed": is_unindexed,
    "sampled_tip_atom": is_sampled_tip_atom,
}


def build_loss_weight_upweights(
    cfg: dict | None,
) -> Iterator[tuple[Callable[[AtomArray], np.ndarray], float]]:
    """Yield ``(predicate, factor)`` tuples from a config dict.

    Args:
      cfg: mapping like ``{"non_standard": 1.0, ...}``. Unknown keys raise
        :py:class:`KeyError`. Entries with ``factor == 0.0`` are skipped (no-ops).

    Yields:
      One tuple per active upweight, suitable for instantiating
      :py:class:`UpweightLossWeightOnSubset`.
    """
    if not cfg:
        return
    for name, factor in cfg.items():
        if name not in PREDICATES:
            raise KeyError(f"Unknown loss_weight predicate {name!r}; available: {sorted(PREDICATES.keys())}")
        if float(factor) != 0.0:
            yield PREDICATES[name], float(factor)
