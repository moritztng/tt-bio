# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Utilities for identifying the chain of interest in a multi-chain structure."""

import logging
import numpy as np
from tt_bio._vendor.atomworks.io.utils.annotator import _register_lazy_annotator, ensure_annotations
from tt_bio._vendor.atomworks.ml.conditions import C_CRD, C_SEQ
from biotite.structure import AtomArray, AtomArrayStack

from .conditions import C_CRDGRP

logger = logging.getLogger(__name__)


@_register_lazy_annotator("is_chain_of_interest")
def annotate_chain_of_interest(atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
    """Select the unambiguous designed chain, using the first frame for stacks.
    Remove this cached lazy annotation after changing chain identities or conditioning.
    """
    if atom_array.array_length() == 0 or (isinstance(atom_array, AtomArrayStack) and atom_array.stack_depth() == 0):
        raise ValueError("Cannot determine chain of interest from an empty structure.")
    frame = atom_array[0] if isinstance(atom_array, AtomArrayStack) else atom_array
    chain, unambiguous = get_chain_of_interest(frame)
    if chain is None or not unambiguous:
        raise ValueError("Cannot unambiguously determine chain of interest; use an explicit chain query.")
    chain_attr = "chain_iid" if "chain_iid" in frame.get_annotation_categories() else "chain_id"
    return frame.get_annotation(chain_attr) == chain


def get_chain_of_interest(atom_array: AtomArray) -> tuple[str | None, bool]:
    """Get the chain ID or IID of the chain of interest in the AtomArray.

    The chain of interest corresponds to the primary designed chain, and is determined by the following heuristic:
    1. If there is only one unique chain, return that chain.
    2. Subset to only consider protein chains. If none remain, return the first
       chain in sorted order with a warning.
    3. Subset to chains with the LEAST sequence conditioning relative to chain length
       (the designed chain has its sequence generated, so it is least sequence-conditioned).
    4. Tie-break: subset to chains with the LEAST coordinate conditioning relative to chain
       length (absolute or internal).
    5. If this is still ambiguous, return the first chain in sorted order and give a warning.

    TODO: This heuristic will need to be adjusted for more complex use-cases.

    Args:
        atom_array: Input AtomArray.

    Returns:
        Tuple of (chain_id_or_iid, is_unambiguous). ``chain_id_or_iid`` is
        ``None`` when the input contains no protein chains.
    """
    if "chain_iid" in atom_array.get_annotation_categories():
        chain_identifiers = atom_array.get_annotation("chain_iid")
    else:
        chain_identifiers = atom_array.get_annotation("chain_id")
    unique_chain_identifiers = np.unique(chain_identifiers)

    if len(unique_chain_identifiers) < 2:
        return unique_chain_identifiers[0], True

    # Prefer protein chains when present, but fall back to all chains for
    # all-RNA/all-DNA/all-nucleic multichain cases.
    ensure_annotations(atom_array, "is_protein")
    protein_chain_identifiers = [
        cid for cid in unique_chain_identifiers if atom_array[chain_identifiers == cid].is_protein.all()
    ]
    if len(protein_chain_identifiers) == 1:
        return protein_chain_identifiers[0], True

    if len(protein_chain_identifiers) == 0:
        logger.warning("No pure-protein chains found; defaulting to first chain in sorted order.")
        return sorted(unique_chain_identifiers)[0], False

    chain_subset_by_id = {cid: atom_array[chain_identifiers == cid] for cid in protein_chain_identifiers}

    # Subset by sequence conditioning relative to chain length. The designed chain has its
    # sequence generated rather than given, so it is the least sequence-conditioned chain.
    sequence_conditioned_fraction_by_chain = {
        cid: C_SEQ.mask(chain_subset_by_id[cid]).mean() for cid in protein_chain_identifiers
    }
    least_sequence_conditioned_fraction = min(sequence_conditioned_fraction_by_chain.values())
    least_sequence_conditioned_chain_identifiers = [
        cid
        for cid, frac in sequence_conditioned_fraction_by_chain.items()
        if frac == least_sequence_conditioned_fraction
    ]
    if len(least_sequence_conditioned_chain_identifiers) == 1:
        return least_sequence_conditioned_chain_identifiers[0], True

    # Tie-break by coordinate conditioning relative to chain length (absolute or internal).
    coord_conditioned_fraction_by_chain = {
        cid: C_CRD.mask(chain_subset_by_id[cid]).mean() + C_CRDGRP.is_internal(chain_subset_by_id[cid]).mean()
        for cid in least_sequence_conditioned_chain_identifiers
    }
    least_coord_conditioned_fraction = min(coord_conditioned_fraction_by_chain.values())
    least_coord_conditioned_chain_identifiers = [
        cid for cid, frac in coord_conditioned_fraction_by_chain.items() if frac == least_coord_conditioned_fraction
    ]
    if len(least_coord_conditioned_chain_identifiers) == 1:
        return least_coord_conditioned_chain_identifiers[0], True

    # Ambiguous - return first chain with warning
    logger.warning("Could not unambiguously determine chain of interest; defaulting to first chain in sorted order.")
    return sorted(least_coord_conditioned_chain_identifiers)[0], False
