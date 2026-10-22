# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Shared selection-string-to-mask evaluation.

This is the *only* way to turn a string into an atom mask in this codebase: a
"selection" is just one query string, handed directly to the atomworks query
engine (:py:class:`~atomworks.io.utils.query.QueryExpression`) -- no separate
structured-dict combinator syntax. Originally written for the condition-spec
format (``docs/condition_spec_yaml.md``); also backs the metric region/atom/
alignment specs in :py:mod:`~rfproteina.metrics.atom_pair_distance_metric`,
:py:mod:`~rfproteina.metrics.physics.tmol_ddg_core`,
:py:mod:`~rfproteina.metrics.oracles.af3_oracle`, and
:py:mod:`~rfproteina.metrics.gt_structure_comparison`, which used to each
re-derive their own "ensure referenced annotations, then evaluate" logic (or,
for the TMol/atom-pair metrics, a separate dict-based field-filter combinator).
Supersedes the selection-combinator design in ``design_mask_spec_refactor.md``
/ ``docs/mask_spec.md``.
"""

import ast

import numpy as np
from tt_bio._vendor.atomworks.io.utils.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from tt_bio._vendor.atomworks.io.utils.query import QueryExpression
from tt_bio._vendor.atomworks.io.utils.selection import get_annotation_categories
from biotite.structure import AtomArray, AtomArrayStack

from . import utils_chain  # noqa: F401 (register lazy chain annotations)


def resolve_mask(atom_array: AtomArray | AtomArrayStack, query: str) -> np.ndarray:
    """Evaluate a selection query string against ``atom_array``.

    This is the one shared implementation of "ensure any annotations a query
    references, then evaluate it into a boolean mask" -- every selection
    string anywhere in the codebase (condition specs, metric region/alignment
    specs, ...) should call this rather than re-deriving annotations by hand.

    An annotation the query references but that isn't already present is derived
    first -- but *only* if it has a real generator in
    :py:data:`~atomworks.io.utils.annotator.ANNOTATOR_REGISTRY`. Everything else
    is left for :py:class:`~atomworks.io.utils.query.QueryExpression` itself:
    query-function names like ``has_bonds`` resolve there, and a genuinely absent
    annotation raises ``NameError``.

    Args:
      atom_array: Structure (or stack) to evaluate the query against.
      query: Boolean-expression query string, e.g. ``"chain_id == 'A' and
        res_id >= 10"``.

    Returns:
      Boolean mask, one entry per atom.

    Raises:
      NameError: If the query references an annotation that is neither present on
        ``atom_array``, derivable, nor a query-engine builtin.
    """
    tree = ast.parse(query, mode="eval")
    referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    present = set(get_annotation_categories(atom_array, n_body="all"))
    for name in referenced_names - present:
        if name in ANNOTATOR_REGISTRY:
            ensure_annotations(atom_array, name)
    return QueryExpression(query).mask(atom_array)
