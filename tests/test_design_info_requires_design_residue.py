"""Regression: DesignInfo.is_valid must reject a spec where no residue is
marked for design.

A design chain is only recognised via a residue-count range in its sequence
(e.g. ``sequence: 80..120`` — parsed by schema.py's ``parse_entity`` via
``bool(re.search(r"\\d", sequence))``). A chain written as literal placeholder
characters (e.g. "XXXX...X", a natural mistake — 'X' is standard FASTA
notation for "unknown residue") has no digits, so it is silently parsed as a
normal *fixed* chain of unknown residues instead of a design target.

Before this fix, a spec where every chain does this produced an
all-``False`` ``res_design_mask`` with no validation error. Downstream, the
design-folding stage extracts residues via that mask, retokenizes the (now
empty) structure, and ``np.concatenate([])`` on the empty per-token feature
lists crashes with the opaque "need at least one array to concatenate" —
deep in a fleet-dispatched design shard, after several minutes of wasted
compute across every worker.

This is observed to reproduce in production: three real design jobs
(same public demo) crashed with exactly this error after being pasted with
'X'-placeholder or missing design chains.
"""
from __future__ import annotations

import numpy as np
import pytest

from tt_bio.boltzgen.data.data import DesignInfo


def _design_info(res_design_mask: np.ndarray) -> DesignInfo:
    n = len(res_design_mask)
    return DesignInfo(
        res_design_mask=res_design_mask,
        res_structure_groups=np.zeros(n, dtype=np.int_),
        res_ss_types=np.zeros(n, dtype=np.int_),
        res_binding_type=np.zeros(n, dtype=np.int_),
        res_aa_constraint_mask=np.zeros((n, 20), dtype=np.float32),
    )


def test_all_false_design_mask_is_rejected():
    """The exact bug: no chain used range syntax -> nothing marked for design."""
    info = _design_info(np.zeros(70, dtype=bool))
    with pytest.raises(ValueError, match="No residues are marked for design"):
        DesignInfo.is_valid(info)


def test_some_true_design_mask_is_accepted():
    """A normal spec (target chain + a ranged design chain) must still pass."""
    mask = np.zeros(211, dtype=bool)
    mask[141:] = True  # the 70-residue binder chain is marked for design
    info = _design_info(mask)
    DesignInfo.is_valid(info)  # must not raise
