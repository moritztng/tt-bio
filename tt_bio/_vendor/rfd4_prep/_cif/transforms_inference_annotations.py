# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Prepare source provenance annotations for inference inputs."""

import biotite.structure as struc
import numpy as np
from tt_bio._vendor.atomworks.io.utils.standard_annotations import S_SEGMAX, S_SEGMIN
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray

from .utils_standard_annotations import SA_SRC


def build_src_component_annotation(atom_array: AtomArray) -> np.ndarray:
    """Compute per-atom :py:data:`~rfproteina.utils.standard_annotations.SA_SRC` labels.

    Each non-sentinel residue is labeled with its chain and residue ID (e.g. ``"A3"``).
    Expandable-segment sentinels are labeled ``"{seg_min}-{seg_max}"``.

    Returns:
      String array of shape ``(n_atoms,)`` with dtype matching ``SA_SRC``.
    """
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    starts, stops = res_starts[:-1], res_starts[1:]

    is_expandable_segment_atom = S_SEGMIN.mask(atom_array)

    chain_id = atom_array.chain_id[starts]
    res_id = atom_array.res_id[starts].astype(int)
    is_expandable_segment = is_expandable_segment_atom[starts]
    seg_min = S_SEGMIN.annotation(atom_array)[starts].astype(int)
    seg_max = S_SEGMAX.annotation(atom_array)[starts].astype(int)

    src = np.empty(atom_array.array_length(), dtype=SA_SRC.dtype)

    for i, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if is_expandable_segment[i]:
            src[start:stop] = f"{seg_min[i]}-{seg_max[i]}"
        else:
            src[start:stop] = f"{chain_id[i]}{res_id[i]}"

    return src


class AddSrcComponentAnnotation(Transform):
    """Annotate :py:data:`~rfproteina.utils.standard_annotations.SA_SRC` source labels.

    Must run **before** :py:class:`~atomworks.ml.transforms.realize_expandable_segments.RealizeExpandableSegments`
    so that original ``res_id`` values are preserved on non-sentinel residues.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        SA_SRC.set_annotation(atom_array, build_src_component_annotation(atom_array))
        data["atom_array"] = atom_array
        return data
