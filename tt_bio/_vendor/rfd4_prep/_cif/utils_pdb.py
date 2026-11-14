# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Protein data type utilities using atomworks.

Note: This module writes CIF files (not PDB) using atomworks.
"""

import os
import re
from collections.abc import Mapping
from typing import Any

import tt_bio._vendor.atomworks.io.transforms.atom_array as ta
import numpy as np
from tt_bio._vendor.atomworks.io.utils.io_utils import to_cif_file
from tt_bio._vendor.atomworks.ml.transforms.encoding import atom_array_from_encoding
from biotite.structure import AtomArray

from .encoding import UNIFIED_ATOM37_ENCODING

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the CIF format.
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
MAX_CHAINS = len(CHAIN_IDS)  # := 62.


def _prepare_bondless_atom_array_for_cif(atom_array: AtomArray) -> AtomArray:
    """Set label_entity_id so atomworks skips bond-based entity annotation.

    atom37 dumps from training have no BondList; atomworks to_cif_file always
    calls _ensure_label_entity_id, which otherwise invokes annotate_entities.
    """
    if "label_entity_id" in atom_array.get_annotation_categories():
        return atom_array
    unique_chains = np.unique(atom_array.chain_id)
    entity_by_chain = {chain_id: i + 1 for i, chain_id in enumerate(unique_chains)}
    atom_array.set_annotation(
        "label_entity_id",
        np.array([entity_by_chain[chain_id] for chain_id in atom_array.chain_id], dtype=np.int8),
    )
    return atom_array


def atom37_to_atom_array(
    coords_atom37: np.ndarray,
    residue_type: np.ndarray,
    chain_index: np.ndarray = None,
) -> AtomArray:
    """Convert atom37 coordinates to AtomArray using atomworks encoding.

    Args:
        coords_atom37: Atom37 coordinates, shape [n_residues, 37, 3]
        residue_type: Residue type indices from UNIFIED_ATOM37_ENCODING, shape [n_residues]
        chain_index: Chain indices for each residue (optional)

    Returns:
        AtomArray with proper annotations
    """
    # Create mask: True where coordinates are valid (non-zero)
    coords_valid = np.any(coords_atom37 != 0, axis=-1)  # [n_res, 37]
    encoded_mask = coords_valid.astype(bool)

    # Determine chain_id from chain_index if provided
    if chain_index is not None:
        unique_chains = np.unique(chain_index)
        if len(unique_chains) == 1:
            chain_id = "A"
        else:
            # For multi-chain, create AtomArray per chain and concatenate
            atom_arrays = []
            for _i, chain_idx in enumerate(unique_chains):
                chain_mask = chain_index == chain_idx
                chain_coords = coords_atom37[chain_mask]
                chain_seq = residue_type[chain_mask]
                chain_mask_valid = encoded_mask[chain_mask]

                chain_id = CHAIN_IDS[int(chain_idx)]
                aa = atom_array_from_encoding(
                    encoded_coord=chain_coords,
                    encoded_mask=chain_mask_valid,
                    encoded_seq=chain_seq,
                    encoding=UNIFIED_ATOM37_ENCODING,
                    chain_id=chain_id,
                )
                atom_arrays.append(aa)

            # Concatenate all chains
            atom_array = atom_arrays[0]
            for aa in atom_arrays[1:]:
                atom_array = atom_array + aa

            # Add annotations
            atom_array.set_annotation("transformation_id", np.array(["1"] * len(atom_array)))
            atom_array = ta.add_chain_iid_annotation(atom_array)

            # Fix dtypes
            atom_array.res_name = atom_array.res_name.astype("U3")
            atom_array.chain_id = atom_array.chain_id.astype("U10")
            atom_array.element = atom_array.element.astype("U2")
            atom_array.atom_name = atom_array.atom_name.astype("U4")

            return atom_array
    else:
        chain_id = "A"

    # Single chain case
    atom_array = atom_array_from_encoding(
        encoded_coord=coords_atom37,
        encoded_mask=encoded_mask,
        encoded_seq=residue_type,
        encoding=UNIFIED_ATOM37_ENCODING,
        chain_id=chain_id,
    )

    # Add transformation_id and chain_iid annotation
    atom_array.set_annotation("transformation_id", np.array(["1"] * len(atom_array)))
    atom_array = ta.add_chain_iid_annotation(atom_array)

    # Fix dtypes
    atom_array.res_name = atom_array.res_name.astype("U3")
    atom_array.chain_id = atom_array.chain_id.astype("U10")
    atom_array.element = atom_array.element.astype("U2")
    atom_array.atom_name = atom_array.atom_name.astype("U4")

    return atom_array


def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: str,
    aatype: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    overwrite: bool = False,
    no_indexing: bool = False,
    b_factors: np.ndarray | None = None,
) -> str:
    """Write protein structure to CIF file using atomworks.

    Note: Despite the function name (kept for backward compatibility),
    this writes CIF files, not PDB files.

    Args:
        prot_pos: Atom37 coordinates, shape [n_res, 37, 3] or [n_models, n_res, 37, 3]
        file_path: Path to output file (will be converted to .cif extension)
        aatype: Residue type indices from UNIFIED_ATOM37_ENCODING
        chain_index: Chain indices for each residue
        overwrite: If False, adds index to filename to avoid overwriting
        no_indexing: If True, use exact file_path without adding index
        b_factors: B-factors (not currently used with atomworks)

    Returns:
        Path to saved CIF file
    """
    # Convert any PDB extension to CIF
    file_path = file_path.replace(".pdb", ".cif")

    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).replace(".cif", "")
        if os.path.exists(file_dir):
            existing_files = [x for x in os.listdir(file_dir) if file_name in x]
            max_existing_idx = max(
                [int(re.findall(r"_(\d+)\.cif", x)[0]) for x in existing_files if re.findall(r"_(\d+)\.cif", x)] + [0]
            )
        else:
            max_existing_idx = 0

    if not no_indexing:
        save_path = file_path.replace(".cif", "") + f"_{max_existing_idx + 1}.cif"
    else:
        save_path = file_path

    if aatype is None:
        aatype = np.zeros(prot_pos.shape[-3], dtype=int)

    # Handle single model case (3D: [n_res, 37, 3])
    if prot_pos.ndim == 3:
        atom_array = atom37_to_atom_array(
            coords_atom37=prot_pos,
            residue_type=aatype,
            chain_index=chain_index,
        )
        atom_array = _prepare_bondless_atom_array_for_cif(atom_array)
        to_cif_file(atom_array, save_path, include_entity_categories=False)
    else:
        raise ValueError(f"Invalid positions shape {prot_pos.shape}")

    return save_path
