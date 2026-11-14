# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Schema for Proteina with explicit specs: (dtype, default, dims)."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
from tt_bio._vendor.atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from biotite.structure import AtomArray
from torch.nn import functional as F

from .conditions import (
    C_CATH,
    C_CENTROID,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_HPHOB,
    C_IDXGRP,
    C_INTCRD,
    C_MCOORD,
    C_PICATION,
    C_PISTACK,
    C_RAS,
    C_SALT,
    C_SEC,
    SEQUENCE_PROBABILITY_CONDITIONS,
    HBondEnum,
    MetalCoordEnum,
    PiCationEnum,
    RasaEnum,
    SaltBridgeEnum,
    SSEnum,
)
from .datapipes_schema_base import DataSample
from .encoding import TOKEN_TYPE_IDS, UNIFIED_ATOM37_ENCODING
from .transforms_chirality_feature import MAX_OVERLAPPING_CENTERS


class TokenType:
    """Token type constants, as carried by the `token_type` annotation and feature."""

    PROTEIN = TOKEN_TYPE_IDS["aa"]
    RNA = TOKEN_TYPE_IDS["rna"]
    DNA = TOKEN_TYPE_IDS["dna"]
    ATOMIZED = TOKEN_TYPE_IDS["atomized"]
    UNKNOWN = TOKEN_TYPE_IDS["unknown"]
    GLYCAN = TOKEN_TYPE_IDS["glycan"]
    NCAA = TOKEN_TYPE_IDS["ncaa"]


# Tensor row count for direct ID indexing: gaps leave unused rows; aliases share a row.
NUM_TOKEN_TYPES = max(TOKEN_TYPE_IDS.values()) + 1


# Specs: name -> (dtype, default_value, dim_string)
# dim_string uses: I=tokens, numbers=fixed dims
SPECS: dict[str, tuple] = {
    "centroid_of_atoms_without_coordinate_conditioning": (torch.float32, 0.0, "3"),
    # Token-level [I]
    "residue_type": (torch.int64, -1, "I"),
    C_SEQ.full_name: (torch.int64, UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"], "I"),
    "token_type": (torch.int32, TokenType.UNKNOWN, "I"),
    "tok_idx": (torch.int16, -1, "I"),
    "chain_iid": (torch.int16, -1, "I"),
    "molecule_iid": (torch.int16, -1, "I"),
    "molecule_entity_id": (torch.int16, -1, "I"),
    "supports_local_latent": (torch.bool, False, "I"),
    "is_in_context_conditioning": (torch.bool, False, "I"),
    "is_resolved_mask": (torch.bool, False, "I"),
    "insertion_source_idx": (torch.int32, -1, "I"),
    "loss_weight": (torch.float32, 1.0, "I"),
    "coord_cond_mask": (torch.bool, False, "I"),
    C_SEC.full_name: (torch.int16, SSEnum.NONE.value, "I"),
    C_IDXGRP.full_name: (torch.int16, -1, "I"),
    "na_ss_state": (torch.int16, 0, "I"),
    "na_ss_state_mask": (torch.bool, False, "I"),
    # Token-level with extra dims [I, ...]
    C_CATH.full_name: (torch.int32, -1, "I 3"),
    C_CENTROID.full_name: (torch.float32, 0.0, "I 3"),
    "chirality": (torch.float32, 0.0, f"I {MAX_OVERLAPPING_CENTERS * 3}"),
    "chiral_frame_atoms": (torch.int64, -1, f"I {MAX_OVERLAPPING_CENTERS} 8"),
    "chiral_slot_owner": (torch.int64, -1, f"I {MAX_OVERLAPPING_CENTERS}"),
    "tetrahedral_vertices": (torch.int64, -1, "I 4"),
    # Atom37-level [I, 37, ...]
    "coords": (torch.float32, 0.0, "I 37 3"),
    "element": (torch.int16, -1, "I 37"),
    "ground_truth_element": (torch.int16, 0, "I 37"),
    "charge": (torch.int16, 0, "I 37"),
    "incident_bond_type_counts": (torch.int16, 0, "I 37 10"),
    "atom37_mask": (torch.bool, False, "I 37"),
    C_CRD.full_name: (torch.float32, float("nan"), "I 37 3"),
    C_INTCRD.full_name: (torch.float32, float("nan"), "I 37 3"),
    C_IDX.full_name: (torch.bool, True, "I 37"),
    C_CRDGRP.full_name: (torch.int16, C_CRDGRP.UNKNOWN_COORDS, "I 37"),
    C_HOT.full_name: (torch.bool, False, "I 37"),
    C_RAS.full_name: (torch.int16, RasaEnum.UNSPECIFIED.value, "I 37"),
    C_HB.full_name: (torch.int16, HBondEnum.NONE.value, "I 37"),
    C_PISTACK.full_name: (torch.bool, False, "I 37"),
    C_PICATION.full_name: (torch.int16, PiCationEnum.NONE.value, "I 37"),
    C_SALT.full_name: (torch.int16, SaltBridgeEnum.NONE.value, "I 37"),
    C_HPHOB.full_name: (torch.bool, False, "I 37"),
    C_MCOORD.full_name: (torch.int16, MetalCoordEnum.NONE.value, "I 37"),
    # Pair-level [I, I]
    "bond_mask": (torch.bool, False, "I I"),
    "bond_order": (torch.int16, 0, "I I"),
    "is_aromatic": (torch.bool, False, "I I"),
    "na_ss_pair": (torch.int16, 0, "I I"),
    "na_ss_pair_mask": (torch.bool, False, "I I"),
    "token_bond_path_length": (torch.int16, 0, "I I"),
    "canonical_polymer_ordering": (torch.int8, 0, "I I"),
}

# NaN leaves padded probability channels unspecified.
for _condition in SEQUENCE_PROBABILITY_CONDITIONS:
    SPECS[_condition.full_name] = (torch.float32, float("nan"), "I")


@dataclass
class ProteinaDataSample(DataSample):
    """Flat data sample - tensors dict with specs-based padding."""

    example_id: str
    tensors: dict[str, torch.Tensor]
    atom_array: AtomArray | None = None
    task: str = ""
    metrics_tags: set = field(default_factory=set)
    extra_info: dict[str, Any] = field(default_factory=dict)
    # Input design job metadata (spec from condition_spec / CIF datasets),
    # distinct from tensor ``SPECS`` above; collated as ``specification_batch``.
    specification: dict[str, Any] | None = None
    dataset_name: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProteinaDataSample":
        feats = data["feats"]
        tensors = {}
        for k, v in feats.items():
            if not (isinstance(v, torch.Tensor) or hasattr(v, "__array__")):
                continue
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            # Apply dtype from specs if defined
            if k in SPECS:
                dtype = SPECS[k][0]
                if t.dtype != dtype:
                    t = t.to(dtype)
            tensors[k] = t

        return cls(
            example_id=data["example_id"],
            tensors=tensors,
            atom_array=data.get("atom_array"),
            task=data.get("task", ""),
            metrics_tags=data.get("metrics_tags", set()),
            extra_info=data.get("extra_info", {}),
            specification=data.get("specification"),
            dataset_name=data.get("dataset_name", "unknown"),
        )

    @cached_property
    def max_lengths(self) -> dict[str, int]:
        lengths = {}
        for name, t in self.tensors.items():
            if t.ndim == 0:
                continue
            dims = SPECS[name][2].split() if name in SPECS else ["I"]
            for i, d in enumerate(dims):
                if not d.isdigit() and i < t.ndim:
                    lengths[d] = max(lengths.get(d, 0), t.shape[i])
        return lengths

    def pad_to_max_lengths(self, max_lengths: dict[str, int]) -> "ProteinaDataSample":
        padded = {}
        for name, t in self.tensors.items():
            if t.ndim == 0:
                padded[name] = t
                continue

            spec = SPECS.get(name)
            dims = spec[2].split() if spec else ["I"] + ["?"] * (t.ndim - 1)
            pv = spec[1] if spec else (False if t.dtype == torch.bool else 0)

            # Build padding (F.pad: last dim first)
            pad = []
            for i in range(t.ndim - 1, -1, -1):
                if i < len(dims) and not dims[i].isdigit() and dims[i] in max_lengths:
                    pad.extend([0, max_lengths[dims[i]] - t.shape[i]])
                else:
                    pad.extend([0, 0])

            padded[name] = F.pad(t, pad, value=pv) if any(p > 0 for p in pad) else t

        # Zero-fill missing SPECS keys so every batch has the same dict schema
        # (one torch.compile graph instead of N).
        for name in sorted(SPECS.keys()):
            if name in padded:
                continue
            dtype, default_val, dim_str = SPECS[name]
            shape = tuple(int(d) if d.isdigit() else max_lengths.get(d, 0) for d in dim_str.split())
            padded[name] = torch.full(shape, default_val, dtype=dtype)

        return ProteinaDataSample(
            example_id=self.example_id,
            tensors=padded,
            atom_array=self.atom_array,
            task=self.task,
            metrics_tags=self.metrics_tags,
            extra_info=self.extra_info,
            specification=self.specification,
            dataset_name=self.dataset_name,
        )
