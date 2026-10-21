# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Pipeline for training for La Proteina-style models."""

import logging
import os
import pickle
from typing import ClassVar

import numpy as np
from tt_bio._vendor.atomworks.constants import (
    MASKED,
    UNKNOWN_ATOM_NAME,
)
from tt_bio._vendor.atomworks.enums import ChainType
from tt_bio._vendor.atomworks.io.utils.annotator import (
    ensure_annotations,
)
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus
from tt_bio._vendor.atomworks.io.utils.standard_annotations import S_SEGMIN
from tt_bio._vendor.atomworks.ml.conditions import (
    C_CRD,
    C_CTR,
    C_IDX,
    C_NTR,
    C_SEQ,
)
from tt_bio._vendor.atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_atom_array_has_bonds,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from tt_bio._vendor.atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinPolyResIdxAnnotation,
    ApplyFunctionToAtomArray,
    CopyAnnotation,
)
from tt_bio._vendor.atomworks.ml.transforms.atomize import AtomizeByCCDName
from tt_bio._vendor.atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    ListBuilder,
    RaiseOnCondition,
    RandomRoute,
    RemoveKeys,
    SubsetToKeys,
    Transform,
)
from tt_bio._vendor.atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from tt_bio._vendor.atomworks.ml.transforms.covalent_modifications import (
    AnnotateCovalentModifications,
)
from tt_bio._vendor.atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from tt_bio._vendor.atomworks.ml.transforms.design_task import SampleDesignTask
from tt_bio._vendor.atomworks.ml.transforms.encoding import atom_array_to_encoded_resnames
from tt_bio._vendor.atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
)
from tt_bio._vendor.atomworks.ml.transforms.mirror_transform import RandomlyMirrorInputs
from tt_bio._vendor.atomworks.ml.transforms.realize_expandable_segments import RealizeExpandableSegments
from tt_bio._vendor.atomworks.ml.transforms.symmetry import find_automorphisms_with_networkx
from tt_bio._vendor.atomworks.ml.utils.token import (
    apply_and_spread_token_wise,
    get_token_starts,
)
from typing import Any, Callable
from biotite.structure import AtomArray

from .conditions import (
    C_BP,
    C_CATH,
    C_CENTROID,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_HPHOB,
    C_IDXGRP,
    C_INTCRD,
    C_MCOORD,
    C_NA_SS_STATE,
    C_PICATION,
    C_PISTACK,
    C_RAS,
    C_SALT,
    C_SEC,
    SEQUENCE_PROBABILITY_CONDITIONS,
    BasePairEnum,
    HBondEnum,
    MetalCoordEnum,
    PiCationEnum,
    RasaEnum,
    SaltBridgeEnum,
)
from .datapipes_bonds import compute_token_bond_features
from .datapipes_schema_proteina import TokenType
from .encoding import (
    RESIDUES_BY_TOKEN_TYPE,
    TOKEN_TYPE_IDS,
    UNIFIED_ATOM37_ENCODING,
)
from .transforms_atom_names import (
    AnnotateAtomNameVisibility,
    FeaturizeAtomNamesLikeAF3,
)
from .transforms_budget import EnforceCropBudget, plan_unindex_expansion
from .transforms_chirality_feature import AddChiralityFeature
from .transforms_dna_crop import ProteinDNAContactContiguousCrop
from .transforms_geometry import CenterOnCoordinateConditionedAtoms, RandomRotationTransform
from .transforms_inference_annotations import AddSrcComponentAnnotation
from .transforms_internal_coords import ConvertAbsoluteToInternalCoords, MaterializeInternalCoords
from .transforms_invariants import ValidateBatchInvariants
from .transforms_loss_weight_predicates import build_loss_weight_upweights
from .transforms_metals import DropoutCoordinationBonds
from .transforms_na_geom import (
    FeaturizeNucleicAcidSecondaryStructure,
)
from .transforms_random_atomize_residues import RandomAtomizeResidues
from .transforms_sequence_conditioning import (
    CanonicalizeSequenceProbabilityConditions,
    FinalizeSequenceConditioning,
)
from .transforms_unindexing import UnindexFlaggedTokens
from .utils_common import true_with_prob
from .utils_constants import DEFAULT_UNDESIRED_RES_NAMES, LOCAL_LATENT_RESIDUES
from .utils_fold import extract_cath_code_by_level
from .utils_token_encoding import (
    annotate_token_centers,
    build_atom37_slots,
    has_missing_token_reference,
    missing_token_reference_mask,
)



# Config path
CATH_MAPPING_PATH = "<packaged>"  # see cath_mapping.py; shipped as source, not a pickle

logger = logging.getLogger(__name__)


class PrecomputeAtomPermutation(Transform):
    """Search for chemically equivalent atoms, in the dataloader workers.

    Must run after :py:class:`UnindexFlaggedTokens`, the last transform to change atom count
    or order, because the automorphisms index atoms globally.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName, UnindexFlaggedTokens]

    def __init__(self, max_automorphs: int = 1000):
        self.max_automorphs = max_automorphs

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_atom_array_annotation(data, required=["token_id", "pn_unit_iid", "is_polymer"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        extra_info = data.get("extra_info", {})
        automorphisms = find_automorphisms_with_networkx(atom_array, max_automorphs=self.max_automorphs)
        # ... an identity-only unit cannot permute anything; dropping it here halves the
        # per-unit loop the training step still has to run
        extra_info["atom_automorphisms"] = [group for group in automorphisms if len(group) > 1]
        extra_info["atom37_layout"] = {
            "token_ids": np.asarray(atom_array.token_id, dtype=np.int64),  # [n_atoms]
            "slots": build_atom37_slots(atom_array),  # [n_atoms], -1 where absent
        }
        data["extra_info"] = extra_info
        return data


class InferQueryPNUnitIIDsForDimers(Transform):
    """Automatically infer query_pn_unit_iids for dimer structures.

    For simple datasets (TED dimers, AFDB dimers, protein-nucleic complexes) that
    don't have query_pn_unit_iids in the parquet file, this transform automatically
    detects the interface chains and sets query_pn_unit_iids for either:

    - a protein-protein dimer (exactly 2 protein chains, no nucleic acid), or
    - a protein-nucleic complex (exactly 1 protein chain + one or more nucleic-acid
      chains), which lets the binder design task place a protein binder against a DNA/RNA target.
      The nucleic target may span multiple chains (e.g. double-stranded DNA is two chains).

    This enables protein binder design tasks to work with simple dimer datasets without needing
    metadata columns in the parquet file.
    """

    def forward(self, data: dict) -> dict:
        """Infer query_pn_unit_iids from atom_array if not already present."""
        # Skip if query_pn_unit_iids already exists (from PDB interfaces loader)
        if "query_pn_unit_iids" in data:
            return data

        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_protein", "is_nucleic_acid", "pn_unit_iid")

        protein_pn_unit_iids = np.unique(atom_array.pn_unit_iid[atom_array.is_protein])
        nucleic_pn_unit_iids = np.unique(atom_array.pn_unit_iid[atom_array.is_nucleic_acid])

        if len(protein_pn_unit_iids) == 2 and len(nucleic_pn_unit_iids) == 0:
            # Protein-protein dimer.
            data["query_pn_unit_iids"] = protein_pn_unit_iids.tolist()
        elif len(protein_pn_unit_iids) == 1 and len(nucleic_pn_unit_iids) >= 1:
            # Protein binder against a (possibly multi-chain) nucleic-acid target.
            data["query_pn_unit_iids"] = protein_pn_unit_iids.tolist() + nucleic_pn_unit_iids.tolist()

        return data


class ConvertToAtomLevelProteinaFeatures(Transform):
    """Convert inputs to Proteina features and conditioning annotations.

    Output features:
        - coords: [L, 3] coordinates in Angstroms, centered at origin
        - residue_type: [L] residue type encoding from UNIFIED_ATOM37_ENCODING
        - tok_idx: [L] token index for each atom
        - element: [L] atomic number (provided for sequence-conditioned tokens)
        - chain_iid: [L] chain instance ID
        - Conditioning masks and values (C_CRD, C_SEQ, C_IDXGRP, etc.) of shape [L]
        - supports_local_latent: [L] bool mask for generated, non-atomized residue types in the AE latent alphabet
        - is_in_context_conditioning: [L] bool mask for tokens used only as in-context conditioning
    """

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["supports_local_latent", "coord_to_be_noised"])

    def forward(self, data: dict) -> dict:
        """Convert encoded data into atom-level Proteina features."""
        feats = data.get("feats", {})
        aa = as_atom_array_plus(data["atom_array"])
        ensure_annotations(
            aa,
            "is_standard_unknown_or_masked_aa",
            "is_standard_unknown_or_masked_dna",
            "is_standard_unknown_or_masked_rna",
            "atomize",
        )

        # ... index features
        feats["tok_idx"] = aa.token_id  # [L]

        # Get unique chain IDs and random permutation
        # NOTE: We ensure that the chain_iid is randomly assigned for each chain in the structure,
        # rather than always assigning the first chain as chain_iid=0, second as chain_iid=1, etc.
        unique_chains, inverse_indices = np.unique(aa.chain_iid, return_inverse=True)
        random_chain_indices = np.random.permutation(len(unique_chains))
        feats["chain_iid"] = random_chain_indices[inverse_indices]  # [L]

        # Get unique molecule IDs and random permutation (same logic as chain_iid)
        unique_molecules, mol_inverse_indices = np.unique(aa.molecule_iid, return_inverse=True)
        random_molecule_indices = np.random.permutation(len(unique_molecules))
        feats["molecule_iid"] = random_molecule_indices[mol_inverse_indices]  # [L]

        # molecule_entity groups all copies of one molecule (same sequence)
        # NOTE: Only used as a pair feature via equality check; absolute values are not meaningful
        assert "molecule_entity" in aa.get_annotation_categories(), "molecule_entity annotation missing"

        _, entity_ids = np.unique(aa.molecule_entity, return_inverse=True)  # [L], contiguous 0..k-1
        if data.get("is_inference", False):
            # Any masked sequence gives the entire chain a private entity.
            # Fully sequence-conditioned chains retain their native entity relationships.
            masked_sequence = ~C_SEQ.mask(aa, default="raise")
            next_entity = int(entity_ids.max()) + 1 if entity_ids.size else 0
            for chain in np.unique(aa.chain_iid[masked_sequence]):
                entity_ids[aa.chain_iid == chain] = next_entity
                next_entity += 1

        # TODO: Rename to molecule_entity after performing checkpoint surgery
        feats["molecule_entity_id"] = entity_ids.astype(np.int16)  # [L]

        feats["residue_type"] = atom_array_to_encoded_resnames(aa, UNIFIED_ATOM37_ENCODING, atomize_token="<A>")  # [L]

        # ground truth coordinates
        feats["coords"] = aa.coord_to_be_noised  # [L, 3]

        token_type = np.full(aa.array_length(), TokenType.UNKNOWN, dtype=np.int16)
        token_type[aa.get_annotation("is_standard_unknown_or_masked_dna")] = TokenType.DNA
        token_type[aa.get_annotation("is_standard_unknown_or_masked_rna")] = TokenType.RNA
        token_type[aa.get_annotation("is_standard_unknown_or_masked_aa")] = TokenType.PROTEIN
        for group, names in RESIDUES_BY_TOKEN_TYPE.items():
            if TOKEN_TYPE_IDS[group] != TokenType.UNKNOWN:
                token_type[np.isin(aa.res_name, names)] = TOKEN_TYPE_IDS[group]
        token_type[aa.get_annotation("atomize")] = TokenType.ATOMIZED
        aa.set_annotation("token_type", token_type)
        feats["token_type"] = token_type  # [L]

        # ... resolution (token-uniform so first-atom pooling preserves the all-atoms contract)
        is_resolved = (aa.occupancy > 0.0) & ~np.isnan(aa.coord).any(axis=-1)
        feats["is_resolved_mask"] = apply_and_spread_token_wise(aa, is_resolved, np.all)  # [L]

        # ... encoding annotations
        # (Will be pooled to the token-level later)
        feats["supports_local_latent"] = aa.supports_local_latent  # [L]
        feats["is_in_context_conditioning"] = apply_and_spread_token_wise(
            aa, ~C_IDX.mask(aa, default="generate"), np.any
        )  # [L]

        # Per-token MSE weight: defaults to 1.0 if no UpweightLossWeightOnSubset ran.
        if "loss_weight" in aa.get_annotation_categories():
            feats["loss_weight"] = aa.loss_weight  # [L] float32
        else:
            feats["loss_weight"] = np.ones(aa.array_length(), dtype=np.float32)  # [L]

        feats["coord_cond_mask"] = C_CRD.mask(aa, default="generate")

        # Insertion source index for insertion-prediction training task:
        # -1 for indexed (generated) tokens; j>=0 for unindexed tokens (source token index)
        if "insertion_source_idx" in aa.get_annotation_categories():
            feats["insertion_source_idx"] = aa.insertion_source_idx  # [L]

        ###########################################################
        # Conditioning information
        ###########################################################

        # +------ Sequence conditioning (residue type conditioned) ------+
        # C_SEQ.annotation holds the frozen conditioning res-code (MASKED where unset); encode it directly.
        seq_cond_mask = C_SEQ.mask(aa, default="raise")  # [L] True where seq-conditioned
        _mask_token_name = (
            MASKED
            if MASKED in UNIFIED_ATOM37_ENCODING.token_to_idx
            else UNIFIED_ATOM37_ENCODING.resolve_unknown_token_name(MASKED, False)
        )
        seq_annotation = C_SEQ.annotation(aa, default="generate")  # [L] res_name strings; MASKED where unset
        c_seq_encoded = atom_array_to_encoded_resnames(
            aa, UNIFIED_ATOM37_ENCODING, atomize_token="<A>", res_names=seq_annotation
        )  # [L]
        c_seq_encoded[(~seq_cond_mask) & (~aa.atomize)] = UNIFIED_ATOM37_ENCODING.token_to_idx[_mask_token_name]
        feats[C_SEQ.full_name] = c_seq_encoded  # [L]

        # Elements are provided if they are part of a sequence conditioned token (which should already include
        # ligands and DNA/RNA). If the token is not sequence conditioned, we pass through the auto-encoder without
        # element information, but we'll remove it here to be thorough
        feats["ground_truth_element"] = aa.atomic_number.astype(np.int16)
        element_mask = seq_cond_mask
        feats["element"] = np.full(aa.array_length(), -1, dtype=np.int32)
        if any(element_mask):
            feats["element"][element_mask] = aa.atomic_number[element_mask]  # [L]

        # Charges are provided for sequence-conditioned tokens (especially ligands)
        # If charge annotation doesn't exist, initialize with zeros
        if "charge" not in aa.get_annotation_categories():
            aa.set_annotation("charge", np.zeros(aa.array_length(), dtype=np.int16))
        feats["charge"] = aa.charge.astype(np.int16)  # [L]

        # +------ CAT domain annotations from CATH ------+
        cath_codes = C_CATH.annotation(aa, default="generate")
        unique_cath_codes = np.unique(cath_codes)  # [num_codes]

        output_cath_codes = np.full((len(cath_codes), 3), -1, dtype=np.int32)
        if CATH_MAPPING_PATH:
            from .cath_mapping import CATH_MAPPING as cath_class_mapping

            cath_class_mapping_C = cath_class_mapping["C"]
            cath_class_mapping_A = cath_class_mapping["A"]
            cath_class_mapping_T = cath_class_mapping["T"]

            remapped_unique_cath_codes = np.zeros((len(unique_cath_codes), 3), dtype=np.int32)
            for i, cath_code in enumerate(unique_cath_codes):
                remapped_unique_cath_codes[i, 0] = cath_class_mapping_C.get(
                    extract_cath_code_by_level(cath_code, "C"), -1
                )
                remapped_unique_cath_codes[i, 1] = cath_class_mapping_A.get(
                    extract_cath_code_by_level(cath_code, "A"), -1
                )
                remapped_unique_cath_codes[i, 2] = cath_class_mapping_T.get(
                    extract_cath_code_by_level(cath_code, "T"), -1
                )

            for i, code in enumerate(unique_cath_codes):
                output_cath_codes[cath_codes == code, :] = remapped_unique_cath_codes[i]

        elif len(unique_cath_codes) > 1 or unique_cath_codes[0] != C_CATH.NULL_VALUE:
            raise ValueError(
                f"CATH code mapping file not found at {CATH_MAPPING_PATH}, CATH conditioning must be default"
            )

        feats[C_CATH.full_name] = output_cath_codes

        # +------ Coordinate conditioning (absolute) ------+
        feats[C_CRD.full_name] = C_CRD.annotation(aa, default="generate")  # [L, 3] conditioning coords

        # +------ Coordinate conditioning (internal) ------+
        feats[C_INTCRD.full_name] = C_INTCRD.annotation(aa, default="generate")  # [L]

        # +------ Other atom-level conditions ------+
        # ... basic
        feats[C_IDX.full_name] = C_IDX.annotation(aa, default="generate")  # [L]

        # ... fancy
        feats[C_HOT.full_name] = C_HOT.annotation(aa, default="generate")
        for condition in SEQUENCE_PROBABILITY_CONDITIONS:
            feats[condition.full_name] = condition.annotation(aa, default="generate")
        for condition in (C_RAS, C_SEC, C_HB, C_PICATION, C_SALT, C_MCOORD):
            feats[condition.full_name] = np.asarray(condition.annotation(aa, default="generate"), dtype=np.int64)
        feats[C_PISTACK.full_name] = C_PISTACK.annotation(aa, default="generate")
        feats[C_HPHOB.full_name] = C_HPHOB.annotation(aa, default="generate")
        feats[C_IDXGRP.full_name] = C_IDXGRP.annotation(aa, default="generate")
        feats[C_CRDGRP.full_name] = C_CRDGRP.annotation(aa, default="generate")
        feats[C_CENTROID.full_name] = C_CENTROID.annotation(
            aa, default="generate"
        )  # (n_atoms, 3) Å, NaN where unincluded

        # ... add to data dict
        data["feats"] = feats
        return data


EXISTING_TOKEN_LEVEL_FEATURES = ["centroid_of_atoms_without_coordinate_conditioning"]
"""Features already at the token level that simply pass through."""

ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL = [
    "residue_type",
    C_SEQ.full_name,
    "token_type",
    "tok_idx",
    "chain_iid",
    "molecule_iid",
    # TODO: Rename molecule_entity
    "molecule_entity_id",
    "supports_local_latent",
    "is_in_context_conditioning",
    "is_resolved_mask",
    "insertion_source_idx",
    "loss_weight",
    "coord_cond_mask",
    C_CTR.full_name,
    C_NTR.full_name,
    C_SEC.full_name,
    *[condition.full_name for condition in SEQUENCE_PROBABILITY_CONDITIONS],
    C_IDXGRP.full_name,
    C_CATH.full_name,
    C_CENTROID.full_name,  # Per-chain centroid xyz (broadcast per atom; first-atom-per-token is fine)
    "chirality",
    "chiral_frame_atoms",
    "chiral_slot_owner",
    "tetrahedral_vertices",
]
"""Features to convert from atom-level to token-level by taking the first atom in each token."""

ATOM37_FEATURES = [
    ("coords", (3,), 0.0),
    ("element", (), -1),
    ("ground_truth_element", (), 0),
    ("charge", (), 0),  # Atomic charge
    (C_CRD.full_name, (3,), np.nan),
    (C_INTCRD.full_name, (3,), np.nan),
    (C_IDX.full_name, (), True),
    (C_CRDGRP.full_name, (), C_CRDGRP.UNKNOWN_COORDS),
    (C_HOT.full_name, (), False),
    (C_RAS.full_name, (), RasaEnum.UNSPECIFIED.value),
    (C_HB.full_name, (), HBondEnum.NONE.value),
    (C_PISTACK.full_name, (), False),
    (C_PICATION.full_name, (), PiCationEnum.NONE.value),
    (C_SALT.full_name, (), SaltBridgeEnum.NONE.value),
    (C_HPHOB.full_name, (), False),
    (C_MCOORD.full_name, (), MetalCoordEnum.NONE.value),
]
"""Features to convert from atom-level to token-level atom37 representation.

Schema: (feature_name, extra_dims, pad_value)
"""


class ConvertToAtom37Format(Transform):
    """Convert atom-level features to token-level atom37 representation.

    Atomized residues place their first (and only) atom in slot 1.

    Input (from :py:class:`ConvertToAtomLevelProteinaFeatures`):
        feats with atom-level tensors:
        - coords: [n_atoms, 3]
        - residue_type: [n_atoms]
        - element: [n_atoms]
        - tok_idx: [n_atoms]
        - etc.

    Output:
        feats with token-level tensors:
        - coords: [n_tokens, 37, 3] - Atom37 coordinates
        - element: [n_tokens, 37] - Atomic numbers per slot
        - atom37_mask: [n_tokens, 37] - Valid atom slots
        - Token-level features: residue_type: [n_tokens], etc.
    """

    def forward(self, data: dict) -> dict:
        """Convert atom-level features to token-level atom37."""

        result = {}

        # Extract basic info
        feats = data["feats"]
        atom_array = data["atom_array"]

        token_starts = get_token_starts(atom_array, add_exclusive_stop=False)
        num_tokens = len(token_starts)
        atom_names = np.char.strip(atom_array.atom_name.astype(str))  # [L]

        # +--------- Pass through existing token-level features ---------+
        for feat_name in EXISTING_TOKEN_LEVEL_FEATURES:
            if feat_name in feats:
                result[feat_name] = feats[feat_name]

        # +--------- Convert atom-level features to token-level by using the first atom in each token ---------+

        for feat_name in ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL:
            if feat_name in feats:
                result[feat_name] = feats[feat_name][token_starts]

        # +--------- Build Atom37 features by scattering atoms into appropriate slots ---------+

        # Initialize atom37 arrays and atom mask
        atom37_arrays = {}
        for feat_name, extra_dims, pad_value in ATOM37_FEATURES:
            if feat_name in feats:
                shape = (num_tokens, 37, *extra_dims)
                atom37_arrays[feat_name] = np.full(shape, pad_value, dtype=feats[feat_name].dtype)  # [L, 37, ...]

        atom37_mask = np.zeros((num_tokens, 37), dtype=bool)  # [L, 37]
        atom_slots = build_atom37_slots(atom_array)

        missing_slot = atom_slots < 0
        coordinate_free_stub = (
            S_SEGMIN.mask(atom_array, default="generate")
            & (atom_array.res_name == MASKED)
            & (atom_names == UNKNOWN_ATOM_NAME)
        )
        unsupported_atom = missing_slot & ~coordinate_free_stub
        if unsupported_atom.any():
            example_indices = np.flatnonzero(unsupported_atom)[:5]
            atoms = ", ".join(f"{atom_array.res_name[i]}:{atom_names[i]}" for i in example_indices)
            raise ValueError(f"Atoms cannot be represented in Atom37: {atoms}")

        has_slot = ~missing_slot
        slot_token_id = atom_array.token_id[has_slot]
        valid_slots = atom_slots[has_slot]

        # Scatter atoms into atom37 arrays (vectorized)
        atom37_mask[slot_token_id, valid_slots] = True

        result["incident_bond_type_counts"] = np.zeros((num_tokens, 37, 10), dtype=np.int16)
        if atom_array.bonds is None:
            raise ValueError("Incident bond-type features require an explicit BondList")
        bonds = atom_array.bonds.as_array().astype(np.int64)
        for endpoint in (0, 1):
            atoms = bonds[:, endpoint]
            valid = has_slot[atoms]
            np.add.at(
                result["incident_bond_type_counts"],
                (atom_array.token_id[atoms[valid]], atom_slots[atoms[valid]], bonds[valid, 2]),
                1,
            )

        # Scatter all features at once
        for feat_name in atom37_arrays:
            atom37_arrays[feat_name][slot_token_id, valid_slots] = feats[feat_name][has_slot]

        result["atom37_mask"] = atom37_mask
        result.update(atom37_arrays)

        # Residue-level NA secondary structure is assembled at tokenization time:
        # - sparse 2D base-pair edges live on representative token atoms
        # - loop/specification state lives as 1D atom annotations
        n_tokens = len(token_starts)
        na_ss_pair = np.zeros((n_tokens, n_tokens), dtype=np.int16)
        na_ss_pair_mask = np.zeros((n_tokens, n_tokens), dtype=bool)

        if C_BP.full_name in atom_array.get_annotation_categories(n_body=2):
            pair_annotation = C_BP.annotation(atom_array, default="raise")
            atom_to_token = np.full(atom_array.array_length(), -1, dtype=np.int32)
            atom_to_token[token_starts] = np.arange(n_tokens, dtype=np.int32)
            token_pairs = atom_to_token[pair_annotation.pairs]
            valid_pairs = np.all(token_pairs >= 0, axis=1)
            token_pairs = token_pairs[valid_pairs]
            if len(token_pairs) > 0:
                na_ss_pair[token_pairs[:, 0], token_pairs[:, 1]] = 1
                na_ss_pair_mask[token_pairs[:, 0], token_pairs[:, 1]] = True

        na_ss_state = np.full((n_tokens,), BasePairEnum.UNSPECIFIED.value, dtype=np.int16)
        na_ss_state_mask = np.zeros((n_tokens,), dtype=bool)
        if C_NA_SS_STATE.full_name in atom_array.get_annotation_categories():
            na_ss_state = C_NA_SS_STATE.annotation(atom_array, default="generate")[token_starts].astype(np.int16)
            na_ss_state_mask = C_NA_SS_STATE.mask(atom_array, default="generate")[token_starts].astype(bool)

        # Preserve the old "explicit negative row/column" semantics for loop tokens via the pair mask.
        loop_inds = np.where(na_ss_state == BasePairEnum.LOOP.value)[0]
        if len(loop_inds) > 0:
            na_ss_pair_mask[loop_inds, :] = True
            na_ss_pair_mask[:, loop_inds] = True

        result["na_ss_pair"] = na_ss_pair
        result["na_ss_pair_mask"] = na_ss_pair_mask
        result["na_ss_state"] = na_ss_state
        result["na_ss_state_mask"] = na_ss_state_mask

        # Override existing atom-level features with atom37 versions
        data["feats"] = result

        return data


class ComputeTokenLevelBondFeaturesLikeAF3(Transform):
    """Compute token-level bond features like AF3.

    Computes the following token-level bond features:
    - bond_mask: [n_tokens, n_tokens] - True if there is a bond between any atom in token i and token j
    - bond_order: [n_tokens, n_tokens] - Kekulé bond order (0=none, 1=single, 2=double, 3=triple, 4=coordination)
    - is_aromatic: [n_tokens, n_tokens] - True if the bond is aromatic
    - token_bond_path_length: [n_tokens, n_tokens] - shortest bond-graph path length between tokens

    Path lengths run on the widest bond graph the pipeline still holds -- the snapshot from before
    :py:class:`RemoveTokensWithAnyNaNAtom` where it dropped anything, else the pre-crop array -- so
    that hops through tokens since removed are still counted.

    NOTE: This is computed after DesignTask application so that C_IDXGRP is known
    """

    def __init__(
        self,
        path_length_search_cutoff: int | None = None,
        fill_res_id_gaps: bool = False,
    ):
        self.fill_res_id_gaps = fill_res_id_gaps
        self.path_length_cutoff = path_length_search_cutoff

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array", "feats"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_has_bonds(data)
        check_atom_array_annotation(data, ["is_polymer", "atomize", "insertion_source_idx"])

    def forward(self, data: dict) -> dict:
        crop_info = data.get("crop_info", {})
        nan_info = data.pop("pre_nan_removal_info", None)

        if nan_info is not None:
            source_atom_array = nan_info["atom_array"]
            source_token_idxs = nan_info["kept_token_idxs"]
            if "atom_array" in crop_info:
                source_token_idxs = source_token_idxs[crop_info["crop_token_idxs"]]
        elif "atom_array" in crop_info:
            # No NaN token was dropped, so the pre-crop array is the widest graph available.
            source_atom_array = crop_info["atom_array"]
            source_token_idxs = crop_info["crop_token_idxs"]
        else:
            source_atom_array = source_token_idxs = None

        bond_feats = compute_token_bond_features(
            data["atom_array"],
            path_length_cutoff=self.path_length_cutoff,
            source_atom_array=source_atom_array,
            source_token_idxs=source_token_idxs,
            fill_res_id_gaps=self.fill_res_id_gaps,
        )
        data["feats"].update(bond_feats)
        return data


class DropoutAromaticFlag(Transform):
    """Drop out the ``is_aromatic`` flag per-example with a given probability.

    With probability ``p``, all ``is_aromatic`` entries are zeroed for the example.
    Only added to pipeline during training.
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def forward(self, data: dict) -> dict:
        if self.p > 0 and np.random.rand() < self.p:
            data["feats"]["is_aromatic"][:] = False

        return data


def add_supports_local_latent_annotation(
    atom_array: AtomArray,
    residue_names_supporting_local_latents: list | set,
) -> AtomArray:
    """Annotate generated, non-atomized polymer tokens represented by the autoencoder latent alphabet.

    This classifies token eligibility only. Runtime attention, output, and
    supervision masks also account for padding and resolution.

    Returns:
        atom_array with the `supports_local_latent` annotation set
    """
    is_supported_residue = np.isin(atom_array.res_name, residue_names_supporting_local_latents)

    has_atomize = "atomize" in atom_array.get_annotation_categories()
    if has_atomize:
        is_atomized = atom_array.atomize
    else:
        is_atomized = np.zeros(atom_array.array_length(), dtype=bool)

    is_in_context_conditioning = apply_and_spread_token_wise(
        atom_array, ~C_IDX.mask(atom_array, default="generate"), np.any
    )
    supports_local_latent = is_supported_residue & ~is_atomized & ~is_in_context_conditioning
    atom_array.set_annotation("supports_local_latent", supports_local_latent)

    return atom_array


class AddSupportsLocalLatentAnnotation(Transform):
    """Annotate generated, non-atomized tokens represented by the autoencoder latent alphabet."""

    def __init__(
        self,
        residue_names_supporting_local_latents: list | set,
    ):
        super().__init__()
        self.residue_names_supporting_local_latents = residue_names_supporting_local_latents

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        add_supports_local_latent_annotation(atom_array, self.residue_names_supporting_local_latents)
        return data


class UpweightLossWeightOnSubset(Transform):
    """Add ``factor`` to per-atom ``loss_weight`` on atoms matching ``predicate``.

    Implements AF3-style additive composition: ``w = 1 + Σ αₖ · is_k(atom_array)``.

    NOTE: The predicate's atom-level mask is spread token-wise (``np.max``), so a predicate
    that flags even a single atom in a token (e.g. the side-chain tip atom)
    upweights the whole token.

    Args:
      predicate: ``AtomArray -> np.ndarray[bool]`` of shape ``[n_atoms]``.
      factor: scalar alpha. ``factor=0.0`` is a no-op fast-path.
    """

    def __init__(
        self,
        predicate: Callable[[AtomArray], np.ndarray],
        factor: float,
    ):
        self.predicate = predicate
        self.factor = float(factor)

    def forward(self, data: dict) -> dict:
        if self.factor == 0.0:
            return data

        # Get or initialize loss_weight annotation
        atom_array = data["atom_array"]
        if "loss_weight" not in atom_array.get_annotation_categories():
            weight = np.ones(atom_array.array_length(), dtype=np.float32)
        else:
            weight = np.asarray(atom_array.loss_weight, dtype=np.float32)

        # Apply predicate to get atom-level mask

        is_subset = self.predicate(atom_array).astype(bool)
        # Spread token-wise: any matching atom upweights the whole token.
        is_subset_token_uniform = apply_and_spread_token_wise(atom_array, is_subset, np.max)
        weight = weight + (self.factor * is_subset_token_uniform.astype(np.float32))

        # Update annotation
        atom_array.set_annotation("loss_weight", weight.astype(np.float32))

        return data


class RemoveTokensWithAnyNaNAtom(Transform):
    """Remove tokens with NaN coordinates or a missing residue reference atom.

    There are a few types of situations where NaNs can show up in a token:
    1. A residue has a single missing side-chain atom
        In this case, we cannot generate a latent representation fo the sidechain and therefore cannot create noise
        to train
    2. A residue is entirely nan coordinates
        Obviously in this case we cannot encode the token and therefore cannot make a noisy version of the token
    3. An atomized token has NaN coordinates (e.g. a ligand is missing an atom coordinate)
        We would need to have a "CA" coordinate to add noise to, so we cannot encode this token either

    NOTE: This excises on coordinates, NOT on occupancy. It IS possible to have zero
    occupancy but resolved coordinates (e.g., missing sidechain atoms imputed by Rosetta).

    The input is snapshotted into ``data["pre_nan_removal_info"]`` for
    :py:class:`ComputeTokenLevelBondFeaturesLikeAF3` to run its BFS on, together with the token
    indices into it that survive here.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        starts = get_token_starts(atom_array, add_exclusive_stop=True)
        missing_reference = missing_token_reference_mask(atom_array)
        mask_to_keep = np.ones(atom_array.array_length(), dtype=bool)
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            if missing_reference[i] or np.any(np.isnan(atom_array.coord[s:e])):
                mask_to_keep[s:e] = False
        if mask_to_keep.all():
            return data
        data["pre_nan_removal_info"] = {
            "atom_array": atom_array.copy(),
            "kept_token_idxs": np.where(mask_to_keep[starts[:-1]])[0],
        }
        data["atom_array"] = atom_array[mask_to_keep]
        return data


class ValidateNonEmptyStructure(Transform):
    """Validate that structure has atoms after filtering.

    Raises an error if the atom array is empty, which can happen after
    aggressive filtering of hydrogens, unresolved units, and undesired residues.
    """

    def forward(self, data: dict) -> dict:
        aa = data["atom_array"]
        if aa.array_length() == 0:
            example_id = data.get("example_id", "unknown")
            raise ValueError(
                f"Structure has no atoms after filtering (example_id: {example_id}). This structure will be skipped."
            )
        return data


class FeaturizeExpandedSegments(Transform):
    """
    Currently only polypeptide expandable segments are supported by the model. Here we ensure that only
    polypeptide expandable segments make it through the transform pipeline.

    At inference, unindexed segments contribute only their sampled residue-index gaps; their
    placeholders are removed before creating in-context conditioning tokens.

    The model featurizes coordinates by multiplying the coords from the AtomArray by a mask. Thus, nan
    coords from indexed expandable segments need to be replaced with 0s.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        expanded_segment_mask = S_SEGMIN.mask(atom_array, default="generate")
        if expanded_segment_mask.any():
            # Check for only MASKED_AA gap residues! (TODO: Remove when we support DNA and RNA)
            if not np.all(atom_array.chain_type[expanded_segment_mask] == ChainType.POLYPEPTIDE_L):
                raise ValueError("Only polypeptide expandable segments are currently supported.")

            if data.get("is_inference", False):
                unindexed_segments = expanded_segment_mask & ~C_IDX.mask(atom_array, default="generate")
                if unindexed_segments.any():
                    # Preserve res_id gaps so motif offsets and bond-path lengths retain the sampled spacing.
                    atom_array = atom_array[~unindexed_segments].copy()
                    expanded_segment_mask = expanded_segment_mask[~unindexed_segments]

            atom_array.coord[expanded_segment_mask] = 0.0
        data["atom_array"] = atom_array
        return data


class FeaturizeCentroidOfAtomsWithoutCoordinateConditioning(Transform):
    """Add centroid of non-coordinate-conditioned atoms as a feature.

    Computes the centroid (center of mass) of all atoms that are NOT
    coordinate-conditioned and adds it as the ``centroid_of_atoms_without_coordinate_conditioning`` feature.
    This represents the approximate center of the region to be generated.

    Args:
        coord_annotation: Annotation name for coordinates to use.
            Defaults to ``"coord_to_be_noised"``.
        feature_name: Feature name for output centroid.
            Defaults to ``"centroid_of_atoms_without_coordinate_conditioning"``.
    """

    def __init__(
        self,
        coord_annotation: str = "coord_to_be_noised",
        feature_name: str = "centroid_of_atoms_without_coordinate_conditioning",
    ):
        super().__init__()
        self.coord_annotation = coord_annotation
        self.feature_name = feature_name

    def forward(self, data: dict) -> dict:
        """Compute and add centroid of non-coordinate-conditioned atoms."""
        aa = as_atom_array_plus(data["atom_array"])
        coords = aa.get_annotation(self.coord_annotation)

        # Get mask for atoms without absolute coordinate conditioning
        not_coord_mask = ~C_CRD.mask(aa, default="generate")

        # Compute centroid of non-coordinate-conditioned atoms
        # Use nanmean to ignore NaN coordinates from unresolved atoms
        centroid = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        if not_coord_mask.any():
            centroid = np.nanmean(coords[not_coord_mask], axis=0)

        # Add to features dict
        feats = data.get("feats", {})
        feats[self.feature_name] = centroid
        data["feats"] = feats

        return data


class FeaturizeChainCentroid(Transform):
    """Write per-chain centroid (from ``coord_annotation``, plus Gaussian noise) to ``C_CENTROID``.

    Centroid conditioning is all-or-nothing per example: when the per-example gate fires, every
    chain with at least one resolved coordinate receives a centroid; otherwise none do.

    Args:
        p_add_centroid_condition: Per-example gate probability.
        noise_std_angstrom: Isotropic Gaussian noise std (Å) per included chain.
        coord_annotation: Source coord annotation; must be post-centering for centroids to live in the loss frame.
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = ["CenterOnCoordinateConditionedAtoms"]

    def __init__(
        self,
        p_add_centroid_condition: float = 0.2,
        noise_std_angstrom: float = 5.0,
        coord_annotation: str = "coord_to_be_noised",
    ):
        super().__init__()
        self.p_add_centroid_condition = float(p_add_centroid_condition)
        self.noise_std_angstrom = float(noise_std_angstrom)
        self.coord_annotation = coord_annotation

    def forward(self, data: dict) -> dict:
        aa = as_atom_array_plus(data["atom_array"])
        n_atoms = aa.array_length()

        # Default: NaN centroid (mask is implicit via isfinite).
        centroid_arr = np.full((n_atoms, 3), fill_value=np.nan, dtype=np.float32)

        if true_with_prob(self.p_add_centroid_condition):
            coords = aa.get_annotation(self.coord_annotation)  # (n_atoms, 3) Å, post-centering
            chain_iid = aa.chain_iid
            for chain_id in np.unique(chain_iid):
                chain_mask = chain_iid == chain_id
                chain_coords = coords[chain_mask]
                if np.isnan(chain_coords).all():
                    continue
                chain_centroid = np.nanmean(chain_coords, axis=0).astype(np.float32)  # (3,)
                noise = (np.random.randn(3) * self.noise_std_angstrom).astype(np.float32)  # (3,)
                centroid_arr[chain_mask] = chain_centroid + noise

        C_CENTROID.set_annotation(aa, centroid_arr)
        data["atom_array"] = aa
        return data


def _get_design_task_name(data: dict) -> str | None:
    """Get the design task name from the data dict."""
    return (data.get("task") or {}).get("name")


def _has_missing_token_reference(data: dict[str, Any]) -> bool:
    """Adapt the reference-atom check to pipeline data for spawned workers."""
    return has_missing_token_reference(data["atom_array"])


def _build_design_clean_and_sample_pipeline(
    crop_size: int | None = None,
    crop_size_range: tuple[int, int] | tuple[float, float] | None = None,
    max_crop_size: int | None = None,
    p_no_subcrop: float = 0.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    dna_contact_crop_probability: float = 0.0,
    crop_center_cutoff_distance: float = 15.0,
    max_atoms_in_crop: int | None = None,
    is_inference: bool = False,
    atomization_probabilities: dict[str, float] | None = None,
    design_tasks: dict[str, dict[str, Any]] = {},
    undesired_res_names: list[str] = DEFAULT_UNDESIRED_RES_NAMES,
    b_factor_occupancy_min: float | None = None,  # Zero occupancy below this b-factor; off by default
    p_random_rotation: float = 1.0,  # Probability of applying random rotation
    p_mirror: float = 0.0,  # Probability of mirroring inputs (chirality augmentation)
    **kwargs,  # to dump remaining args for now
) -> list[Transform]:
    # T = transform shorthand for readability
    T = ListBuilder()

    # (We may want to run the train pipe with is_inference=True; e.g., for simple validations of our training objective)
    T += AddData({"is_inference": is_inference})
    ###########################################################
    ### Step 0: Data Augmentation                           ###
    ###########################################################
    # Apply random rotation for data augmentation (only during training)
    if not is_inference and p_random_rotation > 0.0:
        T += RandomRotationTransform(p_apply=p_random_rotation)

    # Apply random mirroring for chirality augmentation (only during training)
    if not is_inference and p_mirror > 0.0:
        T += RandomlyMirrorInputs(mirror_prob=p_mirror)

    ###########################################################
    ### Step 1: Filter unwanted information                 ###
    ###########################################################
    # ... shared cleanup (safe for both train and inference)
    T += RemoveKeys(["atom_array_stack"], require_keys_exist=False)
    T += RemoveHydrogens()
    T += RemoveTerminalOxygen()
    T += RemoveNucleicAcidTerminalOxygen()

    # ... training-only heavy cleanup (drops things an inference user may want preserved,
    # or relies on extra_info that inference inputs don't carry)
    if not is_inference:
        T += FilterToSpecifiedPNUnits(extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing")
        T += RemoveUnresolvedPNUnits()
        T += HandleUndesiredResTokens(undesired_res_tokens=undesired_res_names)
        T += RemovePolymersWithTooFewResolvedResidues(min_residues=1)
        T += SetOccToZeroOnBfactor(bmin=b_factor_occupancy_min)

    # Validate structure has atoms after filtering (defensive; useful for both)
    T += ValidateNonEmptyStructure()

    # ... add basic annotations
    ###########################################################
    ### 2. Add basic annotations                            ###
    ###########################################################
    T += AddGlobalAtomIdAnnotation(allow_overwrite=True)
    # Keep modified polymer residues in their PN units and preserve existing atomization.
    T += AnnotateCovalentModifications(
        reassign_modified_residue=False,
        atomize_modified_residue=False,
    )
    if not is_inference:
        T += RandomAtomizeResidues(probabilities=atomization_probabilities)
    T += AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=LOCAL_LATENT_RESIDUES,
        move_atomized_part_to_end=False,
        validate_atomize=False,
    )
    T += AddGlobalTokenIdAnnotation()

    ###########################################################
    ### Step 3: Crop And Remove Unusable tokens             ###
    ###########################################################
    # Remove tokens with any NaN coordinates (e.g., from unresolved atoms). We cannot generate noisy versions of these.
    if not is_inference:
        T += RemoveTokensWithAnyNaNAtom()
        T += ValidateNonEmptyStructure()
    else:
        T += RaiseOnCondition(
            condition=_has_missing_token_reference,
            error_message="Residue tokens lack their required atom37 slot-1 reference atom. "
            "Provide the reference atom from the original structure or explicitly supply atomized context.",
        )

    if (
        not is_inference
        and (crop_size or crop_size_range)
        and (crop_contiguous_probability > 0 or crop_spatial_probability > 0 or dna_contact_crop_probability > 0)
    ):
        if crop_size_range is not None:
            assert crop_size is None, "crop_size must be None when crop_size_range is set, to avoid ambiguity"
        else:
            assert crop_size is not None and crop_size > 0, "Crop size must be greater than 0"
        assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"
        T += ApplyFunctionToAtomArray(annotate_token_centers)

        crop_routes = [
            (
                crop_contiguous_probability,  # DISCUSS: Does this even make sense for design??
                CropContiguousLikeAF3(
                    crop_size=crop_size,
                    crop_size_range=crop_size_range,
                    max_crop_size=max_crop_size,
                    p_no_subcrop=p_no_subcrop,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms_in_crop,
                    annotate_crop_boundary=True,
                ),
            ),
            (
                crop_spatial_probability,
                CropSpatialLikeAF3(
                    crop_size=crop_size,
                    crop_size_range=crop_size_range,
                    max_crop_size=max_crop_size,
                    p_no_subcrop=p_no_subcrop,
                    crop_center_cutoff_distance=crop_center_cutoff_distance,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms_in_crop,
                    raise_if_missing_crop_center=False,
                    annotate_crop_boundary=True,
                ),
            ),
        ]

        # Protein-DNA contact-centered contiguous crop. Keeps
        # a contiguous protein region + contiguous DNA duplex around a sampled
        # binding contact; falls back to a spatial crop when no contact exists.
        # Needs the within_poly_res_idx annotation to walk contiguous windows.
        # TODO: We should be able to delete this in favor of a generalized ProteinBinder design task and crop
        if dna_contact_crop_probability > 0:
            T += AddWithinPolyResIdxAnnotation()
            crop_routes.append(
                (
                    dna_contact_crop_probability,
                    ProteinDNAContactContiguousCrop(
                        crop_size=crop_size,
                        crop_size_range=crop_size_range,
                        max_crop_size=max_crop_size,
                        crop_center_cutoff_distance=crop_center_cutoff_distance,
                        protein_contact_type="all",
                        dna_contact_type="base",
                        keep_uncropped_atom_array=True,
                        max_atoms_in_crop=max_atoms_in_crop,
                    ),
                )
            )

        T += RandomRoute.from_list(crop_routes)

    ###########################################################
    ### Step 3.5: Infer query_pn_unit_iids for dimers      ###
    ###########################################################
    # For simple datasets (TED dimers, AFDB dimers) that don't have query_pn_unit_iids
    # from the parquet loader, automatically detect and set them for binder design tasks
    if not is_inference:
        T += InferQueryPNUnitIIDsForDimers()

    ###########################################################
    ### Step 4: Build design task                           ###
    ###########################################################
    # Design task sampling: Sample and apply design tasks
    #
    # This section implements a flexible design task sampling system where different protein design
    # objectives can be conditionally applied based on the input data. The system works as follows:
    # 1. SampleDesignTask evaluates all available design tasks and samples one based on their
    #    frequencies among eligible tasks (those that can be applied to the current data)
    # 2. Each design task implements can_apply() to determine applicability and
    #    annotate_for_task_selection() to add required annotations
    # 3. The sampled task is then applied via ConditionalRoute to generate the appropriate
    #    design problem configuration
    # 4. Possible `DesignTaskModifier` functions can then be applied afterwards to modify the task
    #    These exist to reduce the overhead of writing full-fledged new design tasks by simply allowing
    #    to change the condition annotation values (e.g. `rasa`, `sequence`, `coordinates`) for a given,
    #    existing task.
    #
    # If no design tasks are provided (or empty dict), skip design task routing entirely.
    if not is_inference and design_tasks:
        T += SampleDesignTask(design_tasks=design_tasks)

        # ... create the design task
        T += ConditionalRoute(
            condition_func=_get_design_task_name,
            transform_map={name: task["transform"] for name, task in design_tasks.items()},
        )

        # ... (optionally) modify the design task post-hoc
        # TODO: Template this

        if crop_size or crop_size_range:
            T += EnforceCropBudget(crop_size=crop_size, plan_expansion=plan_unindex_expansion)

    return T.tolist()


# Coordinate-valued conditioning annotations always re-centered with the coords, so they stay in the
# model's frame
CONDITIONING_ANNOTATIONS_TO_CENTER = (C_CRD.full_name, C_CENTROID.full_name)


def _build_design_featurize_pipeline(
    residue_names_supporting_local_latents: list | set,
    metrics_tags: list[str] | set[str] | None = None,
    apply_na_ss: bool = True,
    is_inference: bool = False,
    precompute_atom_permutation: bool = False,
    # Bond feature dropout (training-only regularization)
    coordination_bond_dropout_p_all: float = 0.5,
    coordination_bond_dropout_p_individual: float = 0.5,
    aromatic_dropout_p: float = 0.4,
    path_length_search_cutoff: int | None = None,
    # Per-chain centroid conditioning (training-only; None disables)
    centroid_kwargs: dict | None = None,
    # Internal-coordinate-only conditioning ablation (training only)
    convert_absolute_to_internal: bool = False,
    # Loss weighting
    loss_weight_cfg: dict[str, float] | None = None,
) -> list[Transform]:
    """Build shared featurization pipeline across both training and validation.

    Args:
        residue_names_supporting_local_latents: Residue names that support local latents.
        metrics_tags: Tags to use for determining which Metrics apply to the outputs of this pipeline.

    Returns:
        List of transforms for featurization pipeline.
    """
    T = ListBuilder()

    # Add metrics tags if provided
    if metrics_tags is not None:
        T += AddData({"metrics_tags": metrics_tags})

    ###########################################################
    ### Step 1: Annotation fixes                            ###  # TODO: Should these be in validate / train portions?
    ###########################################################
    # Finalize sequence conditioning for both training tasks and inference inputs.
    T += FinalizeSequenceConditioning()
    T += CanonicalizeSequenceProbabilityConditions()

    # ... Annotate finalized token ids after we are sure we are not motifying `atomize` feature anymore
    T += AddGlobalTokenIdAnnotation(allow_overwrite=True)

    T += FeaturizeExpandedSegments()

    if not is_inference and convert_absolute_to_internal:
        T += ConvertAbsoluteToInternalCoords()

    # NA-SS annotation follows the same high-level pattern as other conditioning:
    # task-driven in train, specification-driven in inference.
    if apply_na_ss:
        T += FeaturizeNucleicAcidSecondaryStructure()

    # Create in-context conditioning corresponding to unindexed regions
    T += MaterializeInternalCoords()
    T += UnindexFlaggedTokens()
    # ... after the last transform that changes atom count or order, so indices stay valid
    if precompute_atom_permutation:
        T += PrecomputeAtomPermutation()

    # Prepare coordinates for noising (without modifying the ground truth)
    # ... add placeholder coordinates for noising
    T += CopyAnnotation(annotation_to_copy="coord", new_annotation="coord_to_be_noised")

    # Center coordinates and add centroid of the atoms without coordinate conditioning as a feature.
    T += CenterOnCoordinateConditionedAtoms(
        coord_annotation="coord_to_be_noised",
        conditioning_annotations=CONDITIONING_ANNOTATIONS_TO_CENTER,
    )
    # NOTE: Previously, we called this feature the "ori_token", which is a misnomer since it's no longer the origin
    T += FeaturizeCentroidOfAtomsWithoutCoordinateConditioning(
        coord_annotation="coord_to_be_noised",
        feature_name="centroid_of_atoms_without_coordinate_conditioning",
    )

    # Per-chain centroid conditioning channel (training-only opt-in)
    if not is_inference and centroid_kwargs is not None:
        T += FeaturizeChainCentroid(**centroid_kwargs)

    # Annotate and encode atom names using AF3 encoding
    T += AnnotateAtomNameVisibility()
    T += FeaturizeAtomNamesLikeAF3()

    # Per-token loss weighting
    # Each Upweight transform adds factor·is_k(atom_array) to ``loss_weight``
    for predicate, factor in build_loss_weight_upweights(loss_weight_cfg):
        T += UpweightLossWeightOnSubset(predicate, factor=factor)

    # Encode standard amino acids, RNA, and DNA nucleotides (atomised things are filtered out already)
    T += AddSupportsLocalLatentAnnotation(residue_names_supporting_local_latents=residue_names_supporting_local_latents)

    T += AddChiralityFeature()
    T += ConvertToAtomLevelProteinaFeatures()
    T += ConvertToAtom37Format()
    T += ComputeTokenLevelBondFeaturesLikeAF3(
        path_length_search_cutoff=path_length_search_cutoff,
        fill_res_id_gaps=is_inference,
    )
    if not is_inference:
        T += DropoutCoordinationBonds(
            p_all=coordination_bond_dropout_p_all, p_individual=coordination_bond_dropout_p_individual
        )
        T += DropoutAromaticFlag(p=aromatic_dropout_p)
    T += ConvertToTorch(keys=["feats"])

    # Subset to necessary keys only
    keys_to_keep = [
        "example_id",
        "feats",
        "atom_array",
        "extra_info",
        "task",
        "metrics_tags",
        "num_tokens",
        "specification",
    ]

    T += ValidateBatchInvariants(is_inference=is_inference)
    T += SubsetToKeys(keys_to_keep)

    return T.tolist()


def build_design_train_pipeline(
    crop_size: int | None = None,
    crop_size_range: tuple[int, int] | tuple[float, float] | None = None,
    max_crop_size: int | None = None,
    p_no_subcrop: float = 0.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    dna_contact_crop_probability: float = 0.0,
    crop_center_cutoff_distance: float = 15,
    max_atoms_in_crop: int | None = None,
    b_factor_occupancy_min: float | None = None,
    atomization_probabilities: dict[str, float] | None = None,
    p_random_rotation: float = 1.0,
    p_mirror: float = 0.0,
    design_tasks: dict[str, dict] | None = None,
    is_inference: bool = False,
    precompute_atom_permutation: bool = False,
    metrics_tags: list[str] | set[str] | None = None,
    residue_names_supporting_local_latents: list | set | None = None,
    # Nucleic acid secondary structure featurization
    apply_na_ss: bool = True,
    path_length_search_cutoff: int | None = None,
    # Per-chain centroid conditioning (training-only; None disables)
    centroid_kwargs: dict | None = None,
    # Internal-coordinate-only conditioning ablation (training only)
    convert_absolute_to_internal: bool = False,
    # Loss weighting
    loss_weight_cfg: dict[str, float] | None = None,
    sampling_transforms: list[Transform] | None = None,
    expandable_segment_rng: np.random.Generator | None = None,
) -> Transform:
    """Build training pipeline for protein design tasks.

    Supports both monomer and multi-chain (PPI) training.
    Pipeline: filter → augment → crop → design task → featurize

    Args:
        crop_size: Max residues after cropping. Either this or `crop_size_range`
            must be set (not both).
        crop_size_range: If set, the crop size is resampled per-example from this
            inclusive ``(min, max)`` range instead of using a fixed `crop_size`
            (which must then be `None`). Either a pair of absolute token counts,
            or a pair of fractions of the structure's token count (requires
            `max_crop_size`). See
            :py:func:`~atomworks.ml.transforms.crop.sample_dynamic_crop_size`.
            Defaults to `None`.
        max_crop_size: Upper bound on the sampled crop size. Required when
            `crop_size_range` is fraction-based. Defaults to `None`.
        crop_contiguous_probability: Probability of contiguous cropping
        crop_spatial_probability: Probability of spatial cropping
        dna_contact_crop_probability: Probability of protein-DNA contact-centered
            contiguous cropping (see
            :py:class:`~rfproteina.transforms.dna_crop.ProteinDNAContactContiguousCrop`).
            The three crop probabilities should sum to 1.0. Defaults to ``0.0``.
        crop_center_cutoff_distance: Distance cutoff (Å) for spatial crop
        atomization_probabilities: Per-residue-class atomization probabilities. Defaults to 0.5% for
            standard amino acids/RNA/DNA and 5% for vocabulary NCAAs/glycans.
        p_random_rotation: Probability of random rotation augmentation
        p_mirror: Probability of mirroring inputs (chirality augmentation).
            Defaults to ``0.0`` (disabled).
        design_tasks: Dictionary of design tasks where each task is a dict with
                     'frequency' and 'transform' keys. The 'transform' key holds
                     the instantiated DesignTask object.
                     If None, no design tasks are applied.
                     Example:
                         {
                             "unconditional": {
                                 "frequency": 2.0,
                                 "transform": UnconditionalTask(...)
                             }
                         }
        is_inference: If True, realizes expandable segments and skips augmentation.
        sampling_transforms: Optional transforms applied in order after source annotation and
            inference-only segment expansion, before cleanup and featurization.
        expandable_segment_rng: Optional local RNG for inference-only segment expansion.
        metrics_tags: Tags to use for determining which Metrics apply to the outputs of this pipeline.
    Returns:
        Compose of transforms for training pipeline
    """

    transforms = [AddSrcComponentAnnotation()]
    if is_inference:
        transforms.append(RealizeExpandableSegments(rng=expandable_segment_rng))
    transforms.extend(sampling_transforms or [])
    if residue_names_supporting_local_latents is None:
        residue_names_supporting_local_latents = LOCAL_LATENT_RESIDUES
    transforms += _build_design_clean_and_sample_pipeline(
        crop_size=crop_size,
        crop_size_range=crop_size_range,
        max_crop_size=max_crop_size,
        p_no_subcrop=p_no_subcrop,
        crop_contiguous_probability=crop_contiguous_probability,
        crop_spatial_probability=crop_spatial_probability,
        dna_contact_crop_probability=dna_contact_crop_probability,
        crop_center_cutoff_distance=crop_center_cutoff_distance,
        max_atoms_in_crop=max_atoms_in_crop,
        b_factor_occupancy_min=b_factor_occupancy_min,
        design_tasks=design_tasks,
        is_inference=is_inference,
        atomization_probabilities=atomization_probabilities,
        p_random_rotation=p_random_rotation,
        p_mirror=p_mirror,
    )
    transforms += _build_design_featurize_pipeline(
        metrics_tags=metrics_tags,
        apply_na_ss=apply_na_ss,
        residue_names_supporting_local_latents=residue_names_supporting_local_latents,
        is_inference=is_inference,
        precompute_atom_permutation=precompute_atom_permutation,
        path_length_search_cutoff=path_length_search_cutoff,
        centroid_kwargs=centroid_kwargs,
        convert_absolute_to_internal=convert_absolute_to_internal,
        loss_weight_cfg=loss_weight_cfg,
    )

    return Compose(transforms)


def build_design_validation_pipeline(
    metrics_tags: list[str] | set[str] | None = None,
    residue_names_supporting_local_latents: list | set | None = None,
    path_length_search_cutoff: int | None = None,
    # Loss weighting
    loss_weight_cfg: dict[str, float] | None = None,
    precompute_atom_permutation: bool = False,
    sampling_transforms: list[Transform] | None = None,
    expandable_segment_rng: np.random.Generator | None = None,
) -> Transform:
    """Build the validation/inference pipeline.

    Delegates to :py:func:`build_design_train_pipeline` with ``is_inference=True``,
    which gates out training-only transforms (augmentation, heavy filtering,
    random atomization, NaN-token removal, cropping, dimer query-pn-unit
    inference, and design-task sampling) and feature-dropout regularization.

    Args:
        metrics_tags: Tags to use for determining which Metrics apply to the outputs of this pipeline.
        residue_names_supporting_local_latents: Residue names that support local latents.
        path_length_search_cutoff: BFS depth limit for token bond-graph path lengths.
            ``None`` resolves all paths; connected paths above a finite cutoff use cutoff + 1.
            Generation inherits the checkpoint's training cutoff unless explicitly overridden.
        sampling_transforms: Optional transforms applied immediately after segment expansion,
            before cleanup and featurization. Omit when validating inputs without sampling.
        expandable_segment_rng: Optional local RNG for segment expansion during prevalidation.
    """
    return build_design_train_pipeline(
        is_inference=True,
        metrics_tags=metrics_tags,
        apply_na_ss=bool(metrics_tags is not None and "na_ss_conditioning" in set(metrics_tags)),
        residue_names_supporting_local_latents=residue_names_supporting_local_latents,
        path_length_search_cutoff=path_length_search_cutoff,
        loss_weight_cfg=loss_weight_cfg,
        precompute_atom_permutation=precompute_atom_permutation,
        sampling_transforms=sampling_transforms,
        expandable_segment_rng=expandable_segment_rng,
    )
