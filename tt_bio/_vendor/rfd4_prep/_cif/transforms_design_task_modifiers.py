# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
from tt_bio._vendor.atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_POLYMER_RESIDUES, STANDARD_RNA
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus
from tt_bio._vendor.atomworks.io.utils.selection import get_residue_starts
from tt_bio._vendor.atomworks.ml.conditions import (
    C_CRD,
    C_SEQ,
)
from tt_bio._vendor.atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from biotite.structure import apply_residue_wise, get_residue_count, spread_residue_wise
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .conditions import (
    C_CATH,
    C_CRDGRP,
    C_HB,
    C_HPHOB,
    C_MCOORD,
    C_PICATION,
    C_PISTACK,
    C_RAS,
    C_SALT,
    C_SEC,
    SEQUENCE_PROBABILITY_CONDITION_BY_TOKEN,
    SEQUENCE_PROBABILITY_CONDITIONS,
    ConditionBase,
    HBondEnum,
    MetalCoordEnum,
    PiCationEnum,
    RasaEnum,
    SaltBridgeEnum,
)
from .metrics_sse_metrics import annotate_secondary_structure
from .transforms_utils_conditioning_utils import sample_island_tokens
from .utils_common import true_with_prob
from .utils_sequence_sampling import UNIFORM_DNA_PRIOR, UNIFORM_RNA_PRIOR, UNIPROT_AA_PRIOR


def add_rasa_condition(
    data: dict,
    p_add_rasa_condition: float = 0.0,
    probe_radius: float = 1.4,
    atom_radii: str | np.ndarray = "element",
    point_number: int = 100,
    p_include_non_ligand: float = 0.1,
    rasa_max: float = 0.1,
) -> dict:
    """Add binned RASA conditioning to an example.

    Atoms with nonfinite RASA remain unconditioned.

    Args:
        data: Example dictionary containing an atom array.
        p_add_rasa_condition: Probability of adding this condition at all. Defaults to 0.0.
        probe_radius: Van der Waals radius of the probe in Angstrom. Defaults to
            1.4 for water.
        atom_radii: Atom radii used for the calculation. Defaults to ``"element"``,
            matching training and supporting both protein and ligand atoms.
        point_number: Number of points sampled by the Shrake-Rupley algorithm.
            Defaults to 100.
        p_include_non_ligand: Probability of conditioning non-ligand atoms.
            Defaults to 0.1.
        rasa_max: Inclusive upper boundary of the intermediate bin. Defaults to 0.1.
            The buried/intermediate boundary is 0.02.

    Returns:
        The example dictionary with RASA annotations added to its atom array.
    """
    if not true_with_prob(p_add_rasa_condition):
        return data

    atom_array = as_atom_array_plus(data["atom_array"])

    ensure_annotations(atom_array, "is_ligand", "atomize")

    # Determine valid residues for rasa conditioning
    if np.random.rand() < p_include_non_ligand:
        seq_mask = C_SEQ.mask(atom_array, default="generate")
        crd_mask = C_CRD.mask(atom_array, default="generate") | C_CRDGRP.is_internal(atom_array)
        inclusion_mask = seq_mask | crd_mask
    else:
        inclusion_mask = atom_array.is_ligand | atom_array.atomize

    rasa = calculate_atomwise_rasa(atom_array, probe_radius, atom_radii, point_number, sasa_mask=inclusion_mask)

    inclusion_mask = inclusion_mask & np.isfinite(rasa)

    # Round to the SASA sampling grid; the upper threshold belongs to intermediate.
    rasa = np.rint(rasa.astype(float) * point_number) / point_number
    bin_edges = [0.0, 0.02, np.nextafter(rasa_max, np.inf)]
    bins = np.zeros(len(rasa), dtype=int)
    bins[inclusion_mask] = np.digitize(rasa[inclusion_mask], bin_edges, right=False)
    if (bins[inclusion_mask] == 0).any():
        raise ValueError("Negative RASA value was detected. This indicates an error.")

    # Convert to RasaEnum
    enum_rasa_values = C_RAS.default_annotation(atom_array)
    enum_rasa_values[bins == 1] = RasaEnum.BURIED
    enum_rasa_values[bins == 2] = RasaEnum.INTERMEDIATE
    enum_rasa_values[bins == 3] = RasaEnum.EXPOSED
    enum_rasa_values[~inclusion_mask] = RasaEnum.UNSPECIFIED

    C_RAS.set_annotation(atom_array, enum_rasa_values.astype(int))

    data["atom_array"] = atom_array
    data.setdefault("task", {})["added_rasa_condition"] = True
    return data


def subsample_condition_enum(
    condition_values: np.ndarray,
    fraction: float,
    none_value: Any = 0,
    groups: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Randomly clear a subset of the active condition values, keeping ``fraction`` of them.

    Args:
      condition_values: Per-atom condition values (boolean mask or integer enum).
      fraction: Fraction of active (non-``none_value``) entries to keep. Must be in ``(0, 1]``.
      none_value: Value denoting "no condition". Defaults to ``0``.
      groups: Index groups kept or cleared as a unit, e.g. the atoms of one aromatic ring. Groups
        may overlap, so an atom shared by a kept and a cleared ring stays; any active entry outside
        every group is its own group. ``fraction`` then counts groups rather than atoms.

    Returns:
      A copy of ``condition_values`` with ``1 - fraction`` of its active entries reset to
      ``none_value``. At least one active entry is always kept.
    """
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be in the range (0, 1].")

    condition_values = condition_values.copy()
    grouped = np.zeros(len(condition_values), dtype=bool)
    for group in groups:
        grouped[group] = True
    ungrouped_active = np.where((condition_values != none_value) & ~grouped)[0]
    units = [*groups, *(np.array([atom]) for atom in ungrouped_active)]

    keep_count = max(1, int(len(units) * fraction))
    order = np.arange(len(units))
    np.random.shuffle(order)

    kept = np.zeros(len(condition_values), dtype=bool)
    for unit in order[:keep_count]:
        kept[units[unit]] = True
    condition_values[~kept] = none_value

    return condition_values


def _carries(condition_values: np.ndarray, base_value: Any, unions: dict[Any, tuple[Any, ...]]) -> np.ndarray:
    """Atoms whose value is ``base_value``, or a union that includes it."""
    return np.isin(condition_values, [base_value, *(u for u, bases in unions.items() if base_value in bases)])


def _whole_ring_groups(
    atom_array: Any,
    condition_values: np.ndarray,
    none_value: Any,
    ring_level_values: tuple[Any, ...],
    unions: dict[Any, tuple[Any, ...]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group the atoms of each annotated aromatic ring, so a ring is only ever conditioned whole.

    Rings are the minimal cycles of the bond graph induced by the annotated atoms alone. We sidestep
    aromaticity detection since any rings annotated with these pi-system-based interactions must be
    aromatic.

    Args:
      condition_values: Per-atom condition values to group.
      none_value: Value denoting "no condition".
      ring_level_values: The subset of those values that lives on a ring; see
        :py:func:`_subset_ground_truth_interaction_condition`.
      unions: Maps each value that combines several base values to those bases, e.g.
        ``PiCationEnum.BOTH``.

    Returns:
      A copy of ``condition_values`` with any ring-level value outside a whole ring cleared, and one
      index array per ring. Groups overlap where rings are fused.
    """
    if not ring_level_values:
        return condition_values, []
    ring_atoms = np.logical_or.reduce(
        [_carries(condition_values, ring_level_value, unions) for ring_level_value in ring_level_values]
    )
    if not ring_atoms.any():
        return condition_values, []

    bonds = np.empty((0, 2), dtype=int) if atom_array.bonds is None else atom_array.bonds.as_array()[:, :2]
    bonds = bonds[ring_atoms[bonds].all(axis=1)]

    groups = []
    if len(bonds):
        atoms = np.unique(bonds)
        edges = np.searchsorted(atoms, bonds)
        adjacency = coo_matrix(
            (np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])), shape=(len(atoms), len(atoms))
        )
        _, component_of_atom = connected_components(adjacency, directed=False)
        degree = np.bincount(edges.ravel(), minlength=len(atoms))
        for component in range(component_of_atom.max() + 1):
            members = component_of_atom == component
            if (degree[members] == 2).all():
                # A connected component whose atoms all have two bonds is a single ring already.
                groups.append(atoms[members])
                continue
            # Fused rings, so pay for a minimum cycle basis to split the system into its rings.
            component_edges = edges[members[edges[:, 0]]].tolist()
            groups += [
                atoms[np.fromiter(cycle, dtype=int)]
                for cycle in nx.minimum_cycle_basis(nx.from_edgelist(component_edges))
            ]

    grouped = np.zeros(len(condition_values), dtype=bool)
    for group in groups:
        grouped[group] = True

    # A ring missing some of its atoms leaves no cycle to find; drop its remnant rather than train
    # on a fragment of a ring.
    condition_values = condition_values.copy()
    condition_values[ring_atoms & ~grouped] = none_value

    return condition_values, groups


def _drop_annotations_without_cross_chain_partner(
    atom_array: Any,
    ground_truth: np.ndarray,
    stored: np.ndarray,
    none_value: Any,
    partners: dict[Any, Any],
    unions: dict[Any, tuple[Any, ...]],
    partner_cutoff: float,
    ring_level_values: tuple[Any, ...] = (),
    rings: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Drop two-body annotations whose interaction partner did not survive cropping.

    Args:
      ground_truth: Motif-masked values being revealed.
      stored: Full stored annotation on the crop, searched for partners.
      partners: Maps each base value to the value a partner atom must carry.
      unions: Describes enum values that combine multiple base values, e.g. ``HBondEnum.BOTH``.
      partner_cutoff: Distance cutoff (A) of the detector that produced the annotation.
      ring_level_values: See :py:func:`_subset_ground_truth_interaction_condition`.
      rings: The annotated rings, from :py:func:`_whole_ring_groups`.
    """
    chains = np.asarray(atom_array.chain_iid)
    coords = atom_array.coord

    # Which atoms have a cross-chain atom carrying the complementary value within the cutoff.
    supported = {}
    for base_value, partner_value in partners.items():
        query = np.where(_carries(ground_truth, base_value, unions))[0]
        candidates = _carries(stored, partner_value, unions)
        mask = np.zeros(len(ground_truth), dtype=bool)
        for chain in np.unique(chains[query]):
            rows = query[chains[query] == chain]
            cross_chain = candidates & (chains != chain)
            if not cross_chain.any():
                continue
            tree = cKDTree(coords[cross_chain])
            mask[rows] = tree.query(coords[rows], k=1)[0] <= partner_cutoff
        if base_value in ring_level_values:
            # A ring interacts as a unit, so one supported atom carries the whole ring. Fused rings
            # share atoms, so support is unioned over rings rather than partitioned between them.
            for ring in rings:
                if mask[ring].any():
                    mask[ring] = True
        supported[base_value] = mask

    kept = np.full_like(ground_truth, none_value)
    for base_value, mask in supported.items():
        kept[mask] = base_value
    # A union only stays a union while every value it combines still has a partner; otherwise the
    # loop above has already degraded it to the single value that does.
    for union, bases in unions.items():
        kept[np.logical_and.reduce([supported[base] for base in bases])] = union
    return kept


def _subset_ground_truth_interaction_condition(
    data: dict,
    condition: type[ConditionBase],
    none_value: Any,
    task_key: str,
    p_add_condition: float,
    p_subsample: float,
    subsample_frac: float,
    partners: dict[Any, Any],
    unions: dict[Any, tuple[Any, ...]],
    partner_cutoff: float,
    ring_level_values: tuple[Any, ...] = (),
) -> dict:
    """Subset a ground-truth interaction Condition already present on the AtomArray.

    The AtomArray carries the full, un-gated ground-truth interaction annotation. This either
    reveals a motif-masked, optionally subsampled view of it, or clears it back to its default so
    that the ground truth does not leak through when this modifier is not "applied".

    NOTE: Only meaningful on datasets loaded with ``load_standard_annotations: true``; elsewhere the
    annotation is absent and the "revealed" condition is empty.

    Args:
      data: Data dict containing atom_array.
      condition: The interaction ``ConditionBase`` subclass to subset (e.g. ``C_HB``).
      none_value: The "no interaction" value for this condition's dtype.
      task_key: Key set to ``True`` in ``data["task"]`` when the condition is applied.
      p_add_condition: Probability of revealing this condition at all.
      p_subsample: Probability of additionally subsampling the revealed condition.
      subsample_frac: Fraction of active entries to keep when subsampling.
      partners: See :py:func:`_drop_annotations_without_cross_chain_partner`.
      unions: See :py:func:`_drop_annotations_without_cross_chain_partner`.
      partner_cutoff: Distance cutoff (A) of the detector that produced the annotation.
      ring_level_values: Values that belong to an aromatic ring rather than to its individual
        atoms, such as pi-stacking and the pi side of pi-cation. Their rings are dropped
        and subsampled whole, since conditioning on part of a ring would be unphysical.

    Returns:
      Updated data dict with ``condition`` set on the AtomArray.
    """
    atom_array = as_atom_array_plus(data["atom_array"])

    ground_truth = None
    rings: list[np.ndarray] = []
    if np.random.rand() < p_add_condition:
        seq_mask = C_SEQ.mask(atom_array, default="generate")
        crd_mask = C_CRD.mask(atom_array, default="generate")
        intcrd_mask = C_CRDGRP.is_internal(atom_array)
        motif_mask = seq_mask | crd_mask | intcrd_mask

        stored = condition.annotation(atom_array, default="generate")
        ground_truth = stored.copy()
        ground_truth[~motif_mask] = none_value
        ground_truth, rings = _whole_ring_groups(atom_array, ground_truth, none_value, ring_level_values, unions)

        # The annotation predates cropping, so drop entries whose partner the crop removed.
        if np.any(ground_truth != none_value):
            ground_truth = _drop_annotations_without_cross_chain_partner(
                atom_array,
                ground_truth,
                stored,
                none_value,
                partners,
                unions,
                partner_cutoff,
                ring_level_values,
                rings,
            )

        # Nothing left to reveal (single-chain input, or the crop took every partner): fall
        # through to the "not applied" path rather than flagging an empty condition.
        if not np.any(ground_truth != none_value):
            ground_truth = None

    if ground_truth is None:
        condition.set_annotation(atom_array, condition.default_annotation(atom_array))
        data["atom_array"] = atom_array
        return data

    if np.random.rand() < p_subsample:
        # The partner might have dropped some rings, so we check for this explicitly
        ground_truth = subsample_condition_enum(
            ground_truth,
            subsample_frac,
            none_value=none_value,
            groups=[ring for ring in rings if (ground_truth[ring] != none_value).all()],
        )

    condition.set_annotation(atom_array, ground_truth)
    data.setdefault("task", {})[task_key] = True
    data["atom_array"] = atom_array
    return data


def add_hbond_condition(
    data: dict,
    p_add_hbond_condition: float = 0.0,
    p_hbond_subsample: float = 0.5,
    hbond_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth hydrogen-bond conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_HB,
        HBondEnum.NONE,
        "added_hbond_condition",
        p_add_hbond_condition,
        p_hbond_subsample,
        hbond_subsample_frac,
        partners={HBondEnum.DONOR: HBondEnum.ACCEPTOR, HBondEnum.ACCEPTOR: HBondEnum.DONOR},
        unions={HBondEnum.BOTH: (HBondEnum.DONOR, HBondEnum.ACCEPTOR)},
        partner_cutoff=4.1,
    )


def add_pistack_condition(
    data: dict,
    p_add_pistack_condition: float = 0.0,
    p_pistack_subsample: float = 0.5,
    pistack_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth pi-stacking conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_PISTACK,
        False,
        "added_pistack_condition",
        p_add_pistack_condition,
        p_pistack_subsample,
        pistack_subsample_frac,
        partners={True: True},
        unions={},
        # 5.5 A ring centre-to-centre maximum, plus a ring radius at each end.
        partner_cutoff=8.5,
        # Both sides are rings: keep or drop each one whole.
        ring_level_values=(True,),
    )


def add_pication_condition(
    data: dict,
    p_add_pication_condition: float = 0.0,
    p_pication_subsample: float = 0.5,
    pication_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth pi-cation conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_PICATION,
        PiCationEnum.NONE,
        "added_pication_condition",
        p_add_pication_condition,
        p_pication_subsample,
        pication_subsample_frac,
        partners={PiCationEnum.AROMATIC: PiCationEnum.CATION, PiCationEnum.CATION: PiCationEnum.AROMATIC},
        unions={PiCationEnum.BOTH: (PiCationEnum.AROMATIC, PiCationEnum.CATION)},
        # 6.0 A cation-to-ring-centre maximum, plus a ring radius.
        partner_cutoff=7.5,
        # The aromatic side is a ring: keep or drop it whole. The cation stays per-atom.
        ring_level_values=(PiCationEnum.AROMATIC,),
    )


def add_salt_condition(
    data: dict,
    p_add_salt_condition: float = 0.0,
    p_salt_subsample: float = 0.5,
    salt_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth salt-bridge conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_SALT,
        SaltBridgeEnum.NONE,
        "added_salt_condition",
        p_add_salt_condition,
        p_salt_subsample,
        salt_subsample_frac,
        partners={SaltBridgeEnum.POSITIVE: SaltBridgeEnum.NEGATIVE, SaltBridgeEnum.NEGATIVE: SaltBridgeEnum.POSITIVE},
        unions={},
        partner_cutoff=5.5,
    )


def add_hydrophobic_condition(
    data: dict,
    p_add_hydrophobic_condition: float = 0.0,
    p_hydrophobic_subsample: float = 0.5,
    hydrophobic_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth hydrophobic-contact conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_HPHOB,
        False,
        "added_hydrophobic_condition",
        p_add_hydrophobic_condition,
        p_hydrophobic_subsample,
        hydrophobic_subsample_frac,
        partners={True: True},
        unions={},
        partner_cutoff=4.0,
    )


def add_metal_condition(
    data: dict,
    p_add_metal_condition: float = 0.0,
    p_metal_subsample: float = 0.5,
    metal_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth metal-coordination conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_MCOORD,
        MetalCoordEnum.NONE,
        "added_metal_condition",
        p_add_metal_condition,
        p_metal_subsample,
        metal_subsample_frac,
        partners={MetalCoordEnum.METAL: MetalCoordEnum.COORDINATING, MetalCoordEnum.COORDINATING: MetalCoordEnum.METAL},
        unions={},
        partner_cutoff=3.0,
    )


def add_secondary_structure_condition(
    data: dict[str, Any],
    p_add_secondary_structure_condition: float = 0.0,
    max_secondary_structure_frac_to_provide: float = 0.4,
    min_ss_island_len: int = 1,
    max_ss_island_len: int = 10,
    n_islands_max: int = 3,
) -> dict[str, Any]:
    """Add secondary structure conditioning.

    Uses P-SEA to annotate secondary structure, samples islands from residues
    with valid SSE annotations, and sets C_SEC condition. Sampling is done at
    residue granularity (not token granularity) so that residues spanning
    multiple tokens — e.g., atomized PTM residues — receive a uniform value, satisfying the
    ``Level.RESIDUE`` invariant on the secondary-structure condition.

    Args:
      data: Data dict containing atom_array.
      p_add_secondary_structure_condition: Probability of adding this condition at all. Defaults to 0.0.
      max_secondary_structure_frac_to_provide: Max fraction of valid SS residues
        to condition. Defaults to 0.4.
      min_ss_island_len: Minimum island length (in residues). Defaults to 1.
      max_ss_island_len: Maximum island length (in residues). Defaults to 10.
      n_islands_max: Maximum number of islands. Defaults to 3.

    Returns:
      Updated data dict with C_SEC conditioning.
    """
    if not true_with_prob(p_add_secondary_structure_condition):
        return data

    atom_array = as_atom_array_plus(data["atom_array"])

    # Step 1: Annotate with P-SEA (residue-uniform sse / sse_is_valid)
    atom_array = annotate_secondary_structure(
        atom_array, annotation_name="sse", is_valid_annotation_name="sse_is_valid"
    )

    # Step 2: Sample islands from eligible residues
    residue_starts = get_residue_starts(atom_array)
    residue_level_array = atom_array[residue_starts]

    # Eligible: residues with valid SSE annotations
    eligible_for_ss_info_mask = residue_level_array.sse_is_valid

    n_residues_with_valid_ss = eligible_for_ss_info_mask.sum()
    max_residues_with_ss_conditioning = int(np.ceil(n_residues_with_valid_ss * max_secondary_structure_frac_to_provide))

    where_to_show_ss = sample_island_tokens(
        n_residues_with_valid_ss,
        island_len_min=min_ss_island_len,
        island_len_max=max_ss_island_len,
        n_islands_min=1,
        n_islands_max=n_islands_max,
        max_length=max_residues_with_ss_conditioning,
    )

    # Convert to residue-level mask and spread to atoms
    residue_level_ss_mask = np.zeros(len(residue_starts), dtype=bool)
    residue_level_ss_mask[eligible_for_ss_info_mask] = where_to_show_ss
    ss_mask = spread_residue_wise(atom_array, residue_level_ss_mask)

    # Step 3: Set C_SEC annotation (SSEnum.NONE for unspecified, sse values where sampled)
    ss_annotation = np.full(atom_array.array_length(), 0, dtype=int)  # 0 is SSEnum.NONE
    ss_annotation[ss_mask] = atom_array.sse[ss_mask]
    C_SEC.set_annotation(atom_array, ss_annotation)

    data["atom_array"] = atom_array
    data.setdefault("task", {})["added_secondary_structure_condition"] = True
    return data


def _sample_truncated_power_law(alpha: float, maximum: int) -> int:
    """Sample from ``P(N=k) ∝ k^-alpha`` for integer counts from one through ``maximum``."""
    counts = np.arange(1, maximum + 1)
    probabilities = counts.astype(float) ** -alpha
    return int(np.random.choice(counts, p=probabilities / probabilities.sum()))


def diversify_sequence_conditioning(
    data: dict[str, Any],
    p_exact: float = 0.5,
    p_zero_only: float = 1 / 6,
    p_positive_only: float = 1 / 6,
    p_zero_and_positive: float = 1 / 6,
    dirichlet_concentration: float = 3.0,
    exclusion_count_alpha: float = 2.0,
    p_full_probability_vector: float = 0.75,
    p_augment_unconditioned: float = 0.0,
    min_augmentation_fraction: float = 0.05,
    max_augmentation_fraction: float = 0.2,
    p_zero_only_whole_chain: float = 0.0,
) -> dict[str, Any]:
    """Sample calibrated sequence conditions on residues whose identity is not otherwise revealed.

    Existing exact conditions are diversified, while an independently sampled subset of
    unconditioned residues can receive additional conditioning. Atomized residues and residues
    with conditioned non-backbone coordinates are never modified. A zero-only draw may cover a
    whole chain by broadcasting identities absent from that chain.

    Args:
      data: Example dictionary containing an atom array.
      p_exact: Relative weight for retaining ordinary exact sequence conditioning.
      p_zero_only: Relative weight for exposing only exact-zero exclusions.
      p_positive_only: Relative weight for exposing positive probabilities without exact zeros.
      p_zero_and_positive: Relative weight for exposing both exclusions and positive probabilities.
      dirichlet_concentration: Concentration of the restricted prior before truth size-biasing.
      exclusion_count_alpha: Exponent of the truncated power-law distributions for exclusion and
        exposed-positive counts. Counts range from one through the maximum valid for the mode.
      p_full_probability_vector: Probability of exposing every positive probability; otherwise
        expose a nonempty proper subset.
      p_augment_unconditioned: Per-example probability of adding conditions to otherwise
        unconditioned eligible residues.
      min_augmentation_fraction: Minimum fraction of eligible unconditioned residues selected.
      max_augmentation_fraction: Maximum fraction of eligible unconditioned residues selected.
      p_zero_only_whole_chain: Conditional probability that a zero-only draw uses one shared set
        of exclusions across a whole chain.
    """
    mode_probabilities = np.asarray([p_exact, p_zero_only, p_positive_only, p_zero_and_positive], dtype=float)
    if (mode_probabilities < 0).any() or mode_probabilities.sum() <= 0:
        raise ValueError("Sequence-probability mode weights must be nonnegative with positive total weight.")
    mode_probabilities /= mode_probabilities.sum()
    if dirichlet_concentration <= 0:
        raise ValueError(f"dirichlet_concentration must be positive, got {dirichlet_concentration}.")
    if exclusion_count_alpha <= 0:
        raise ValueError(f"exclusion_count_alpha must be positive, got {exclusion_count_alpha}.")
    if not 0 <= p_full_probability_vector <= 1:
        raise ValueError(f"p_full_probability_vector must lie in [0, 1], got {p_full_probability_vector}.")
    if not 0 <= p_augment_unconditioned <= 1:
        raise ValueError(f"p_augment_unconditioned must lie in [0, 1], got {p_augment_unconditioned}.")
    if not 0 <= min_augmentation_fraction <= max_augmentation_fraction <= 1:
        raise ValueError(
            "Augmentation fractions must satisfy "
            f"0 <= min <= max <= 1, got {min_augmentation_fraction} and {max_augmentation_fraction}."
        )
    if not 0 <= p_zero_only_whole_chain <= 1:
        raise ValueError(f"p_zero_only_whole_chain must lie in [0, 1], got {p_zero_only_whole_chain}.")
    if mode_probabilities[0] == 1.0 and p_augment_unconditioned == 0.0:
        return data
    chain_event_probability = mode_probabilities[1] * p_zero_only_whole_chain

    atom_array = as_atom_array_plus(data["atom_array"])
    seq_mask = C_SEQ.mask(atom_array, default="generate")

    ensure_annotations(
        atom_array,
        "atomize",
        "is_protein_backbone",
        "is_rna_backbone",
        "is_dna_backbone",
    )
    residue_boundaries = get_residue_starts(atom_array, add_exclusive_stop=True)
    residue_starts = residue_boundaries[:-1]
    backbone = atom_array.is_protein_backbone | atom_array.is_rna_backbone | atom_array.is_dna_backbone
    coord_conditioned = C_CRD.mask(atom_array, default="generate") | C_CRDGRP.is_internal(atom_array)
    has_conditioned_nonbackbone = apply_residue_wise(atom_array, coord_conditioned & ~backbone, np.any)
    must_keep_exact = apply_residue_wise(atom_array, atom_array.atomize, np.any) | has_conditioned_nonbackbone
    has_probability_condition = apply_residue_wise(
        atom_array,
        np.logical_or.reduce(
            [condition.mask(atom_array, default="generate") for condition in SEQUENCE_PROBABILITY_CONDITIONS]
        ),
        np.any,
    )
    residue_names = atom_array.res_name[residue_starts]
    exact_condition = apply_residue_wise(atom_array, seq_mask, np.any)
    locally_eligible = np.isin(residue_names, STANDARD_POLYMER_RESIDUES) & ~must_keep_exact & ~has_probability_condition
    locally_augmented = np.zeros(len(residue_starts), dtype=bool)
    augmentable = np.flatnonzero(locally_eligible & ~exact_condition)
    if len(augmentable) and true_with_prob(p_augment_unconditioned):
        augmentation_fraction = np.random.uniform(min_augmentation_fraction, max_augmentation_fraction)
        n_to_augment = min(len(augmentable), int(np.ceil(augmentation_fraction * len(augmentable))))
        if n_to_augment:
            locally_augmented[np.random.choice(augmentable, size=n_to_augment, replace=False)] = True
    locally_selected = locally_eligible & (exact_condition | locally_augmented)

    modes = np.full(len(residue_starts), -1, dtype=np.int8)
    whole_chain_selected = np.zeros(len(residue_starts), dtype=bool)
    condition_values: dict[type[ConditionBase], np.ndarray] = {}
    polymer_specs = (
        (STANDARD_AA, UNIPROT_AA_PRIOR),
        (STANDARD_RNA, UNIFORM_RNA_PRIOR),
        (STANDARD_DNA, UNIFORM_DNA_PRIOR),
    )
    chain_ids = atom_array.chain_iid if "chain_iid" in atom_array.get_annotation_categories() else atom_array.chain_id
    residue_chain_iids = chain_ids[residue_starts]

    # Reserve part of the zero-only probability for chain-wide exclusions. If the chain-wide
    # branch is not taken, renormalizing the local probabilities preserves the requested marginal
    # mode frequencies for every candidate residue.
    local_mode_weights = mode_probabilities.copy()
    local_mode_weights[1] *= 1 - p_zero_only_whole_chain
    if chain_event_probability < 1:
        local_mode_weights /= 1 - chain_event_probability

    for tokens, _ in polymer_specs:
        family_mask = np.isin(residue_names, tokens)
        family_atom_mask = np.isin(atom_array.res_name, tokens)
        for chain_iid in np.unique(residue_chain_iids[family_mask]):
            chain_residue_mask = family_mask & (residue_chain_iids == chain_iid)
            chain_atom_mask = family_atom_mask & (chain_ids == chain_iid)
            protected_identities = set(residue_names[chain_residue_mask]) | {
                condition.token_name
                for condition in SEQUENCE_PROBABILITY_CONDITIONS
                if condition.token_name in tokens
                and np.any(
                    chain_atom_mask
                    & condition.mask(atom_array, default="generate")
                    & (condition.annotation(atom_array, default="generate") != 0)
                )
            }
            absent_identities = np.flatnonzero(~np.isin(tokens, list(protected_identities)))
            whole_chain_is_possible = len(absent_identities) > 0
            use_whole_chain_zeros = (
                whole_chain_is_possible and chain_event_probability > 0 and true_with_prob(chain_event_probability)
            )

            if use_whole_chain_zeros:
                exclusion_count = _sample_truncated_power_law(exclusion_count_alpha, len(absent_identities))
                excluded = np.random.choice(absent_identities, size=exclusion_count, replace=False)
                for token_index in excluded:
                    condition = SEQUENCE_PROBABILITY_CONDITION_BY_TOKEN[tokens[token_index]]
                    if condition not in condition_values:
                        condition_values[condition] = condition.annotation(atom_array, default="generate").copy()
                    condition_values[condition][chain_atom_mask] = 0.0
                whole_chain_selected |= chain_residue_mask
                locally_selected[chain_residue_mask] = False
            else:
                probabilities = local_mode_weights if whole_chain_is_possible else mode_probabilities
                chain_local_selection = locally_selected & chain_residue_mask
                modes[chain_local_selection] = np.random.choice(4, size=chain_local_selection.sum(), p=probabilities)

    if not whole_chain_selected.any() and not locally_selected.any():
        return data

    updated_seq_mask = seq_mask.copy()
    updated_seq_mask[spread_residue_wise(atom_array, whole_chain_selected & locally_eligible)] = False
    for residue_index in np.flatnonzero(modes >= 0):
        mode = modes[residue_index]
        residue_slice = slice(residue_boundaries[residue_index], residue_boundaries[residue_index + 1])
        if mode == 0:
            updated_seq_mask[residue_slice] = True
            continue

        updated_seq_mask[residue_slice] = False

        truth = str(residue_names[residue_index])
        tokens, prior = next((tokens, prior) for tokens, prior in polymer_specs if truth in tokens)
        truth_index = tokens.index(truth)
        excluded = np.empty(0, dtype=int)
        if mode in (1, 3):
            off_targets = np.delete(np.arange(len(tokens)), truth_index)
            # Mixed soft conditioning must retain at least two allowed identities so that it
            # cannot degenerate into an exact probability-one condition. Negative-only may
            # exclude every off-target identity.
            max_excluded = len(tokens) - (2 if mode == 3 else 1)
            exclusion_count = _sample_truncated_power_law(exclusion_count_alpha, max_excluded)
            excluded = np.random.choice(off_targets, size=exclusion_count, replace=False)

        allowed = np.setdiff1d(np.arange(len(tokens)), excluded)
        probabilities = np.zeros(len(tokens), dtype=np.float32)
        if mode in (2, 3):
            restricted_prior = prior[allowed] / prior[allowed].sum()
            parameters = dirichlet_concentration * restricted_prior
            parameters[np.flatnonzero(allowed == truth_index)[0]] += 1.0
            sampled = np.random.dirichlet(parameters)
            # Keep soft samples strictly inside (0, 1) after float32 storage.
            sampled = np.maximum(sampled, np.finfo(np.float32).eps)
            sampled /= sampled.sum()
            probabilities[allowed] = sampled.astype(np.float32)
            if true_with_prob(p_full_probability_vector):
                exposed_positive = allowed
            else:
                positive_count = _sample_truncated_power_law(exclusion_count_alpha, len(allowed) - 1)
                exposed_positive = np.random.choice(allowed, size=positive_count, replace=False)
        else:
            exposed_positive = np.empty(0, dtype=int)

        for token_index in np.concatenate([excluded, exposed_positive]):
            condition = SEQUENCE_PROBABILITY_CONDITION_BY_TOKEN[tokens[token_index]]
            if condition not in condition_values:
                condition_values[condition] = condition.annotation(atom_array, default="generate").copy()
            condition_values[condition][residue_slice] = probabilities[token_index]

    for condition, values in condition_values.items():
        condition.set_annotation(atom_array, values)

    sequence = C_SEQ.annotation(atom_array, default="generate").copy()
    newly_exact = updated_seq_mask & ~seq_mask
    sequence[newly_exact] = C_SEQ.ground_truth_value(atom_array)[newly_exact]
    sequence[~updated_seq_mask] = C_SEQ.default_annotation(atom_array)[~updated_seq_mask]
    C_SEQ.set_annotation(atom_array, sequence)

    data["atom_array"] = atom_array
    task = data.setdefault("task", {})
    selected = locally_selected | (whole_chain_selected & locally_eligible)
    task["sequence_conditioning_replaced_count"] = int((selected & exact_condition).sum())
    task["sequence_conditioning_augmented_count"] = int((selected & locally_augmented).sum())
    task["sequence_probability_exact_count"] = int((modes == 0).sum())
    task["sequence_probability_zero_only_count"] = int((modes == 1).sum())
    task["sequence_probability_positive_only_count"] = int((modes == 2).sum())
    task["sequence_probability_zero_and_positive_count"] = int((modes == 3).sum())
    task["sequence_probability_whole_chain_zero_count"] = int(whole_chain_selected.sum())
    return data


def _fraction_of_chain_retained(data: dict[str, Any], chain_iid: str) -> float:
    """Fraction of a chain's residues that survived cropping; 1.0 if no crop was applied."""
    pre_crop = data.get("crop_info", {}).get("atom_array", None)
    if pre_crop is None:
        return 1.0

    pre_crop_chain = pre_crop.chain_iid == chain_iid
    if not pre_crop_chain.any():
        return 1.0

    atom_array = data["atom_array"]
    chain = atom_array.chain_iid == chain_iid
    n_post = get_residue_count(atom_array[chain]) if chain.any() else 0
    return n_post / get_residue_count(pre_crop[pre_crop_chain])


def add_cath_condition(
    data: dict[str, Any],
    p_add_cath_c_condition: float = 0.0,
    p_add_cath_ca_condition: float = 0.0,
    p_add_cath_cat_condition: float = 0.0,
    min_retained_chain_fraction: float = 0.75,
) -> dict[str, Any]:
    """Independently sample each chain's training CATH conditioning from stored labels.

    Unselected labels are cleared, including on skipped or ineligible examples. Stored
    labels retain their domain boundaries. Eligible chains can receive different levels
    in the same example. Inference does not use this modifier.

    Args:
      data: Data dict containing atom_array with optional per-residue CATH annotations.
      p_add_cath_c_condition: Per-chain probability of revealing the class only.
      p_add_cath_ca_condition: Per-chain probability of revealing class and architecture.
      p_add_cath_cat_condition: Per-chain probability of revealing class, architecture and topology.
      min_retained_chain_fraction: Minimum fraction of the labelled chain retained by cropping.
    """
    atom_array = data["atom_array"]
    stored = C_CATH.annotation(atom_array, default="generate").copy()
    stored = np.where(np.isin(stored, ["?", ".", ""]), C_CATH.NULL_VALUE, stored)
    cath_annot = C_CATH.default_annotation(atom_array)
    C_CATH.set_annotation(atom_array, cath_annot)
    task = data.setdefault("task", {})
    for level in ("c", "ca", "cat"):
        task[f"added_cath_{level}_condition"] = False

    p_skip = 1 - (p_add_cath_c_condition + p_add_cath_ca_condition + p_add_cath_cat_condition)
    if p_skip >= 1:
        return data

    for chain_iid in np.unique(atom_array.chain_iid):
        chain_mask = atom_array.chain_iid == chain_iid
        if not np.any(stored[chain_mask] != C_CATH.NULL_VALUE):
            continue
        if _fraction_of_chain_retained(data, chain_iid) < min_retained_chain_fraction:
            continue

        sampled_value = np.random.rand()
        if sampled_value < p_skip:
            continue
        if sampled_value < p_skip + p_add_cath_c_condition:
            n_levels = 1
        elif sampled_value < p_skip + p_add_cath_c_condition + p_add_cath_ca_condition:
            n_levels = 2
        else:
            n_levels = 3

        for cath_code in np.unique(stored[chain_mask]):
            if cath_code == C_CATH.NULL_VALUE:
                continue
            code_parts = cath_code.split(".")
            if len(code_parts) != 4:
                raise ValueError(f"Incomplete or invalid CATH code: {cath_code}")
            code_parts[n_levels:] = ["x"] * (4 - n_levels)
            cath_annot[chain_mask & (stored == cath_code)] = ".".join(code_parts)
            revealed_depth = sum(part != "x" for part in code_parts)
            if revealed_depth:
                level = ("c", "ca", "cat")[revealed_depth - 1]
                task[f"added_cath_{level}_condition"] = True

    C_CATH.set_annotation(atom_array, cath_annot)
    return data
