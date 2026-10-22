# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Secondary structure evaluation metrics for generated protein structures."""

import itertools
from typing import Any

import numpy as np
from tt_bio._vendor.atomworks.io.transforms.atom_array import ensure_atom_array_stack
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import as_atom_array_plus, stack_atom_array_plus
from tt_bio._vendor.atomworks.io.utils.selection import get_residue_starts
from tt_bio._vendor.atomworks.ml.transforms.dssp import SSEnum
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray, AtomArrayStack, annotate_sse, spread_residue_wise

from .conditions import C_SEC
class Metric:  # tt-bio: rfproteina.metrics.Metric is not vendored
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is an rfproteina evaluation metric; "
                                  "tt-bio vendors only the RFD4 prep path")
from .utils_chain import get_chain_of_interest


def annotate_secondary_structure(
    atom_array: AtomArray,
    annotation_name: str = "sse",
    is_valid_annotation_name: str = "sse_is_valid",
    inplace: bool = False,
) -> AtomArray:
    """Annotate secondary structure using P-SEA algorithm.

    P-SEA is residue-keyed (one code per residue, with ``''`` for non-AA /
    no-CA residues), so the annotation is computed per residue and spread to
    every atom of the residue. This keeps values uniform within a residue
    even when the residue is split across multiple tokens (e.g., partially
    atomized PTM residues whose sidechain atoms become atom-level tokens).

    Only protein residues are assigned secondary structure; all others are
    set to SSEnum.NONE.

    Args:
      atom_array: AtomArray to annotate.
      annotation_name: Name for the SSE annotation. Defaults to ``"sse"``.
      is_valid_annotation_name: Name for validity annotation. Defaults to ``"sse_is_valid"``.
      inplace: If True, annotate ``atom_array`` directly. Defaults to False.

    Returns:
      AtomArray with secondary structure annotations added. When ``inplace=True``
      this is the same object that was passed in.
    """
    atom_array = as_atom_array_plus(atom_array)
    if not inplace:
        atom_array = atom_array.copy()

    # Residue-level table
    residue_starts = get_residue_starts(atom_array)
    n_residues = len(residue_starts)

    # Default all residues to NONE
    sse = np.full(n_residues, SSEnum.NONE, dtype=np.int8)
    is_valid = np.zeros(n_residues, dtype=bool)

    try:
        # P-SEA returns one code per residue in residue order, with ``''`` for
        # non-AA / no-CA residues.
        psea_codes = annotate_sse(atom_array)
        if len(psea_codes) == n_residues:
            sse[psea_codes == "a"] = SSEnum.HELIX
            sse[psea_codes == "b"] = SSEnum.SHEET
            sse[psea_codes == "c"] = SSEnum.LOOP
            is_valid = psea_codes != ""
    except Exception:
        # P-SEA failed - return with all NONE values
        pass

    # Spread residue-wise to all atoms
    sse_spread = spread_residue_wise(atom_array, sse)
    is_valid_spread = spread_residue_wise(atom_array, is_valid)

    # Set annotations
    atom_array.set_annotation(annotation_name, sse_spread)
    atom_array.set_annotation(is_valid_annotation_name, is_valid_spread)

    return atom_array


def _sse_fractions(valid_sse: np.ndarray, suffix: str = "") -> dict[str, float]:
    """Compute SSE fraction and element-count metrics from valid token SSE codes.

    Args:
      valid_sse: Token-level SSEnum values for valid tokens only.
      suffix: Suffix appended to each metric key (e.g. ``"_coi"``). Defaults to ``""``.

    Returns:
      Dict with ``helix_fraction{suffix}``, ``sheet_fraction{suffix}``,
      ``loop_fraction{suffix}``, and ``num_ss_elements{suffix}`` keys. Values are
      NaN when ``valid_sse`` is empty.
    """
    if len(valid_sse) == 0:
        return {
            f"helix_fraction{suffix}": np.nan,
            f"sheet_fraction{suffix}": np.nan,
            f"loop_fraction{suffix}": np.nan,
            f"num_ss_elements{suffix}": np.nan,
        }

    n_valid = len(valid_sse)
    return {
        f"helix_fraction{suffix}": float((valid_sse == SSEnum.HELIX).sum() / n_valid),
        f"sheet_fraction{suffix}": float((valid_sse == SSEnum.SHEET).sum() / n_valid),
        f"loop_fraction{suffix}": float((valid_sse == SSEnum.LOOP).sum() / n_valid),
        f"num_ss_elements{suffix}": float(count_contiguous_sse_elements(valid_sse)),
    }


def compute_sse_distribution(atom_array: AtomArray) -> dict[str, float]:
    """Compute secondary structure distribution using P-SEA algorithm.

    Metrics are reported both over all valid tokens and restricted to the chain
    of interest (``_coi`` suffix), as determined by
    :py:func:`~rfproteina.utils.chain.get_chain_of_interest`.

    Args:
      atom_array: Protein structure to analyze.

    Returns:
      Dict with keys:
        - helix_fraction: Fraction of tokens in alpha helix
        - sheet_fraction: Fraction of tokens in beta sheet
        - loop_fraction: Fraction of tokens in coil/loop
        - num_ss_elements: Number of contiguous SSE regions
        - helix_fraction_coi / sheet_fraction_coi / loop_fraction_coi /
          num_ss_elements_coi: Same metrics restricted to the chain of interest.

      Returns NaN values if annotation fails or no valid annotations exist.
    """
    atom_array = as_atom_array_plus(atom_array)

    # Run P-SEA annotation
    try:
        atom_array = annotate_secondary_structure(
            atom_array, annotation_name="sse", is_valid_annotation_name="sse_is_valid"
        )
    except Exception:
        # Annotation failed - return NaN
        return {**_sse_fractions(np.array([]), ""), **_sse_fractions(np.array([]), "_coi")}

    # Get token-level annotations
    token_starts = get_token_starts(atom_array)
    token_sse = atom_array.sse[token_starts]
    token_is_valid = atom_array.sse_is_valid[token_starts]

    # Overall metrics across all valid tokens
    valid_sse = token_sse[token_is_valid]

    # Chain-of-interest metrics
    chain_of_interest, _ = get_chain_of_interest(atom_array)
    if "chain_iid" in atom_array.get_annotation_categories():
        token_chain = atom_array.get_annotation("chain_iid")[token_starts]
    else:
        token_chain = atom_array.chain_id[token_starts]
    coi_valid_mask = token_is_valid & (token_chain == chain_of_interest)
    valid_sse_coi = token_sse[coi_valid_mask]

    return {**_sse_fractions(valid_sse, ""), **_sse_fractions(valid_sse_coi, "_coi")}


def count_contiguous_sse_elements(sse_array: np.ndarray) -> int:
    """Count number of contiguous secondary structure elements.

    Args:
      sse_array: Array of SSEnum values (token-level).

    Returns:
      Number of contiguous SSE regions.

    Examples:
      >>> sse = np.array([1, 1, 2, 2, 3, 3, 1])  # helix, sheet, loop, helix
      >>> count_contiguous_sse_elements(sse)
      4
    """
    if len(sse_array) == 0:
        return 0

    # Count transitions using groupby
    # Each group represents a contiguous SSE element
    groups = [(k, len(list(g))) for k, g in itertools.groupby(sse_array)]
    return len(groups)


def compute_sse_accuracy(atom_array: AtomArray) -> dict[str, float]:
    """Compute secondary structure conditioning accuracy.

    Compares C_SEC annotation (conditioning) against P-SEA-annotated SSE on the
    same structure. Measures how well the generated structure matches its own
    secondary structure conditioning annotations.

    Args:
      atom_array: Generated structure with C_SEC annotations to validate.

    Returns:
      Dict with keys:
        - overall_accuracy: Fraction of correct predictions
        - helix_recall: Recall for helix conditioning
        - sheet_recall: Recall for sheet conditioning
        - loop_recall: Recall for loop conditioning
        - n_conditioned_atoms: Number of atoms with C_SEC conditioning

      Returns empty dict if no C_SEC annotation present.
    """
    atom_array = as_atom_array_plus(atom_array)

    # Check if C_SEC annotation exists
    if C_SEC.full_name not in atom_array.get_annotation_categories():
        return {}

    c_sec_annotation = C_SEC.annotation(atom_array)

    # Check if there are any conditioned atoms
    conditioned_mask = c_sec_annotation != SSEnum.NONE
    if not conditioned_mask.any():
        return {}

    # Run P-SEA on the structure
    try:
        atom_array = annotate_secondary_structure(
            atom_array,
            annotation_name="sse",
            is_valid_annotation_name="sse_is_valid",
        )
    except Exception:
        # Annotation failed
        return {
            "overall_accuracy": np.nan,
            "helix_recall": np.nan,
            "sheet_recall": np.nan,
            "loop_recall": np.nan,
            "n_conditioned_atoms": int(conditioned_mask.sum()),
        }

    # Filter to atoms that are:
    # 1. Conditioned (C_SEC != NONE)
    # 2. Have valid annotations
    valid_comparison_mask = conditioned_mask & atom_array.sse_is_valid

    if not valid_comparison_mask.any():
        # No valid comparisons possible
        return {
            "overall_accuracy": np.nan,
            "helix_recall": np.nan,
            "sheet_recall": np.nan,
            "loop_recall": np.nan,
            "n_conditioned_atoms": int(conditioned_mask.sum()),
        }

    # Extract conditioned and predicted SSE for valid atoms
    c_sec_conditioned = c_sec_annotation[valid_comparison_mask]
    sse_predicted = atom_array.sse[valid_comparison_mask]

    # Overall accuracy
    overall_accuracy = (c_sec_conditioned == sse_predicted).mean()

    # Per-class recall
    def compute_recall(sse_class: int) -> float:
        """Compute recall for a specific SSE class."""
        conditioned_as_class = c_sec_conditioned == sse_class
        if not conditioned_as_class.any():
            return np.nan
        return (sse_predicted[conditioned_as_class] == sse_class).mean()

    helix_recall = compute_recall(SSEnum.HELIX)
    sheet_recall = compute_recall(SSEnum.SHEET)
    loop_recall = compute_recall(SSEnum.LOOP)

    return {
        "overall_accuracy": float(overall_accuracy),
        "helix_recall": float(helix_recall),
        "sheet_recall": float(sheet_recall),
        "loop_recall": float(loop_recall),
        "n_conditioned_atoms": int(valid_comparison_mask.sum()),
    }


class SecondaryStructureDistribution(Metric):
    """Computes secondary structure distribution of generated protein structures.

    Uses P-SEA to annotate generated structures and reports the fraction of
    residues in helix, sheet, and loop conformations at the token level.
    Only considers tokens with valid annotations.

    Args:
      atom_array_input_key: Name of the input kwarg containing atom arrays.
        Use ``"generated_atom_array"`` for design contexts,
        ``"predicted_atom_array"`` for structure prediction contexts.
        Defaults to ``"generated_atom_array"``.
    """

    def __init__(self, atom_array_input_key: str = "predicted_atom_array", **kwargs):
        """Initialize SSE distribution metric."""
        self.atom_array_input_key = atom_array_input_key
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {"atom_array": self.atom_array_input_key}

    def compute(self, atom_array: AtomArray | AtomArrayStack | list) -> dict[str, float]:
        """Compute SSE distribution for generated structure(s).

        Args:
          atom_array: AtomArray, AtomArrayStack, or list of AtomArrays to analyze.

        Returns:
          Dict with SSE distribution values:
            - helix_fraction: Fraction of tokens in helix
            - sheet_fraction: Fraction of tokens in sheet
            - loop_fraction: Fraction of tokens in loop
            - num_ss_elements: Number of contiguous SSE regions
            - helix_fraction_coi / sheet_fraction_coi / loop_fraction_coi /
              num_ss_elements_coi: Same metrics restricted to the chain of interest.

          For AtomArrayStack inputs, returns nanmean across all structures.
        """
        if isinstance(atom_array, list):
            atom_array = stack_atom_array_plus(atom_array)

        atom_array_stack = ensure_atom_array_stack(atom_array)
        results = [compute_sse_distribution(arr) for arr in atom_array_stack]
        keys = [
            "helix_fraction",
            "sheet_fraction",
            "loop_fraction",
            "num_ss_elements",
            "helix_fraction_coi",
            "sheet_fraction_coi",
            "loop_fraction_coi",
            "num_ss_elements_coi",
        ]
        return {key: float(np.nanmean([r[key] for r in results])) for key in keys}


class SecondaryStructureAccuracy(Metric):
    """Computes accuracy of secondary structure conditioning vs generated structures.

    Compares C_SEC conditioning annotation (carried through from generation) against
    P-SEA-annotated secondary structure of the same generated structures. Validates
    that the generated structure matches its own conditioning annotations.

    Args:
      atom_array_input_key: Name of the input kwarg containing atom arrays.

    """

    def __init__(self, atom_array_input_key: str = "predicted_atom_array", **kwargs):
        """Initialize SSE accuracy metric.

        Args:
          atom_array_input_key: Input kwarg name for atom array.
          **kwargs: Additional arguments passed to parent Metric class.
        """
        self.atom_array_input_key = atom_array_input_key
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {"atom_array": self.atom_array_input_key}

    def compute(self, atom_array: AtomArray | AtomArrayStack) -> dict[str, float]:
        """Compute SSE conditioning accuracy.

        Args:
          atom_array: AtomArray, AtomArrayStack, or list of AtomArrays with C_SEC annotations.

        Returns:
          Dict with accuracy values (only if C_SEC present):
            - overall_accuracy: Fraction of correct predictions
            - helix_recall: Recall for helix conditioning
            - sheet_recall: Recall for sheet conditioning
            - loop_recall: Recall for loop conditioning
            - n_conditioned_atoms: Number of conditioned atoms

          For AtomArrayStack inputs, returns nanmean across all structures.
          Returns empty dict if no C_SEC annotations found.
        """
        if isinstance(atom_array, list):
            atom_array = stack_atom_array_plus(atom_array)

        atom_array_stack = ensure_atom_array_stack(atom_array)
        results = [compute_sse_accuracy(arr) for arr in atom_array_stack]

        # Filter out empty results (no C_SEC annotation)
        non_empty_results = [r for r in results if r]
        if not non_empty_results:
            return {}

        return {
            "overall_accuracy": float(np.nanmean([r["overall_accuracy"] for r in non_empty_results])),
            "helix_recall": float(np.nanmean([r["helix_recall"] for r in non_empty_results])),
            "sheet_recall": float(np.nanmean([r["sheet_recall"] for r in non_empty_results])),
            "loop_recall": float(np.nanmean([r["loop_recall"] for r in non_empty_results])),
            "n_conditioned_atoms": int(np.nanmean([r["n_conditioned_atoms"] for r in non_empty_results])),
        }
