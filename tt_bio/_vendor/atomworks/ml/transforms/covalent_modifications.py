"""Transforms to handle covalent modifications"""

from typing import ClassVar

from biotite.structure import AtomArray

from tt_bio._vendor.atomworks.io.transforms.atomize import flag_and_reassign_covalent_modifications
from tt_bio._vendor.atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from tt_bio._vendor.atomworks.ml.transforms.atomize import AtomizeByCCDName
from tt_bio._vendor.atomworks.ml.transforms.base import Transform


class AnnotateCovalentModifications(Transform):
    """Handles covalent modifications within the AtomArray.

    Independently controls PN-unit reassignment and atomization of polymer residues
    bonded to non-polymers. Modifier atomization is left to downstream transforms.
    The modifier's PN unit is always annotated as a covalent modification.
    """

    incompatible_previous_transforms: ClassVar[list[str | Transform]] = [AtomizeByCCDName, "AddGlobalTokenIdAnnotation"]

    def __init__(
        self,
        reassign_modified_residue: bool = True,
        atomize_modified_residue: bool = True,
    ) -> None:
        """Initialize the transform.

        Args:
          reassign_modified_residue: Move the modified polymer residue into the modifier's PN unit.
          atomize_modified_residue: Force atomization of the modified polymer residue, not the modifier.
        """
        super().__init__()
        self.reassign_modified_residue = reassign_modified_residue
        self.atomize_modified_residue = atomize_modified_residue

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_id", "pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = flag_and_reassign_covalent_modifications(
            data["atom_array"],
            reassign_modified_residue=self.reassign_modified_residue,
            atomize_modified_residue=self.atomize_modified_residue,
        )
        return data


# Backwards-compatible alias for the pre-rename class name.
FlagAndReassignCovalentModifications = AnnotateCovalentModifications
