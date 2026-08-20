"""
Condition system with:
 - an abstract base class (`ConditionBase`) for sharing general functionality across conditions
 - a metaclass that serves as a registry for all concrete conditions (`ConditionMeta`) that
    automatically registers all concrete conditions and sets their attributes
 - a dynamic accessor (`ConditionAccessor`) that provides easy access to all registered conditions
"""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from tt_bio._vendor.atomworks._compat import StrEnum
from typing import Any, ClassVar, Literal

import numpy as np
from biotite.structure import AtomArray

from tt_bio._vendor.atomworks.io.utils import scatter
from tt_bio._vendor.atomworks.io.utils.atom_array_plus import AnnotationList2D
from tt_bio._vendor.atomworks.io.utils.selection import get_annotation, get_residue_starts
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts

__all__ = ["CONDITIONS", "ConditionBase"]


class Level(StrEnum):
    """
    Level-hierarchy for describing information in a molecular structure.

    Used for example to specify the level at which a `Condition` applies.
    """

    ATOM = "atom"
    TOKEN = "token"
    RESIDUE = "residue"
    CHAIN = "chain"
    MOLECULE = "molecule"
    SYSTEM = "system"

    def _get_segment_or_group(self, atom_array: "AtomArray") -> tuple[str, np.ndarray]:
        if self == Level.ATOM:
            raise ValueError("Apply and spread does not make sense for atom level.")
        if self == Level.TOKEN:
            return "segment", get_token_starts(atom_array, add_exclusive_stop=True)
        elif self == Level.RESIDUE:
            return "segment", get_residue_starts(atom_array, add_exclusive_stop=True)
        elif self == Level.CHAIN:
            chain_key = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
            return "group", atom_array.get_annotation(chain_key)
        elif self == Level.MOLECULE:
            return "group", atom_array.get_annotation("molecule_id")
        elif self == Level.SYSTEM:
            return "group", np.ones(atom_array.array_length(), dtype=np.int32)
        else:
            raise ValueError(f"Invalid level: {self}")

    def apply(self, atom_array: "AtomArray", data: np.ndarray, func: "Callable[[AtomArray], Any]") -> "Any":
        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.apply_segment_wise(grouping, data, func)
        elif strategy == "group":
            return scatter.apply_group_wise(grouping, data, func)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def spread(self, atom_array: "AtomArray", data: np.ndarray) -> np.ndarray:
        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.spread_segment_wise(grouping, data)
        elif strategy == "group":
            return scatter.spread_group_wise(grouping, data)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def apply_and_spread(
        self,
        atom_array: "AtomArray",
        data: np.ndarray,
        func: "Callable[[np.ndarray], Any]",
    ) -> np.ndarray:
        """
        Apply a function to the data and spread the result back to the original positions.
        """

        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.apply_and_spread_segment_wise(grouping, data, func)
        elif strategy == "group":
            return scatter.apply_and_spread_group_wise(grouping, data, func)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")


class ConditionMeta(ABCMeta):
    """
    Metaclass that combines ABC functionality with auto-registration.

    It registers all concrete (non-abstract) subclasses into a central
    registry and pre-computes derived class attributes like 'full_name'.
    """

    _registry: ClassVar[dict[str, type["ConditionBase"]]] = {}

    def __new__(meta, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs):  # noqa: N804
        # Create the class as usual
        cls = super().__new__(meta, name, bases, namespace, **kwargs)

        # Register the class if it's a concrete implementation
        # We check for 'name' to ensure it's a condition intended for registration.
        condition_name = namespace.get("name")
        if condition_name:
            # Ensure the `name` is valid
            if "_" in condition_name:
                raise ValueError(
                    f"Condition `{condition_name}` (class `{cls.__name__}`) must not contain underscores in its name."
                    " This is reserved for internal use."
                )

            # Ensure the following attributes are set
            if "n_body" not in namespace:
                raise ValueError(
                    f"Condition `{condition_name}` (class `{cls.__name__}`) must have an n_body attribute."
                )
            if "level" not in namespace:
                raise ValueError(f"Condition `{condition_name}` (class `{cls.__name__}`) must have a level attribute.")
            if "is_symmetric" not in namespace and (cls.n_body > 1):
                raise ValueError(
                    f"Condition `{condition_name}` (class `{cls.__name__}`) must have an is_symmetric attribute if n_body > 1."
                )
            else:
                cls.is_symmetric = True

            if condition_name in meta._registry:
                raise ValueError(f"Condition with name '{condition_name}' is already registered.")

            # Ensure the level is a Level enum
            cls.level = Level(cls.level)

            # --- Eagerly compute and set properties on the class itself ---
            if "is_mask" in namespace:
                cls.is_mask = namespace["is_mask"]
            else:
                cls.is_mask = np.issubdtype(cls.dtype, bool)

            cls.full_name = cls.get_full_name()
            cls.mask_name = cls.get_mask_name()

            # --- Register the class ---
            meta._registry[condition_name] = cls

        return cls

    def __repr__(self) -> str:  # noqa: N804
        try:
            if self.n_body > 1:
                return f"{self.__name__}(name={self.name}, n_body={self.n_body}, level={self.level}, is_symmetric={self.is_symmetric})"
            else:
                return f"{self.__name__}(name={self.name}, n_body={self.n_body}, level={self.level})"
        except AttributeError:
            # NOTE: This is a fallback for the abstract base class to ensure that methods like `__mro__` continue to work
            return super().__repr__()


class ConditionBase(ABC, metaclass=ConditionMeta):
    """Abstract base class for all conditions. Any condition must be a subclass of this class.

    Attributes:
        (required)
        name: The name of the condition.
        n_body: The number of bodies involved in the condition.
        level: The level at which the condition applies.

        (optional)
        is_symmetric: Whether the condition is symmetric. Only
            applies if `n_body > 1`. Otherwise always `True`.
        is_mask: Whether the condition itself is a mask. In this case the
            `full_name` is the same as the `mask_name`.

        (derived)
        full_name: The full, systematic name of the condition of the form
            `condition_{name}_{n_body}_{level}`.
        mask_name: The mask, systematic name of the condition of the form
            `mask_{name}_{n_body}_{level}`.
    """

    # --- Attributes to be defined by all concrete subclasses ---
    name: ClassVar[str]
    n_body: ClassVar[int]
    level: ClassVar[Level]
    dtype: ClassVar[np.dtype | float | int | str | bool]
    is_symmetric: ClassVar[bool]  # only needs to be set if n_body > 1

    # --- Pre-computed attributes (set by metaclass) ---
    full_name: ClassVar[str]
    mask_name: ClassVar[str]
    is_mask: ClassVar[bool]

    def __init__(self):
        """DO NOT INSTANTIATE THIS CLASS. Use its class methods directly. Init will throw a RuntimeError."""
        # Enforce that these classes are not for instantiation
        raise RuntimeError(
            f"{self.__class__.__name__} is a definitional class and should not be instantiated. "
            "Use its class methods directly."
        )

    @classmethod
    def _resolve_level(cls, level: Level | str | None) -> Level:
        """Resolves the level to a Level enum."""
        return Level(level if level is not None else cls.level)

    @classmethod
    def _resolve_n_body(cls, n_body: int | None) -> int:
        """Resolves the number of bodies to an integer."""
        return n_body if n_body is not None else cls.n_body

    # --- name generation methods ---
    @classmethod
    def get_full_name(cls) -> str:
        """Returns the full name of the condition at a given body order and level."""
        if cls.is_mask:
            # ... re-route to the mask name generation for masks
            return cls.get_mask_name()

        return f"condition_{cls.name}_{cls.n_body}_{cls.level}"

    @classmethod
    def get_mask_name(
        cls,
        n_body: int | None = None,
        level: Level | str | None = None,
    ) -> str:
        """Returns the mask name of the condition at a given body order and level.

        By default (i.e. if `n_body` and `level` are not provided), n_body and level
            are resolved to the default values for the condition.

        Args:
            n_body: The number of bodies involved in the condition.
            level: The level at which the condition applies.

        Returns:
            The mask name of the condition.

        Example:
            For a 2-body distance condition which applies at the `ATOM` level, the
            default mask is a 2-body atom-level mask `mask_distance_2_atom`. In the future,
            we may implement automatic derivation of lower-body masks from higher-body masks.
            I.e. the 1-body token-level mask `mask_distance_1_token` could be derived from the default
        """
        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        return f"mask_{cls.name}_{n_body}_{level}"

    @classmethod
    def get_feature_name(
        cls,
        n_body: int | None = None,
        level: Level | str | None = None,
        suffix: str = "",
    ) -> str:
        """Returns the feature name of the condition at a given body order and level.

        Args:
            n_body: The body order to get the feature name for.
            level: The level to get the feature name for.
            suffix: An optional suffix to add to the feature name to allow
             flexibility for adding multiple different features for a single condition.

        Returns:
            The feature name of the condition. This will be of the form:
            `feature-<suffix>_<condition_name>_<n_body>_<level>`
        """
        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        suffix_str = f"-{suffix}" if suffix and not suffix.startswith("-") else suffix
        return f"feature{suffix_str}_{cls.name}_{n_body}_{level}"

    # --- Abstract Default Generation & Validation Methods ---
    @classmethod
    @abstractmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        """Generates the default mask for the condition."""
        raise NotImplementedError(f"Condition `{cls.name}` (class `{cls.__name__}`) does not have a default mask.")

    @classmethod
    @abstractmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.is_mask:
            return cls.default_mask(atom_array)

        raise NotImplementedError(
            f"Condition `{cls.name}` (class `{cls.__name__}`) does not have a default annotation."
        )

    # --- Core Functionality ---
    @classmethod
    def mask(
        cls,
        atom_array: AtomArray,
        n_body: int | None = None,
        level: Level | str | None = None,
        *,
        default: Any | Literal["generate", "raise"] = "generate",
    ) -> np.ndarray:
        """Gets a mask from an AtomArray, falling back to a generated default.

        Args:
            atom_array: The AtomArray to get the mask from.
            n_body: The number of bodies involved in the condition.
            level: The level at which the condition applies.
            default: The default value to return if the mask is not found.
                If `"generate"`, the default mask is generated and returned.
                If `"raise"`, a ValueError is raised if the mask is not found.
                If any other value, that value is returned.

        Returns:
            The mask for the condition.

        Raises:
            ValueError: If the mask is not found and `default` is `"raise"`.
        """
        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        mask_name = cls.get_mask_name(n_body, level)

        mask = get_annotation(atom_array, mask_name, n_body=n_body)

        if mask is None:
            # TODO: Enable default generation for levels & n_body other than the default
            default = "raise" if (level != cls.level) | (n_body != cls.n_body) else default

            if default == "generate":
                mask = cls.default_mask(atom_array)
            elif default == "raise":
                raise ValueError(f"AtomArray is missing {n_body}-body annotation `{mask_name}`.")
            else:
                mask = default

        return mask

    @classmethod
    def annotation(
        cls,
        atom_array: AtomArray,
        default: Any | Literal["generate", "raise"] = "generate",
    ) -> np.ndarray:
        """Gets an annotation from an AtomArray, falling back to a generated default.

        Args:
            atom_array: The AtomArray to get the annotation from.
            default: The default value to return if the annotation is not found.
                If `"generate"`, the default annotation is generated and returned.
                If `"raise"`, a ValueError is raised if the annotation is not found.
                If any other value, that value is returned.

        Returns:
            The annotation for the condition.

        Raises:
            ValueError: If the annotation is not found and `default` is `"raise"`.
        """
        annotation = get_annotation(atom_array, cls.full_name, n_body=cls.n_body)

        if annotation is None:
            if default == "generate":
                annotation = cls.default_annotation(atom_array)
            elif default == "raise":
                raise ValueError(f"AtomArray is missing {cls.n_body}-body annotation `{cls.full_name}`.")
            else:
                annotation = default

        return annotation

    @classmethod
    def set_mask(
        cls,
        atom_array: AtomArray,
        mask: np.ndarray | AnnotationList2D,
        n_body: int | None = None,
        level: Level | None = None,
    ) -> None:
        """Sets the mask for the condition on the AtomArray."""
        level, n_body = cls._resolve_level(level), cls._resolve_n_body(n_body)
        mask_name = cls.get_mask_name(n_body, level)
        if n_body == 1:
            if not np.issubdtype(mask.dtype, bool):
                raise ValueError(f"Mask `{mask_name}` for `{cls.name}` must be a boolean array. Got {mask.dtype}.")
            atom_array.set_annotation(mask_name, mask)
        elif n_body == 2:
            if cls.is_symmetric:
                mask = mask.symmetrized()
            n_atoms = mask.n_atoms
            pairs = mask.pairs
            values = mask.values
            if n_atoms != atom_array.array_length():
                raise ValueError(
                    f"Mask `{mask_name}` for `{cls.name}` must have {atom_array.array_length()} atoms. Got {n_atoms}."
                )
            if len(pairs) and ((pairs.max() >= n_atoms) or (pairs.min() < 0)):
                raise ValueError(
                    f"Mask `{mask_name}` for `{cls.name}` must have pairs within the range [0, {n_atoms}). Got min={min(pairs)}, max={max(pairs)}."
                )
            if not np.issubdtype(pairs.dtype, np.integer):
                raise ValueError(f"Mask `{mask_name}` for `{cls.name}` must have pairs as integers. Got {pairs.dtype}.")
            if not np.issubdtype(values.dtype, np.bool_):
                raise ValueError(f"Mask `{mask_name}` for `{cls.name}` must be a boolean array. Got {mask.dtype}.")
            atom_array.set_annotation_2d(mask_name, pairs, values)
        else:
            raise NotImplementedError(
                f"Currently only 1-body and 2-body conditions are supported. Got {n_body}-body condition."
            )

    @classmethod
    def set_annotation(cls, atom_array: AtomArray, *args, **annotation_kwargs) -> None:
        """Sets the annotation for the condition on the AtomArray. Dynamically dispatches to the correct body order."""
        if cls.n_body == 1:
            # ... handle correct argument logic
            if len(args) == 0:
                array = annotation_kwargs.pop("array")  # biotite .set_annotation expects a single array argument
            elif len(args) == 1:
                array = args[0]
            else:
                raise ValueError(f"Only one argument is allowed for 1-body conditions. Got {len(args)} arguments.")
            assert len(annotation_kwargs) == 0, "No keyword arguments are allowed for 1-body conditions."

            # ... set the annotation
            atom_array.set_annotation(cls.full_name, array)

        elif cls.n_body == 2:
            if not hasattr(atom_array, "_annot_2d"):
                raise ValueError("AtomArray must have 2D annotations to set 2D annotation.")

            # ... handle correct argument logic
            if len(args) == 1:
                assert isinstance(args[0], AnnotationList2D), "Only AnnotationList2D is allowed for 2-body conditions."
                annot = args[0]
                pairs, values = annot.pairs, annot.values
                annotation_kwargs = {"pairs": pairs, "values": values}
            elif len(args) == 0:
                pass
            else:
                raise ValueError(f"Only one argument is allowed for 2-body conditions. Got {len(args)} arguments.")

            # ... set the annotation
            atom_array.set_annotation_2d(cls.full_name, **annotation_kwargs)
        else:
            raise NotImplementedError("Currently only 1-body and 2-body conditions are supported.")

    @classmethod
    def is_valid(cls, atom_array: AtomArray) -> bool:
        """
        Check if the condition's annotation (and mask, if not is_mask) is consistent at the relevant level.
        (i.e. do all atoms/tokens/residues/chains/molecules/systems have the same value for the condition?)

        Returns:
            bool: True if valid, False otherwise.
        """
        if cls.level == Level.ATOM:
            return True

        if cls.n_body == 1:
            is_same = lambda x: np.all(x == x[0]) if len(x) > 0 else True  # noqa: E731
            is_annotation_valid = cls.level.apply(atom_array, cls.annotation(atom_array), is_same)
            if not cls.is_mask:
                is_mask_valid = cls.level.apply(atom_array, cls.mask(atom_array), is_same)
                is_annotation_valid &= is_mask_valid
            return np.all(is_annotation_valid)

        # TODO: Implement n-body aggregation
        return True


class ConditionAccessor:
    """
    Provides dynamic, attribute-based access to all registered conditions.
    No manual updates needed when new conditions are added.
    """

    def __getattr__(self, name: str) -> type[ConditionBase]:
        """Dynamically retrieves a condition class from the registry."""
        # replace any underscores in the name with hyphens
        name = name.replace("_", "-")
        return self.__getitem__(name)

    def __getitem__(self, name: str) -> type[ConditionBase]:
        """Retrieves a condition by its string name."""
        try:
            return ConditionMeta._registry[name]
        except KeyError:
            raise AttributeError(
                f"No condition named '{name}' is registered. Available conditions: {self.list()}"
            ) from None

    def list(self) -> list[str]:
        """Returns a list of names of all registered conditions."""
        return list(ConditionMeta._registry.keys())

    def __iter__(self):
        """Iterate over all registered condition classes."""
        for condition_name in self.list():
            yield self[condition_name]

    def get_valid_full_names(self) -> set[str]:
        """Returns the set of systematic full names of all registered conditions."""
        return frozenset(self[condition].full_name for condition in self.list())

    def get_valid_mask_names(self) -> set[str]:
        """Returns the set of systematic mask names of all registered conditions."""
        return frozenset(self[condition].mask_name for condition in self.list())

    def get(self, name: str) -> type[ConditionBase]:
        """Retrieves a condition by its string name."""
        return self.__getitem__(name)

    def from_mask_name(self, mask_name: str) -> type[ConditionBase]:
        """Resolves a systematic mask name to a condition class."""
        condition_name = mask_name.split("_")[1]
        return self.__getitem__(condition_name)

    def from_full_name(self, full_name: str) -> type[ConditionBase]:
        """Resolves a systematic full name to a condition class."""
        condition_name = full_name.split("_")[1]
        return self.__getitem__(condition_name)

    def __repr__(self) -> str:
        return f"Conditions({self.list()})"


# Singleton instance for easy, clean access
CONDITIONS = ConditionAccessor()
