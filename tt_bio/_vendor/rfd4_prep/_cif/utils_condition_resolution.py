# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Resolve a Condition/StandardAnnotation class from an arbitrary user-facing name.

Backs the ``conditions:`` block of a condition spec
(``docs/how-to/input_prep/yaml-json.md``), where
an annotation may be named by any shorthand alias (``C_CRD``, ``SA_SRC``,
...) or by its full systematic registry name (``coordinate``, ``hotspot``,
...).
"""

import pickle
from enum import IntEnum
from functools import cache

import numpy as np
from tt_bio._vendor.atomworks.io.utils.standard_annotations.base import STANDARD_ANNOTATIONS, StandardAnnotationBase
from tt_bio._vendor.atomworks.ml import conditions as _atomworks_conditions
from biotite.structure import AtomArray

from . import conditions as _rfproteina_conditions
from .conditions import C_CATH
from .datapipes_pipeline import CATH_MAPPING_PATH
from . import utils_standard_annotations as _rfproteina_standard_annotations

# Modules scanned for shorthand aliases (module-level `C_*`/`SA_*` constants), in lookup
# order. Most `C_*` aliases (e.g. `C_CRD`, `C_HOT`, `C_CENTROID`) actually live in
# `atomworks.ml.conditions`, not `rfproteina.conditions` -- both must be scanned.
_ALIAS_MODULES = (_atomworks_conditions, _rfproteina_conditions, _rfproteina_standard_annotations)


def _registered_classes() -> frozenset[type[StandardAnnotationBase]]:
    """Every concretely-registered Condition/StandardAnnotation class (never abstract bases)."""
    return frozenset(STANDARD_ANNOTATIONS)


def _alias_lookup(name: str) -> type[StandardAnnotationBase] | None:
    registered = _registered_classes()
    for module in _ALIAS_MODULES:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type) and candidate in registered:
            return candidate
    return None


def _all_alias_names() -> list[str]:
    registered = _registered_classes()
    names: list[str] = []
    for module in _ALIAS_MODULES:
        for attr_name, candidate in vars(module).items():
            if isinstance(candidate, type) and candidate in registered:
                names.append(attr_name)
    return names


def resolve_standard_annotation(name: str) -> type[StandardAnnotationBase]:
    """Resolve ``name`` to a registered Condition/StandardAnnotation class.

    Tries, in order: (1) a shorthand alias (a module-level ``C_*``/``SA_*``
    constant) defined in ``atomworks.ml.conditions``, ``rfproteina.conditions``,
    or ``rfproteina.utils.standard_annotations``; (2) the class's own systematic
    registry name (e.g. ``"hotspot"``, ``"rasa-binned"``).

    Args:
      name: Shorthand alias or systematic registry name.

    Returns:
      The resolved class.

    Raises:
      ValueError: If ``name`` matches neither, listing every valid option.
    """
    resolved = _alias_lookup(name)
    if resolved is not None:
        return resolved

    for annotation_cls in STANDARD_ANNOTATIONS:
        if name in (annotation_cls.name, annotation_cls.full_name, *annotation_cls.aliases):
            return annotation_cls

    valid_aliases = sorted(set(_all_alias_names()))
    valid_systematic = sorted(STANDARD_ANNOTATIONS.list())
    raise ValueError(
        f"Unrecognized Condition/StandardAnnotation name: {name!r}.\n"
        f"Valid shorthand aliases: {valid_aliases}.\n"
        f"Valid systematic names: {valid_systematic}."
    )


def coerce_enum_value(enum_cls: type[IntEnum], value: str | int) -> int:
    """Coerce a YAML/JSON-provided enum value (member name or raw int) to its int value.

    Args:
      enum_cls: Backing ``IntEnum``.
      value: The member's ``.name`` (a string, case-sensitive), or its raw int value --
        JSON round-trips dict keys through strings, so a numeric-looking string
        (``"3"``) is also accepted and coerced the same as the int ``3``.

    Returns:
      The resolved int value.

    Raises:
      ValueError: If ``value`` is a bool (almost always a copy-pasted boolean-condition
        entry, not a real enum value -- ``bool`` is a subclass of ``int`` in Python and
        would otherwise silently resolve to member ``0``/``1``), or doesn't match any
        member name or value.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{enum_cls.__name__} value must be a member name or int, got bool {value!r} "
            "(did you mean to set a boolean mask condition instead?)."
        )
    if isinstance(value, str) and not value.lstrip("-").isdigit():
        try:
            return int(enum_cls[value])
        except KeyError:
            valid_names = [m.name for m in enum_cls]
            raise ValueError(
                f"{enum_cls.__name__} has no member named {value!r}. Valid names: {valid_names}."
            ) from None
    return int(enum_cls(int(value)))


def parse_centroid_value(value: str | list[float] | tuple[float, ...]) -> np.ndarray:
    """Parse a ``ChainCentroid`` (``C_CENTROID``) value into a 3-vector.

    Accepts either a comma-separated string (``"12.3,-4.5,0.0"``) or a real
    list/tuple of three floats -- see ``design_yaml_condition_spec.md`` §4.4
    for why both forms are supported.

    Args:
      value: Coordinate value from a condition spec.

    Returns:
      Length-3 ``float32`` array.

    Raises:
      ValueError: If ``value`` does not parse into exactly 3 floats.
    """
    parts = value.split(",") if isinstance(value, str) else list(value)
    if len(parts) != 3:
        raise ValueError(f"ChainCentroid value must have exactly 3 components (x, y, z), got {value!r}.")
    return np.array([float(p) for p in parts], dtype=np.float32)


@cache
def _valid_cat_labels() -> frozenset[str]:
    """Load the model's CAT vocabulary once per process."""
    with open(CATH_MAPPING_PATH, "rb") as f:
        vocabulary = pickle.load(f)
    valid_labels = {"x.x.x"}
    for depth, level in enumerate(("C", "A", "T")):
        valid_labels.update(code + ".x" * (2 - depth) for code in vocabulary[level])
    return frozenset(valid_labels)


def set_cath_condition(atom_array: AtomArray, indices: np.ndarray | list[int], value: str) -> None:
    """Set a C.A.T label, retaining the internal C.A.T.x storage representation."""
    if not isinstance(value, str):
        raise ValueError("CAT must be a quoted C.A.T label, e.g. '1.x.x' or '2.60.40'; only C/A/T are supported.")
    if len(value.split(".")) == 4 and value.endswith(".x"):
        value = value[:-2]
    if value not in _valid_cat_labels():
        raise ValueError(
            f"Unknown CAT label {value!r}; choose a C/A/T label from the model vocabulary, or 'x.x.x' to clear."
        )
    value += ".x"
    labels = C_CATH.annotation(atom_array, default="generate").astype(object)
    labels[indices] = value
    C_CATH.set_annotation(atom_array, labels.astype(str))
