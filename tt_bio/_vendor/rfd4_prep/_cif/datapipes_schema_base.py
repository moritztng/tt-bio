# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Schema base classes for batched data samples.

This module provides the infrastructure for padding, batching, and dict-like access
to structured data samples used in training and inference.

Key Concepts:
    - DataSample: A single training example with typed tensor fields
    - BatchedDataSample: A batch of DataSamples with padding
    - PaddedBatchedDataSample: Provides dict-like access for backward compatibility

Dimension Naming:
    - "I": Token/index dimension (primary sequence length dimension)
    - Other dimensions follow jaxtyping conventions

Mask Convention:
    All masks use **True = valid/real, False = padding** throughout, both internally
    in ``padding_masks`` and in the public API.

Special Keys (accessed via ``batch[key]``):
    - ``"padding_mask"``: Token-level validity mask from ``padding_masks["I"]``.
      Shape ``[batch, I]``. True = real token.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator
from dataclasses import Field, dataclass, fields, is_dataclass
from functools import cached_property
from typing import Any, Union, get_args, get_origin, get_type_hints

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# Dimension name for the primary token/index dimension in jaxtyping annotations
TOKEN_DIM = "I"


def unwrap_optional(field_type: Any) -> Any:
    """Unwrap Optional[T] to get T, handling both Optional and Union[T, None]."""
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        # Optional[X] is Union[X, None], so filter out NoneType
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
    return field_type


def extract_dimension_sizes(obj: "Features") -> dict[str, int]:
    """Auto-extract dimension sizes from jaxtyped fields.

    Returns:
        Dict like {'I': 128, 'C_s': 384, 'C_z': 128}
    """
    hints = get_type_hints(obj.__class__, include_extras=True)
    dim_sizes = {}

    for field in fields(obj):
        if field.name in ("is_padded", "is_expanded"):
            continue

        field_type = hints.get(field.name)
        if field_type is None:
            logger.debug(f"Skipping field {field.name}: no type hint")
            continue

        # Unwrap Optional[T] to get T
        inner_type = unwrap_optional(field_type)
        if not hasattr(inner_type, "dim_str"):
            logger.debug(f"Skipping field {field.name}: no jaxtyping info")
            continue

        # Extract shape string: "I C_s" -> ['I', 'C_s']
        shape_str = inner_type.dim_str
        symbolic_dims = shape_str.split()

        # Get actual tensor
        tensor = getattr(obj, field.name)
        if not isinstance(tensor, torch.Tensor):
            continue

        # Map symbolic dims to actual sizes
        for i, dim_name in enumerate(symbolic_dims):
            if i >= len(tensor.shape):
                break

            actual_size = tensor.shape[i]

            if dim_name.isdecimal():
                assert actual_size == int(dim_name), (
                    f"Field {field.name} has fixed size dimension {dim_name} " f"but actual size is {actual_size}"
                )
                continue  # Skip fixed-size dimensions

            # Validate consistency (all 'I' dims should have same size)
            if dim_name in dim_sizes:
                if dim_sizes[dim_name] != actual_size:
                    raise ValueError(
                        f"Dimension {dim_name} has inconsistent sizes: "
                        f"{dim_sizes[dim_name]} vs {actual_size}"
                        f" in field {field.name}"
                    )
            else:
                dim_sizes[dim_name] = actual_size

    return dim_sizes


def pad_dimension(tensor: torch.Tensor, dim: int, max_length: int, value: float = 0.0) -> torch.Tensor:
    """Pad a specific dimension of a tensor to a maximum length.

    Args:
        tensor: The input tensor to be padded.
        dim: The dimension index to pad.
        max_length: The target length for the specified dimension.
        value: Fill value for padding. Defaults to 0.0.

    Returns:
        The padded tensor.
    """
    current_length = tensor.size(dim)

    # No padding needed
    if current_length >= max_length:
        return tensor

    padding_amount = max_length - current_length

    # F.pad expects padding in reverse dimension order:
    # (left_last, right_last, left_second_to_last, right_second_to_last, ...)
    # Build padding list: [0, 0] for all dims except the target
    padding = []

    for i in range(len(tensor.shape) - 1, -1, -1):
        if i == dim:
            # Pad this dimension on the right side
            padding.extend([0, padding_amount])
        else:
            # Don't pad this dimension
            padding.extend([0, 0])

    return F.pad(tensor, pad=padding, mode="constant", value=value)


def batch_dataclass_recursive(instances: list) -> Any:
    """
    Recursively batch a list of instances (could be dataclasses, tensors, primitives, etc.)
    """
    if not instances:
        raise ValueError("Cannot batch empty list")

    first = instances[0]

    # Base case 1: Tensor
    if isinstance(first, torch.Tensor):
        return torch.stack(list(instances), dim=0)

    # Base case 2: Primitives (int, float, str, bool, None)
    elif isinstance(first, int | float | str | bool | type(None)):
        return instances  # Keep as list or convert to tensor if all same type

    # Base case 3: Lists/tuples
    elif isinstance(first, list | tuple):
        # Option A: Keep as nested list
        # Option B: Try to stack if all elements are tensors
        return instances

    # Base case 4: Dicts - merge and batch each key
    elif isinstance(first, dict):
        # Sort keys so batched dict has deterministic insertion order; set
        # iteration would make torch.compile recompile on order shifts.
        all_keys = set()
        for instance in instances:
            all_keys.update(instance.keys())
        all_keys = sorted(all_keys)

        batched_dict = {}
        for key in all_keys:
            # Collect values for this key from all instances
            values = [instance.get(key) for instance in instances if key in instance]
            if len(values) == len(instances):
                # All instances have this key - batch the values
                batched_dict[key] = batch_dataclass_recursive(values)
            else:
                # Pad missing entries with None so the key is present for all instances
                batched_dict[key] = [instance.get(key) for instance in instances]
        return batched_dict

    # Recursive case: Dataclass
    elif is_dataclass(first):
        batched_fields = {}

        # Iterate through each field in the dataclass
        for field in fields(first):
            # Collect this field's values from all instances
            field_values = [getattr(instance, field.name) for instance in instances]

            # Recursively batch this field
            batched_fields[field.name] = batch_dataclass_recursive(field_values)

        # Construct new instance of the same dataclass type with batched fields
        return type(first)(**batched_fields)

    else:
        # Unknown type - keep as list
        return instances


def get_default_value(field: Field, hints: dict[str, Any], padding_overrides: dict[str, Any]) -> Any:
    """Return default padding value based on field type.

    Uses jaxtyping annotations to determine appropriate padding values:
        - Bool tensors: 0.0 (False)
        - Int tensors: -1 (invalid index sentinel)
        - Float tensors: NaN (clearly invalid value)
        - Unknown/fallback: 0.0

    Args:
        field: Dataclass field descriptor.
        hints: Type hints dict from get_type_hints().
        padding_overrides: Dict of field_name -> custom padding value.

    Returns:
        Default value to use for padding this field.
    """
    if field.name in padding_overrides:
        return padding_overrides[field.name]

    field_type = hints.get(field.name)
    if field_type is not None:
        # Unwrap Optional[T] to get T
        inner_type = unwrap_optional(field_type)
        origin = getattr(inner_type, "__origin__", None)
        if origin is not None:
            origin_name = getattr(origin, "__name__", "")
            if origin_name == "Bool":
                return 0.0  # False
            elif origin_name == "Int":
                return -1  # int is usually index tensor, pad with -1
            elif origin_name == "Float":
                return float("nan")
    return 0.0


@dataclass
class Features:
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Features":
        return cls(**data)

    @cached_property
    def get_default_padding_value_overrides(self) -> dict[str, Any]:
        """
        Allows the user to specify custom default padding values for specific fields.
        Returns a dictionary mapping field names to their desired default padding values.
        """
        return {}

    def pad_to_max_lengths(
        self, max_lengths: dict[str, int], exclude: str | Collection[str] | None = None
    ) -> "Features":
        # use the jaxtyping annotations to determine which dimensions to pad
        # for each field, check if its a tensor, get its jaxtyping shape, and pad accordingly
        padded_fields = {}
        hints = get_type_hints(self.__class__, include_extras=True)
        for field in fields(self):
            field_value = getattr(self, field.name)
            field_type = hints.get(field.name)
            if field_type is None:
                padded_fields[field.name] = field_value
                continue

            if not isinstance(field_value, torch.Tensor):
                padded_fields[field.name] = field_value
                continue

            if field.name == exclude or (exclude is not None and field.name in exclude):
                padded_fields[field.name] = field_value
                continue

            # Unwrap Optional[T] to get T
            inner_type = unwrap_optional(field_type)
            if not hasattr(inner_type, "dim_str"):
                padded_fields[field.name] = field_value
                continue

            shape_str = inner_type.dim_str
            symbolic_dims = shape_str.split()

            padded_tensor = field_value
            for i, dim_name in enumerate(symbolic_dims):
                if dim_name in max_lengths:
                    target_length = max_lengths[dim_name]
                    padded_tensor = pad_dimension(
                        padded_tensor,
                        dim=i,
                        max_length=target_length,
                        value=get_default_value(field, hints, self.get_default_padding_value_overrides),
                    )

            padded_fields[field.name] = padded_tensor
        return self.__class__(**padded_fields)


@dataclass
class DataSample(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataSample":
        pass  # Placeholder for from_dict method

    @abstractmethod
    def pad_to_max_lengths(self, max_lengths: dict[str, int]) -> "DataSample":
        pass  # Placeholder for padding method

    @cached_property
    def max_lengths(self) -> dict[str, int]:
        """
        Get maximum lengths for each dimension in this data sample.
        """
        max_lengths = {}
        for field in fields(self):
            if isinstance(getattr(self, field.name), Features):
                feature_obj = getattr(self, field.name)
                feature_max_lengths = extract_dimension_sizes(feature_obj)
                for dim_name, size in feature_max_lengths.items():
                    if dim_name in max_lengths:
                        max_lengths[dim_name] = max(max_lengths[dim_name], size)
                    else:
                        max_lengths[dim_name] = size
        return max_lengths


@dataclass
class BatchedDataSample(ABC):
    """Batch a list of DataSample instances with padding.

    Steps:
        1. Extract max lengths for each dimension across the batch.
        2. Pad each DataSample to these max lengths and compute ``padding_masks``
           (True = valid, False = padding) for each dimension.
        3. Stack the padded DataSamples into a single ``PaddedBatchedDataSample``.
    """

    data_samples: list[DataSample]

    def __init__(
        self,
        data_samples: list[DataSample],
        default_padding_dimensions: dict[str, int | list[int]] | None = None,
    ):
        super().__init__()
        self.data_samples = data_samples
        assert {type(s) for s in data_samples} == {type(data_samples[0])}, "All padded samples must be of the same type"

        #  Optional: specify default padding dimensions for torch.compile
        self.default_padding_dimensions = default_padding_dimensions or {}

    def get_max_lengths(self) -> dict[str, int]:
        """
        Get maximum lengths for each dimension across the batch.
        First, retrieve the max lengths from each data sample,
        then compute the overall maximum for each dimension.
        If default_padding_dimensions are provided, use them as the target sizes.
        """
        max_lengths = {}
        for data_sample in self.data_samples:
            sample_max_lengths = data_sample.max_lengths
            for dim_name, size in sample_max_lengths.items():
                if dim_name in max_lengths:
                    max_lengths[dim_name] = max(max_lengths[dim_name], size)
                else:
                    max_lengths[dim_name] = size

        # Apply fixed padding dimensions (override with fixed sizes). A value is either
        # an int (one static size) or a sequence of ints (buckets: pad up to the
        # smallest bucket that fits this batch, so torch.compile sees a finite set of
        # shapes instead of one shape per length).
        for dim_name, fixed_size in self.default_padding_dimensions.items():
            observed = max_lengths.get(dim_name, 0)
            if not isinstance(fixed_size, int):
                buckets = sorted(int(b) for b in fixed_size)
                fixed_size = next((b for b in buckets if b >= observed), None)
                if fixed_size is None:
                    raise ValueError(
                        f"Sample dimension {dim_name}={observed} exceeds the largest bucket "
                        f"{buckets[-1]}. Add a bucket >= {observed} (at inference, a longer "
                        "value in `compile_lengths`) to run an input this long."
                    )
            elif observed > fixed_size:
                raise ValueError(f"Sample dimension {dim_name}={observed} exceeds fixed size {fixed_size}")
            max_lengths[dim_name] = fixed_size

        return max_lengths

    def get_padding_masks(self, max_lengths: dict[str, int]) -> dict[str, torch.Tensor]:
        """Generate padding masks for each dimension based on max lengths.

        Convention: **True = valid/real position, False = padding.**
        This matches the convention used by all consumers (model, feature factories, etc.).

        Returns:
            Dictionary mapping dimension names (e.g. ``"I"``, ``"S"``) to boolean masks
            of shape ``[batch_size, max_length]``.
        """
        padding_masks = {}
        for dim_name, max_size in max_lengths.items():
            masks = []
            for data_sample in self.data_samples:
                sample_max_lengths = data_sample.max_lengths
                current_size = sample_max_lengths.get(dim_name, 0)
                mask = torch.zeros(max_size, dtype=torch.bool)
                mask[:current_size] = True  # True = valid position
                masks.append(mask)
            padding_masks[dim_name] = torch.stack(masks)
        return padding_masks

    def pad_and_stack(self) -> "DataSample":
        """
        Take a list of DataSample instances and batch them into a single DataSample instance,
        padding dimensions as necessary to ensure uniform sizes across the batch.
        """
        max_lengths = self.get_max_lengths()
        padding_masks = self.get_padding_masks(max_lengths)
        padded_samples = [sample.pad_to_max_lengths(max_lengths) for sample in self.data_samples]

        # take all fields with same name and stack them
        stacked_fields = batch_dataclass_recursive(padded_samples)
        return PaddedBatchedDataSample(data_samples=stacked_fields, padding_masks=padding_masks)


@dataclass
class PaddedBatchedDataSample(BatchedDataSample):
    padding_masks: dict[str, torch.Tensor]

    # Key mappings from old dict-style access to new dataclass structure
    # Override this in subclasses for schema-specific mappings
    _KEY_MAPPINGS: dict[str, str] = None  # Set in __post_init__ or subclass

    def _get_nested_attr(self, path: str) -> Any:
        """Get a nested attribute by dot-separated path (e.g. 'token_features.residue_type')."""
        obj = self.data_samples
        for part in path.split("."):
            if obj is None:
                return None
            obj = getattr(obj, part, None)
        return obj

    def _build_key_mappings(self) -> dict[str, str]:
        """Build mappings from flat keys to nested paths. Override in subclasses."""
        # For the simplified ProteinaDataSample, all tensors are in 'tensors' dict
        # This mapping handles both direct field names and C_* condition names
        return {
            # Top-level attributes (not in tensors dict)
            "example_id_batch": "example_id",
            "dataset_name_batch": "dataset_name",
            "atom_array_batch": "atom_array",
            "task_batch": "task",
            "metrics_tags_batch": "metrics_tags",
            "specification_batch": "specification",
        }
        # Note: Most keys are looked up directly in data_samples.tensors

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for backward compatibility with proteina.py."""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        try:
            self[key]
            return True
        except KeyError:
            return False

    def pop(self, key: str, *args) -> Any:
        """Get value by key with optional default (does NOT remove the key).

        Note:
            Unlike dict.pop(), this method does not remove the key from the dataclass.
            Dataclass fields are immutable, so removal is not supported. This method
            exists for backward compatibility with code that uses dict.pop() syntax.

        Args:
            key: The key to look up.
            *args: Optional default value if key is not found.

        Returns:
            The value associated with key, or default if provided and key not found.

        Raises:
            KeyError: If key not found and no default provided.
        """
        try:
            return self[key]
        except KeyError:
            if args:
                return args[0]
            raise

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like assignment for adding computed fields at runtime."""
        # Store in a dynamic dict for runtime-added keys
        if not hasattr(self, "_extra_fields"):
            object.__setattr__(self, "_extra_fields", {})
        self._extra_fields[key] = value

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backward compatibility with proteina.py."""
        # Check extra fields first (runtime-added via batch["key"] = value)
        if hasattr(self, "_extra_fields") and key in self._extra_fields:
            return self._extra_fields[key]

        # Special key: "padding_mask" returns token-level validity mask
        # Convention: True = real/valid token, False = padding
        if key == "padding_mask":
            if TOKEN_DIM in self.padding_masks:
                return self.padding_masks[TOKEN_DIM]
            raise KeyError(f"No '{TOKEN_DIM}' dimension in padding_masks")

        # Check tensors dict (simplified schema - keys are passed through as-is)
        if (
            hasattr(self.data_samples, "tensors")
            and isinstance(self.data_samples.tensors, dict)
            and key in self.data_samples.tensors
        ):
            return self.data_samples.tensors[key]

        # Build key mappings if not cached
        if self._KEY_MAPPINGS is None:
            object.__setattr__(self, "_KEY_MAPPINGS", self._build_key_mappings())

        # Check if key is in mappings (for top-level attrs like example_id_batch)
        if key in self._KEY_MAPPINGS:
            path = self._KEY_MAPPINGS[key]
            result = self._get_nested_attr(path)
            if result is not None:
                return result

        # Try direct attribute access on data_samples
        if hasattr(self.data_samples, key):
            return getattr(self.data_samples, key)

        raise KeyError(f"Key '{key}' not found. Available: {list(self.keys())[:20]}...")

    def keys(self) -> set[str]:
        """Return all available keys for dict-like access."""
        k = set()
        if hasattr(self, "_extra_fields"):
            k.update(self._extra_fields.keys())
        k.add("padding_mask")
        if hasattr(self.data_samples, "tensors") and isinstance(self.data_samples.tensors, dict):
            k.update(self.data_samples.tensors.keys())
        for f in fields(self.data_samples):
            k.add(f.name)
        return k

    def items(self) -> Iterator[tuple[str, Any]]:
        """Return dict-like ``(key, value)`` pairs."""
        return ((key, self[key]) for key in self.keys())
