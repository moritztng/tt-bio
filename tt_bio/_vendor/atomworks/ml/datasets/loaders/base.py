"""Shared utilities for dataset loaders."""

from pathlib import Path
from typing import Any

import pandas as pd

from tt_bio._vendor.atomworks.io.utils.io_utils import apply_sharding_pattern


def _construct_metadata_hierarchy(row: pd.Series, attrs: dict | None = None) -> dict[str, Any]:
    """Construct metadata dictionary with precedence hierarchy.

    Assembles metadata from multiple sources with the following precedence (lowest to highest priority):
        1. DataFrame-level attributes (row.attrs)
        2. Row-level data (row.to_dict())
        3. Loader-specific attributes (attrs parameter)

    Args:
        row: pandas Series representing one dataset example
        attrs: Optional loader-specific attributes to merge with highest precedence

    Returns:
        Dictionary containing merged metadata with proper hierarchy
    """
    # Start with DataFrame-level attributes (lowest precedence)
    extra_info = row.attrs.copy() if hasattr(row, "attrs") else {}

    # Add row-level data (middle precedence)
    extra_info.update(row.to_dict())

    # Add loader-specific attributes (highest precedence)
    extra_info.update(attrs or {})

    return extra_info


def _construct_structure_path(
    path: str, base_path: str | None, extension: str | None, sharding_pattern: str | None = None
) -> Path:
    """Construct file path with optional base_path, extension, and sharding pattern.

    Args:
        path: The base path or identifier (e.g., PDB ID)
        base_path: Base directory to prepend
        extension: File extension to add/replace
        sharding_pattern: Pattern for organizing files in subdirectories
            - "/1:2/": Use characters 1-2 for first directory level
            - "/1:2/0:2/": Use chars 1-2 for first dir, then chars 0-2 for second dir
            - None: No sharding (default)
    """
    sharded_path = apply_sharding_pattern(path, sharding_pattern)

    if base_path:
        final_path = Path(base_path) / sharded_path
    else:
        final_path = sharded_path

    if extension:
        final_path = final_path.with_suffix(extension)

    return final_path
