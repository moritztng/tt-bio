"""CIF-based dataset loaders."""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from tt_bio._vendor.atomworks.io.parser import STANDARD_PARSER_ARGS, parse

from .base import _construct_metadata_hierarchy, _construct_structure_path


def _load_structure_from_path(path: Path, assembly_id: str, parser_args: dict | None = None) -> dict[str, Any]:
    """Load structure from file path using the CIF parser, merging with STANDARD_PARSER_ARGS."""
    # Merge STANDARD_PARSER_ARGS with parser_args (parser_args takes precedence)
    merged_args = {**STANDARD_PARSER_ARGS, **(parser_args or {})}
    result_dict = parse(
        filename=path,
        build_assembly=(assembly_id,),
        **merged_args,
    )
    return result_dict


def _base_loader_function(
    row: pd.Series,
    example_id_colname: str,
    path_colname: str,
    assembly_id_colname: str | None,
    attrs: dict,
    base_path: str,
    extension: str,
    sharding_pattern: str | None,
    parser_args: dict | None,
) -> dict[str, Any]:
    """Base loader function (picklable when used with functools.partial)."""
    # Prepare loader-specific attributes
    loader_attrs = attrs.copy()
    if base_path and "base_path" not in loader_attrs:
        loader_attrs["base_path"] = base_path
    if extension and "extension" not in loader_attrs:
        loader_attrs["extension"] = extension

    extra_info = _construct_metadata_hierarchy(row, loader_attrs)

    assembly_id = row[assembly_id_colname] if assembly_id_colname is not None and assembly_id_colname in row else "1"
    path = _construct_structure_path(
        row[path_colname], extra_info.get("base_path"), extra_info.get("extension"), sharding_pattern
    )
    result_dict = _load_structure_from_path(path, assembly_id, parser_args)

    # Remove used columns from extra_info
    exclude_cols = (
        [example_id_colname, path_colname]
        + ([assembly_id_colname] if assembly_id_colname else [])
        + ["base_path", "extension"]
    )
    extra_info = {k: v for k, v in extra_info.items() if k not in exclude_cols}

    return {
        "example_id": row[example_id_colname],
        "path": path,
        "assembly_id": assembly_id,
        "extra_info": extra_info,
        "atom_array": result_dict["assemblies"][assembly_id][0],
        "atom_array_stack": result_dict["assemblies"][assembly_id],
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }


def create_base_loader(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    attrs: dict | None = None,
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    parser_args: dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a picklable base loader for AtomWorks datasets.

    Args:
        example_id_colname: Name of column containing unique example identifiers
        path_colname: Name of column containing paths to structure files
        assembly_id_colname: Optional column name containing assembly IDs.
            If None, assembly_id defaults to "1" for all examples.
        attrs: Additional attributes to merge with highest precedence into the metadata hierarchy
            (and ultimately included in the output dictionary's "extra_info" key).
        base_path: Base path to prepend to file paths if not included in path column
        extension: File extension to add/replace if not included in path column
        sharding_pattern: Pattern for how files are organized in subdirectories, if not specified in the path
            - "/1:2/": Use characters 1-2 for first directory level
            - "/1:2/0:2/": Use chars 1-2 for first dir, then chars 0-2 for second dir
            - None: No sharding (default)
        parser_args: Optional dictionary of arguments to pass to the CIF parser when loading the structure file.

    Returns:
        A picklable loader function (via functools.partial) for multiprocessing.
    """
    return functools.partial(
        _base_loader_function,
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        attrs=attrs or {},
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )


def _loader_with_query_pn_units_function(
    row: pd.Series,
    base_loader: Callable,
    pn_unit_iid_colnames: list[str],
) -> dict[str, Any]:
    """Loader with query pn_units (picklable when used with functools.partial)."""
    result = base_loader(row)
    result["extra_info"] = {k: v for k, v in result["extra_info"].items() if k not in pn_unit_iid_colnames}
    query_pn_unit_iids = [row[colname] for colname in pn_unit_iid_colnames]
    result["query_pn_unit_iids"] = query_pn_unit_iids
    return result


def create_loader_with_query_pn_units(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    pn_unit_iid_colnames: str | list[str] | None = None,
    assembly_id_colname: str | None = "assembly_id",
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    attrs: dict | None = None,
    parser_args: dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a picklable loader for pipelines with query pn_units (chains).

    For instance, in the interfaces dataset, each sampled row contains two pn_unit instance IDs
    that should be included in the cropped structure.

    Examples:
        Interfaces dataset:
            >>> loader = create_loader_with_query_pn_units(
            ...     pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"], assembly_id_colname="assembly_id"
            ... )

        Chains dataset:
            >>> loader = create_loader_with_query_pn_units(
            ...     pn_unit_iid_colnames="pn_unit_iid", base_path="/data/structures", extension=".cif.gz"
            ... )
    """
    # Normalize pn_unit_iid_colnames to list format
    if isinstance(pn_unit_iid_colnames, str):
        pn_unit_iid_colnames = [pn_unit_iid_colnames]
    pn_unit_iid_colnames = pn_unit_iid_colnames or []

    # Create base loader with common parameters
    base_loader = create_base_loader(
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )

    return functools.partial(
        _loader_with_query_pn_units_function,
        base_loader=base_loader,
        pn_unit_iid_colnames=pn_unit_iid_colnames,
    )


def _loader_with_interfaces_and_pn_units_to_score_function(
    row: pd.Series,
    base_loader: Callable,
    interfaces_to_score_colname: str | None,
    pn_units_to_score_colname: str | None,
) -> dict[str, Any]:
    """Loader with scoring info (picklable when used with functools.partial)."""
    result = base_loader(row)
    exclude_cols = [interfaces_to_score_colname, pn_units_to_score_colname]
    result["extra_info"] = {k: v for k, v in result["extra_info"].items() if k not in exclude_cols}

    interfaces_to_score = row[interfaces_to_score_colname] if interfaces_to_score_colname is not None else None
    pn_units_to_score = row[pn_units_to_score_colname] if pn_units_to_score_colname is not None else None

    result.update(
        {
            "interfaces_to_score": interfaces_to_score,
            "pn_units_to_score": pn_units_to_score,
        }
    )
    return result


def create_loader_with_interfaces_and_pn_units_to_score(
    example_id_colname: str = "example_id",
    path_colname: str = "path",
    assembly_id_colname: str | None = "assembly_id",
    interfaces_to_score_colname: str | None = "interfaces_to_score",
    pn_units_to_score_colname: str | None = "pn_units_to_score",
    base_path: str = "",
    extension: str = "",
    sharding_pattern: str | None = None,
    attrs: dict | None = None,
    parser_args: dict | None = None,
) -> Callable[[pd.Series], dict[str, Any]]:
    """Factory function that creates a picklable loader for validation datasets with scoring information.

    Example:
        >>> loader = create_loader_with_interfaces_and_pn_units_to_score(
        ...     interfaces_to_score_colname="interfaces_to_score", pn_units_to_score_colname="pn_units_to_score"
        ... )
    """
    # Create base loader with common parameters
    base_loader = create_base_loader(
        example_id_colname=example_id_colname,
        path_colname=path_colname,
        assembly_id_colname=assembly_id_colname,
        attrs=attrs,
        base_path=base_path,
        extension=extension,
        sharding_pattern=sharding_pattern,
        parser_args=parser_args,
    )

    return functools.partial(
        _loader_with_interfaces_and_pn_units_to_score_function,
        base_loader=base_loader,
        interfaces_to_score_colname=interfaces_to_score_colname,
        pn_units_to_score_colname=pn_units_to_score_colname,
    )
