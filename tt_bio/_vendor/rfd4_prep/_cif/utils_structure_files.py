# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Compression-independent structure discovery and filenames."""

from collections.abc import Iterable
from os import PathLike
from pathlib import Path

from tt_bio._vendor.atomworks.io.utils.io_utils import CIF_LIKE_EXTENSIONS

# In this codebase, jsons are never structure files, and instead may be ConditionSpec inputs
_ALL_SUFFIXES = tuple(suffix for suffix in CIF_LIKE_EXTENSIONS if not suffix.startswith(".json"))
STRUCTURE_FILE_SUFFIXES = tuple(suffix for suffix in _ALL_SUFFIXES if not suffix.startswith(".pdb"))


def is_structure_file(path: PathLike | str, *, include_pdb: bool = False) -> bool:
    """Recognize exact structure suffixes, excluding backup and temporary files."""
    return str(path).lower().endswith(_ALL_SUFFIXES if include_pdb else STRUCTURE_FILE_SUFFIXES)


def structure_file_stem(path: PathLike | str) -> str:
    """Remove the structure format and compression suffix, preserving dots in IDs."""
    name = Path(path).name
    for suffix in _ALL_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported structure filename: {path}")


def unique_structure_files(paths: Iterable[PathLike | str]) -> list[Path]:
    """Sort structures and reject multiple encodings of the same sibling design."""
    result = sorted(Path(path) for path in paths)
    identities: dict[Path, Path] = {}
    for path in result:
        key = path.absolute().with_name(structure_file_stem(path))
        if key in identities:
            raise ValueError(f"Ambiguous structure files for {key}: {identities[key]} and {path}. Keep one encoding.")
        identities[key] = path
    return result


def iter_structure_files(
    directory: PathLike | str, *, include_pdb: bool = False, stem_suffix: str | None = None, pattern: str = "**/*"
) -> list[Path]:
    """Discover mixed encodings by glob pattern (recursive by default) and optional artifact role."""
    return unique_structure_files(
        path
        for path in Path(directory).glob(pattern)
        if is_structure_file(path, include_pdb=include_pdb)
        and (stem_suffix is None or structure_file_stem(path).endswith(stem_suffix))
        and path.is_file()
    )


def find_structure_file(nominal_path: PathLike | str) -> Path | None:
    """Find one sibling encoding of a nominal structure path, or return None."""
    nominal_path = Path(nominal_path)
    stem = structure_file_stem(nominal_path)
    if not nominal_path.parent.is_dir():
        return None
    matches = unique_structure_files(
        path
        for path in nominal_path.parent.iterdir()
        if path.is_file() and is_structure_file(path, include_pdb=True) and structure_file_stem(path) == stem
    )
    return matches[0] if matches else None
