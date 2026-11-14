# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
import contextlib
import os
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from typing import Any, Callable, Iterable
from toolz import merge_with


def run_once(fn: Callable) -> Callable:
    """Decorator to ensure a function is only executed once per process.

    Args:
        fn (Callable): The function to decorate.

    Returns:
        Callable: A wrapped function that only executes once.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if getattr(wrapper, "_has_run", False):
            return
        wrapper._has_run = True
        return fn(*args, **kwargs)

    return wrapper


def do_nothing(*args: Any, **kwargs: Any) -> None:
    """Does nothing, just returns None"""
    pass


def exists(obj: Any) -> bool:
    """True iff object is not None"""
    return obj is not None


def default(obj: Any, default: Any) -> Any:
    """Return obj if it exists, otherwise return default"""
    return obj if exists(obj) else default


def exactly_one_exists(*args: object) -> bool:
    """True iff exactly one of the arguments exists"""
    return sum(exists(arg) for arg in args) == 1


def at_least_one_exists(*args: object) -> bool:
    """True iff at least one of the arguments exists"""
    return any(exists(arg) for arg in args)


def concat_dicts(*dicts: dict) -> dict:
    """
    Concatenate a list of dicts with the same keys into a single dict.

    Example:
        >>> d1 = {"a": 1, "b": 2}
        >>> d2 = {"a": 3, "b": 4}
        >>> concat_dicts(d1, d2)
        {'a': [1, 3], 'b': [2, 4]}
    """
    return merge_with(list, *dicts)


def listmap(fn: Callable, lst: Iterable[Any]) -> list:
    """
    Apply a function to each element of a single list.

    Args:
        - fn (Callable): Function to apply to each element
        - lst (list): Input list

    Returns:
        - list: Result of applying fn to each element

    Example:
        >>> listmap(lambda x: x + 1, [1, 2, 3])
        [2, 3, 4]
    """
    return [fn(x) for x in lst]


def listmap_with_idx(fn: Callable[[int, Any], Any], lst: Iterable[Any]) -> list:
    """Maps a function over a list while providing both index and value to the function.

    A convenience wrapper around listmap that allows the mapping function to access both the index and value
    of each element in the input list.

    Args:
        - fn (Callable[[int, Any], Any]): Function that takes two arguments (index, value) and returns
          a transformed value.
        - lst (list): Input list to map over.

    Returns:
        - list: New list containing the results of applying fn to each (index, value) pair.

    Example:
        >>> def add_index(i, x):
        ...     return f"{i}_{x}"
        >>> listmap_with_idx(add_index, ["a", "b", "c"])
        ['0_a', '1_b', '2_c']
    """
    return [fn(idx, x) for idx, x in enumerate(lst)]


def true_with_prob(p: float | np.ndarray, rng: np.random.Generator | None = None) -> bool | np.ndarray:
    """Return True with probability `p`, vectorized for scalars or arrays.

    Args:
        p: Probability or array of probabilities (each in [0.0, 1.0]).
        rng: Random number generator to use. If None, uses numpy's global
            random state.

    Returns:
        bool or np.ndarray: True/False or boolean array, True with probability `p`.

    Raises:
        ValueError: If any probability is not in [0.0, 1.0].

    Example:
        >>> true_with_prob(0.5)
        True  # or False, with 50% chance
        >>> true_with_prob(np.array([0.0, 0.5, 1.0]))
        array([False, True, True])  # stochastic for 0.5
    """
    rng = default(rng, np.random)
    p_arr = np.asarray(p)
    if np.any(p_arr < 0.0):
        raise ValueError(f"`p` must be >= 0. Got {p}")
    if np.any(p_arr > 1.0):
        raise ValueError(f"`p` must be <= 1. Got {p}")

    # Handle edge cases efficiently
    if np.isscalar(p):
        if p == 0.0:
            return False
        if p == 1.0:
            return True
        return rng.random() < p
    else:
        result = np.zeros_like(p_arr, dtype=bool)
        result[p_arr == 1.0] = True
        mask = (p_arr > 0.0) & (p_arr < 1.0)
        if np.any(mask):
            result[mask] = rng.random(size=np.count_nonzero(mask)) < p_arr[mask]
        return result


def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert tensor to target dtype if it's not already that dtype."""
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def ensure_dir(path: "str | os.PathLike") -> Path:
    """Create a directory, tolerating another rank creating it at the same time.

    ``mkdir(exist_ok=True)`` is not enough on NFS: pathlib cannot trust EEXIST, so it falls back to
    an ``is_dir()`` check that races a concurrent mkdir and re-raises FileExistsError, killing the
    task. A name that exists but is not a directory is left to fail at the first write.
    """
    path = Path(path)
    with contextlib.suppress(FileExistsError):
        path.mkdir(parents=True, exist_ok=True)
    return path


def validate_design_id(design_id: str) -> str:
    """Return ``design_id`` unchanged, raising if it cannot serve as a single directory name."""
    if design_id in ("", ".", "..") or any(c in design_id for c in ("/", "\\", "\0")):
        raise ValueError(
            f"Invalid design id {design_id!r}: ids name a single output directory, so they cannot be "
            "blank, '.', '..', or contain a path separator."
        )
    return design_id


def design_output_dir(base_dir: Path | str, design_id: str) -> Path:
    """Return the ``base_dir / design_id`` directory holding every file for one design.

    Design ids come from keys in a user-supplied input file, so an id like ``..`` or ``a/b``
    would otherwise write (or, at the cleanup sites that ``rmtree`` a design directory, delete)
    outside the run's output tree.

    Raises:
      ValueError: If the id is not a plain directory name, or if the join does not land on a
        direct child of ``base_dir`` (e.g. it resolves through a symlink).
    """
    base_dir = Path(base_dir)
    design_dir = base_dir / validate_design_id(design_id)
    if design_dir.resolve().parent != base_dir.resolve():
        raise ValueError(
            f"Refusing to use {design_dir} for design id {design_id!r}: it is not a direct child "
            f"of the output base {base_dir}."
        )
    return design_dir
