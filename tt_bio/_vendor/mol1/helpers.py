# Vendored from Mol-1 (https://github.com/molence/Mol-1) @ e6e64866ee89640a91cef7ee777a85f1d8be03bb, Apache-2.0.
# Copyright 2024 ByteDance and/or its affiliates; modifications Copyright 2026 Molence Bio.
# Regenerate with scripts/mol1_io/vendor_templates.py; do not edit by hand.
import json
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

get_logger = logging.getLogger
_JSON_FILE_CACHE = {}


def pad_to(arr: np.ndarray, shape: tuple, **kwargs) -> np.ndarray:
    """Pads an array to a given shape. Wrapper around np.pad().

    Args:
      arr: numpy array to pad
      shape: target shape, use None for axes that should stay the same
      **kwargs: additional args for np.pad, e.g. constant_values=-1

    Returns:
      the padded array

    Raises:
      ValueError if arr and shape have a different number of axes.
    """
    if arr.ndim != len(shape):
        raise ValueError(
            f"arr and shape have different number of axes. {arr.shape=}, {shape=}"
        )

    num_pad = []
    for axis, width in enumerate(shape):
        if width is None:
            num_pad.append((0, 0))
        else:
            if width >= arr.shape[axis]:
                num_pad.append((0, width - arr.shape[axis]))
            else:
                raise ValueError(
                    f"Can not pad to a smaller shape. {arr.shape=}, {shape=}"
                )
    padded_arr = np.pad(arr, pad_width=num_pad, **kwargs)
    return padded_arr

def load_json_cached(path: Union[str, Path]) -> Any:
    """Upstream mol.utils.file_io.load_json_cached, JSON branch only: load once per process."""
    path_str = str(Path(path))
    if path_str == ".":
        return {}
    if not path_str.endswith(".json"):
        raise ValueError(f"Unsupported file format: {path_str}")
    if path_str not in _JSON_FILE_CACHE:
        with open(path_str, "r") as f:
            _JSON_FILE_CACHE[path_str] = json.load(f)
    return _JSON_FILE_CACHE[path_str]
