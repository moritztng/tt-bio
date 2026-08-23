"""Thin I/O helpers shared between featurizer and inference modules."""

from pathlib import Path
from typing import Optional

from safetensors.torch import safe_open
from torch import Tensor


def load_safetensor(
    path: Path, key: Optional[str] = "embeddings"
) -> Tensor | dict[str, Tensor]:
    """Load a safetensor file.

    Parameters
    ----------
    path
        File path.
    key
        Tensor name to load (default ``embeddings``). If ``None``, returns
        every key as a ``dict[str, Tensor]``.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        if key is None:
            return {k: f.get_tensor(k).float() for k in f.keys()}
        return f.get_tensor(key)
