"""The MSA cache contract: how a search result is named, published and read back.

Every model that searches an MSA writes into the same shared ``msa_dir``, so the
three rules below have to agree across all of them:

* the cache key is ``sha256(sequence)[:16]``, so the same sequence is searched once
  no matter which model asked for it,
* a result is published by rename, so a killed search or a short read leaves a tmp
  file behind instead of a truncated ``{hash}.a3m`` under the final name,
* a cache hit requires a non-empty file, so a zero-byte a3m from a failed search is
  re-searched instead of being accepted forever.

Before this module the rules were written five times with two of them missing the
rename and six of seven readers gating on bare ``Path.exists()``, which is the same
cache-poisoning bug the weight registry fixed at seven download sites.

Stdlib only, no ttnn: the CLI and the worker both import this at module scope.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def seq_hash(seq: str) -> str:
    """The MSA cache key for one sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def cached(path) -> bool:
    """True when an MSA cache file is present and non-empty.

    A zero-byte a3m is a failed search, not a cache hit.
    """
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def publish_text(dst, text: str) -> None:
    """Write an MSA cache file so a reader never sees a partial one."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    tmp.write_text(text)
    os.replace(tmp, dst)


def publish_file(src, dst) -> None:
    """Copy a produced MSA file into the cache under the same publish rule."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
