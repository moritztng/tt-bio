"""The on-disk cache contract: how a produced artifact is published and read back.

Every model shares these caches, so the rules have to agree across all of them:

* a cache entry is published by rename, so a killed search, a dropped connection or
  a short read leaves a tmp file behind instead of a truncated file under the final
  name,
* a cache hit requires a non-empty file, so a zero-byte artifact from a failed
  producer is redone instead of being accepted forever,
* the MSA cache key is ``sha256(sequence)[:16]``, so the same sequence is searched
  once no matter which model asked for it.

Before this module the publish rule was written five times with two producers
missing the rename, six of seven MSA readers gated on bare ``Path.exists()``, and
the cache key was inlined at twelve sites. That is the same cache-poisoning bug the
weight registry fixed at seven download sites, in the MSA path and in the
OpenFold3 template fetch.

Stdlib only, no ttnn: the CLI and the worker both import this at module scope.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from contextlib import contextmanager
from pathlib import Path


def seq_hash(seq: str) -> str:
    """The MSA cache key for one sequence."""
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def cached(path) -> bool:
    """True when a cache entry is present and non-empty.

    A zero-byte a3m is a failed search, not a cache hit.
    """
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


@contextmanager
def staged(dst):
    """Yield a tmp path to produce into, and publish it by rename only on success.

    The one place the publish rule lives. A producer that raises, or a process that
    is killed, leaves the tmp file rather than a partial ``dst`` that every later
    reader accepts.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.tmp"
    try:
        yield tmp
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def publish_text(dst, text: str) -> None:
    """Write a cache entry from text."""
    with staged(dst) as tmp:
        tmp.write_text(text)


def publish_file(src, dst) -> None:
    """Copy a produced file into the cache."""
    with staged(dst) as tmp:
        shutil.copy2(src, tmp)
