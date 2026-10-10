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


def paired_msa_dir(msa_dir, seqs) -> Path | None:
    """Where the paired MSA of a complex with these protein sequences lives, or None when
    the complex does not pair.

    A paired alignment belongs to the complex, not to one chain: row j of chain A lines up
    with row j of its partner, so the same chain paired against a different partner has
    different rows. It is keyed by the set of unique sequences. One sequence (a monomer, or
    a homomer's identical copies) has nothing to pair against, which is the rule Protenix,
    OpenDDE, OpenFold3 and Boltz-2 all apply upstream.

    Under ``paired-v2``: an offline search wrote ``paired/`` with each chain's unpaired hits
    until compute_msa_offline learned to pair, and a cache hit would keep serving those.
    """
    uniq = sorted(set(seqs))
    if len(uniq) < 2:
        return None
    return Path(msa_dir) / "paired-v2" / hashlib.sha256("\n".join(uniq).encode()).hexdigest()[:16]


#: A chain's ``msa: empty`` (YAML) or ``>A|protein|empty`` (FASTA): fold it single-sequence. The
#: reader keeps it as its own value rather than None, because None means "no MSA given" and
#: every MSA-folding model searches for one of those -- which is how ``msa: empty`` came to send
#: the chain to the online server and fold it at full alignment depth.
EMPTY_MSA = "empty"


def msa_pinned(spec) -> bool:
    """True when the input already settled a chain's MSA: ``empty``, or an a3m path that
    exists. A pinned chain is never searched and never picks up the hash cache."""
    return spec == EMPTY_MSA or bool(spec and Path(spec).expanduser().exists())


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

    The tmp name KEEPS the destination's suffix, because a producer is allowed to care
    about it: ``np.savez_compressed`` appends ``.npz`` when the path it is handed does
    not already end in it, so a tmp called ``.x.npz.1234.tmp`` gets written as
    ``.x.npz.1234.tmp.npz`` and the rename then fails on a file that is not there.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.stem}.{os.getpid()}.tmp{dst.suffix}"
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


class TrunkCache:
    """Trunk outputs on disk, keyed by everything that determines them.

    OpenDDE's constraint guidance only acts inside the diffusion sampler, so inputs that
    differ only in their ``constraint`` (and the unconstrained input) fold through the same
    trunk. The key hashes every tensor and value of the model feature dict (guidance features
    are not in it: they travel with the Guidance object), the recycle count, a fingerprint of
    the weights, the --fast mode and every ``TT_BIO_*`` / ``OPENDDE_*`` setting in the
    environment, since those switch trunk numerics. Entries are ``torch.save`` files read back
    with ``weights_only=True``, so a cache directory cannot run code. An unreadable entry is
    recomputed and an unwritable one is skipped, each with a warning.
    """

    def __init__(self, root):
        self.root = Path(root).expanduser()

    @staticmethod
    def key(feats: dict, *parts) -> str:
        import torch
        h = hashlib.sha256()
        for k in sorted(feats):
            v = feats[k]
            h.update(k.encode())
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().contiguous()
                h.update(f"{v.dtype}{tuple(v.shape)}".encode())
                h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                h.update(repr(v).encode())
        for p in parts:
            h.update(repr(p).encode())
        env = sorted((k, v) for k, v in os.environ.items() if k.startswith(("TT_BIO_", "OPENDDE_")))
        h.update(repr(env).encode())
        return h.hexdigest()[:32]

    def load(self, key: str):
        import logging
        import torch
        p = self.root / f"{key}.pt"
        if not cached(p):
            return None
        try:
            return torch.load(p, map_location="cpu", weights_only=True)
        except Exception as e:  # a torn or foreign file is a miss, not a crash
            logging.getLogger(__name__).warning("trunk cache %s unreadable (%s); recomputing", p, e)
            return None

    def save(self, key: str, value) -> None:
        import logging
        import torch
        try:
            with staged(self.root / f"{key}.pt") as tmp:
                torch.save(value, tmp)
        except OSError as e:
            logging.getLogger(__name__).warning("trunk cache %s not written (%s)", self.root, e)
