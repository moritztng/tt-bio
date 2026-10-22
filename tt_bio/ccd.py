"""``ccd.pkl`` as a mapping that reads one molecule at a time.

The CCD pickle is a dict of ~50k RDKit mols, 417 MB; unpickling it builds every mol
and takes 4-5 s, which a protein fold pays once per process to read 20 standard
residues. ``CcdMols`` derives an index from the pickle once (keyed on its path, size,
mtime and the RDKit version) and then maps only that index, so opening it costs
milliseconds and each code is built when it is first asked for.

The index stores, per code, the arguments and state pickle would pass to ``Mol``, not a
re-pickled mol. Reading an entry runs the same ``Mol(binary)`` + ``__setstate__`` that
``pickle.load`` runs, so a caller gets what the full load gives under the RDKit pickle
options in force at that moment, whatever options the index was written under.
"""

from __future__ import annotations

import hashlib
import io
import mmap
import pickle
import struct
import sys
from collections.abc import Mapping
from pathlib import Path

from tt_bio.cache import cached, staged

_MAGIC = b"TTCCD1\0\0"
_MOL = ("rdkit.Chem.rdchem", "Mol")


class _Captured:
    """Stands in for ``Mol`` during the index build: keeps what pickle hands it."""

    def __init__(self, *args):
        self.args, self.state = args, None

    def __setstate__(self, state):
        self.state = state


class _CapturingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _Captured if (module, name) == _MOL else super().find_class(module, name)


def _entry(value) -> bytes:
    if isinstance(value, _Captured):
        return pickle.dumps(("mol", value.args, value.state), protocol=5)
    return pickle.dumps(("obj", value), protocol=5)


def _load_entry(buf):
    kind, *rest = pickle.loads(buf)  # noqa: S301 - our own derived file
    if kind == "obj":
        return rest[0]
    from rdkit import Chem

    args, state = rest
    mol = Chem.Mol(*args)
    if state is not None:
        mol.__setstate__(state)
    return mol


def write_index(ccd_pkl: Path, out) -> None:
    """Write the per-code index for ``ccd_pkl`` into the binary file object ``out``."""
    with open(ccd_pkl, "rb") as f:
        raw = _CapturingUnpickler(f).load()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict[str, Mol] in {ccd_pkl}, got {type(raw).__name__}")
    out.write(_MAGIC + b"\0" * 8)
    offsets = {}
    for code, value in raw.items():
        buf = _entry(value)
        offsets[code] = (out.tell(), len(buf))
        out.write(buf)
    index_at = out.tell()
    pickle.dump(offsets, out, protocol=5)
    out.seek(len(_MAGIC))
    out.write(struct.pack("<Q", index_at))


def index_path(ccd_pkl: Path) -> Path:
    from rdkit import __version__ as rdkit_version

    from tt_bio.weights import cache_root

    st = ccd_pkl.stat()
    key = f"{_MAGIC!r}:{ccd_pkl}:{st.st_size}:{st.st_mtime_ns}:{rdkit_version}"
    return cache_root() / "ccd" / f"{ccd_pkl.stem}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.idx"


class CcdMols(Mapping):
    """Read-only ``dict[code, Mol]`` over a ``ccd.pkl``, one mol built per first lookup."""

    def __init__(self, ccd_pkl: Path | str):
        self.path = Path(ccd_pkl).resolve()
        idx = index_path(self.path)
        try:
            if not cached(idx):
                with staged(idx) as tmp, open(tmp, "wb") as out:
                    write_index(self.path, out)
            with open(idx, "rb") as f:
                self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError as e:
            # A full or read-only cache root: the same index, held in this process.
            print(f"[tt-bio] CCD index not cached ({e}); building it in memory", file=sys.stderr)
            out = io.BytesIO()
            write_index(self.path, out)
            self._mm = out.getbuffer()
        if self._mm[: len(_MAGIC)] != _MAGIC:
            raise ValueError(f"{idx} is not a CCD index")
        (index_at,) = struct.unpack_from("<Q", self._mm, len(_MAGIC))
        self._offsets = pickle.load(io.BytesIO(self._mm[index_at:]))  # noqa: S301
        self._mols: dict = {}

    def __getitem__(self, code):
        if code not in self._mols:
            off, n = self._offsets[code]
            self._mols[code] = _load_entry(self._mm[off : off + n])
        return self._mols[code]

    def __contains__(self, code) -> bool:
        return code in self._offsets

    def __iter__(self):
        return iter(self._offsets)

    def __len__(self) -> int:
        return len(self._offsets)
