"""Read a YAML input file as a mapping, and refuse anything else.

``yaml.safe_load`` returns ``None`` for an empty or comment-only file, and every reader
in this repo then reaches straight for a key. The user gets ``AttributeError: 'NoneType'
object has no attribute 'get'`` from inside a parser, with no mention of the file they
passed. ``main.py``'s rfd3 reader already had the answer -- check the type, name the file
-- and the fix was written there and nowhere else.

The one loader, so a reader cannot forget the check by writing ``yaml.safe_load``
directly. ``tt_bio/data/__init__.py`` is empty, so this costs no import weight to reach
from ``boltzgen`` or ``pxdesign``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_mapping(doc: Any, path: Path | str) -> dict:
    """Return ``doc`` if it is a non-empty mapping, else raise ``ValueError`` naming ``path``.

    Split out from :func:`load_mapping` for the one caller that has to load the document
    itself (a reader that accepts ``.yaml`` or ``.pdb`` and dispatches on the suffix).
    Prefer ``load_mapping``: it cannot be skipped by accident.
    """
    if not isinstance(doc, dict) or not doc:
        what = "empty" if not doc else f"a {type(doc).__name__}, not a mapping"
        raise ValueError(f"{path}: expected a YAML mapping of settings, got {what}")
    return doc


def load_mapping(path: Path | str, *, file=None) -> dict:
    """Parse ``path`` as YAML and return it as a non-empty mapping.

    ``file`` takes an already-open handle for callers inside a ``with`` block, so they
    keep their own file handling and still get the check.
    """
    import yaml

    path = Path(path)
    doc = yaml.safe_load(file) if file is not None else yaml.safe_load(path.read_text())
    return require_mapping(doc, path)


def chain_ids(spec: Any, default: str = "A") -> list[str]:
    """The chain ids a ``sequences:`` entry's ``id:`` field names.

    YAML allows both spellings and inputs in the wild use both: a list (``id: [A, C]``) or a
    comma-separated scalar (``id: A,C``). Six readers and guards expanded that by hand with the
    same two-branch expression, and two of them handled only the list form -- so a
    ``cyclic: true`` chain written ``id: A,B`` was refused as one chain called "A,B". Missing
    ``id`` falls back to ``default``, which is the reader's convention for an unnamed chain
    ("A" for a polymer, "L" for a ligand).
    """
    ids = default if spec is None else spec
    if isinstance(ids, (list, tuple)):
        return [str(x).strip() for x in ids]
    return [c.strip() for c in str(ids).split(",")]
