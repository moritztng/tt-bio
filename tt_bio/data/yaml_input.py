"""Read a YAML input file as a mapping, refuse anything else, and refuse keys nobody reads.

``yaml.safe_load`` returns ``None`` for an empty or comment-only file, and every reader
in this repo then reaches straight for a key. The user gets ``AttributeError: 'NoneType'
object has no attribute 'get'`` from inside a parser, with no mention of the file they
passed. ``main.py``'s rfd3 reader already had the answer -- check the type, name the file
-- and the fix was written there and nowhere else.

The one loader, so a reader cannot forget the check by writing ``yaml.safe_load``
directly. ``tt_bio/data/__init__.py`` is empty, so this costs no import weight to reach
from ``boltzgen`` or ``pxdesign``.

:func:`refuse_unread_keys` is the same idea one level up. A design spec that names a
field the model never reads costs a full run and comes back unconditioned, with nothing
to say the conditioning was dropped -- a mistyped ``hotspot:`` for ``hotspots:`` reads
exactly like a design that ignored its epitope. BoltzGen already refuses an unknown key
by name (``boltzgen/data/parse/schema.py``); this is that check for the readers that had
none, in one place so the three design front doors cannot drift apart on it.
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


def refuse_unread_keys(provided, *, honoured, unimplemented=(), what: str) -> None:
    """Raise ``ValueError`` for spec keys this port will not act on.

    ``provided`` is the user's mapping. ``honoured`` is every key that reaches the model.
    ``unimplemented`` maps a key the upstream schema defines but this port does not
    consume to ``(default, reason)``: a key set to its default asks for nothing and is
    allowed through, any other value is a request that would be silently dropped.
    Everything left over is a typo.
    """
    unimplemented = dict(unimplemented)
    asked = []
    for key, (default, reason) in unimplemented.items():
        if key in provided and provided[key] != default:
            asked.append(f"{key}={provided[key]!r} ({reason})")
    unknown = sorted(k for k in provided
                     if k not in honoured and k not in unimplemented)
    problems = []
    if asked:
        problems.append("this port does not implement " + ", ".join(sorted(asked)))
    if unknown:
        problems.append(f"unknown key(s) {unknown}; readable keys are {sorted(honoured)}")
    if problems:
        raise ValueError(f"{what}: " + "; ".join(problems)
                         + ". Remove them, or the run costs the same and ignores them.")


def refuse_unresolved(name: str, requested, available, *, what: str) -> None:
    """Raise ``ValueError`` when a residue selection names nothing in the structure.

    A selection that resolves to no residue produces the same input as no selection at
    all, so the run succeeds and the conditioning is simply absent. ``available`` is the
    numbering the selection is matched against, which is what the message has to show:
    the usual cause is author numbering given where label_seq is read.
    """
    have = set(available)
    missing = [r for r in requested if r not in have]
    if not missing:
        return
    lo, hi = (min(have), max(have)) if have else ("-", "-")
    raise ValueError(
        f"{what}: {name} {missing} name no residue (matched against label_seq "
        f"{lo}..{hi}, {len(have)} residues). Without them the design runs unconditioned.")
