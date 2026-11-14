"""biotite's CIF text, byte for byte, with the looped categories built as arrays.

biotite's ``CIFCategory._serialize_looped`` escapes and pads every cell in Python: an atom_site of 2,529 atoms is
~51,000 calls to ``_escape`` and as many ``ljust``, 68 ms a file. A Mol-1 fold writes ten (five tt-bio structures,
five in upstream's layout), so 0.68 of its 0.70 s write at 245 tokens was that loop, all of it after the chip's
last op (mol1-bh-256, state/mol1/evidence/bh/b256.md).

Here a category's cells are scanned once as a code-point matrix for anything ``_escape`` would touch (empty, a
leading ``_``, a quote, a blank); only those go through biotite's own ``_escape``, and the rows are padded and
joined as one matrix. A category holding a multiline value, or a whitespace character this scan does not model,
is serialised by biotite itself, so the output is biotite's in every case (tests/test_cif_text.py compares the two
on the structures the folds write).
"""
from __future__ import annotations

import numpy as np
from biotite.structure.io.pdbx import cif as _cif

_QUOTE = np.array([ord(c) for c in " \t'\""], dtype=np.uint32)
#: whitespace str.strip removes that a value could start or end with; any of these and biotite does the category
_ODD = np.array([ord(c) for c in "\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0"], dtype=np.uint32)


def _codes(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a.astype(str, copy=False))
    n = max(a.dtype.itemsize // 4, 1)
    m = np.ascontiguousarray(a.astype(f"<U{n}")).view(np.uint32).reshape(len(a), n)
    # biotite pads to the longest cell, not to the dtype's width
    used = np.flatnonzero(m.any(axis=0))
    return np.ascontiguousarray(m[:, :max(used[-1] + 1 if len(used) else 0, 1)])


def _looped(category: "_cif.CIFCategory", name: str) -> str | None:
    """The text ``category.serialize()`` returns for a looped category, or None to leave it to biotite."""
    columns = []
    for column in category.values():
        m = _codes(column.as_array(str))
        if np.isin(m, _ODD).any() or (m > 0x7f).any() and _unicode_space(m):
            return None
        lead = m[:, 0]
        hit = (lead == 0) | (lead == ord("_")) | np.isin(m, _QUOTE).any(axis=1)
        if hit.any():
            cells = m.view(f"<U{m.shape[1]}").ravel().tolist()
            for i in np.flatnonzero(hit).tolist():
                cells[i] = _cif._escape(cells[i])
            if any("\n" in cells[i] for i in np.flatnonzero(hit).tolist()):
                return None
            m = _codes(np.array(cells))
        columns.append(m)
    # each column padded to its widest cell + 1 (biotite's column_n_chars), NUL padding made blanks
    rows = np.concatenate([np.pad(m, ((0, 0), (0, 1))) for m in columns], axis=1)
    rows[rows == 0] = ord(" ")
    lines = [r.rstrip() for r in rows.view(f"<U{rows.shape[1]}").ravel().tolist()]
    keys = ["_" + name + "." + key + " " for key in category.keys()]
    return "\n".join(["loop_"] + keys + lines + [""])


def _unicode_space(m: np.ndarray) -> bool:
    return any(chr(c).isspace() for c in np.unique(m[m > 0x7f]).tolist())


def serialize(cif_file: "_cif.CIFFile") -> str:
    """``cif_file.serialize()``, the same text."""
    out = []
    for block_name, block in cif_file._blocks.items():
        if isinstance(block, str):
            out.append(block)
            continue
        block.name = block_name
        out.append("data_" + block_name + "\n#\n")
        for name, category in block._categories.items():
            if isinstance(category, str):
                out.append(category)
                continue
            category.name = name
            text = None
            if category._columns and len({len(c) for c in category.values()}) == 1 and len(next(iter(category.values()))) > 1:
                text = _looped(category, name)
            out.append(category.serialize() if text is None else text)
            out.append("#\n")
    out.append("")
    return "".join(out)


def write(cif_file: "_cif.CIFFile", path) -> None:
    """``cif_file.write(path)`` for a path, through :func:`serialize`."""
    with open(path, "w") as f:
        f.write(serialize(cif_file))
