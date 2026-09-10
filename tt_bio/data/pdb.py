"""Fitting a structure into the PDB format's fixed-width columns.

PDB gives the chain id one column and the residue name three; mmCIF accepts a name of any
length. So a chain called ``sample_03`` or a five-character CCD code is legal mmCIF and
unwritable as PDB, and every tt-bio writer that emits PDB has to answer that. They answer it
here, once: relabel to something the columns hold, and record what was relabelled in a
``REMARK 999`` block so the file still carries the names the user submitted.

Before this module the three PDB writers disagreed. The biotite round-trip raised
``BadStructureError`` after the fold was already paid for; the two hand-rolled columnar
writers formatted a long id with ``{tag:>1}``, which does not truncate, so every column after
it shifted and the file was silently wrong.
"""
import string

#: Single-character chain ids, in the order they are handed out. The same 62 the ESM stack
#: uses (``SINGLE_LETTER_CHAIN_IDS``). A 63rd chain cannot be expressed in PDB at all, and a
#: two-character id is not a legal PDB chain id -- biotite refuses it, and the columnar
#: writers shift every field after it.
CHAIN_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits

#: Columns 18-20 of an ATOM record.
RES_NAME_WIDTH = 3


def chain_id_at(index: int) -> str:
    """The ``index``-th PDB chain id, or a readable error past what PDB can express."""
    if index >= len(CHAIN_ALPHABET):
        raise ValueError(
            f"chain {index + 1} does not fit a PDB: the chain column is one character wide, "
            f"so a PDB holds at most {len(CHAIN_ALPHABET)} chains. Write mmCIF instead "
            f"(output_format: cif).")
    return CHAIN_ALPHABET[index]


def chain_map(names) -> dict[str, str]:
    """Original chain name -> the single character it is written as, first-appearance order.

    Empty when every name is already one character, so a structure that was always PDB-legal
    comes out byte-identical and carries no remark.
    """
    seen = list(dict.fromkeys(str(n) for n in names))
    if all(len(n) == 1 for n in seen):
        return {}
    return {name: chain_id_at(i) for i, name in enumerate(seen)}


def res_name_map(names) -> dict[str, str]:
    """Residue name -> its first three characters, for the names that do not fit.

    1512 of the 45227 components in the CCD cache have five-character codes, so this is a
    live path for any ``ccd:`` ligand or ``modifications:`` residue, not a theoretical one.
    Truncating keeps the code recognisable and the remark keeps it exact. Two codes sharing a
    prefix land on the same three characters, which costs nothing structurally: a residue is
    identified by its chain and number, not by its name.
    """
    return {name: name[:RES_NAME_WIDTH]
            for name in dict.fromkeys(str(n) for n in names)
            if len(name) > RES_NAME_WIDTH}


def remarks(chains=None, residues=None) -> list[str]:
    """``REMARK 999`` lines recording every rename. Empty when nothing was renamed."""
    lines = [f"REMARK 999 CHAIN {written} IS {original}"
             for original, written in (chains or {}).items()]
    lines += [f"REMARK 999 RESIDUE {written} IS {original}"
              for original, written in (residues or {}).items()]
    if lines:
        lines.insert(0, "REMARK 999 NAMES RELABELLED IN THIS PDB, WITH THEIR ORIGINALS")
    return lines


def write_atom_array(arr, path) -> None:
    """Write a biotite ``AtomArray`` as a PDB, relabelling whatever does not fit the columns.

    The array is copied before relabelling, so the caller's annotations are untouched.
    """
    import biotite.structure.io.pdb as _pdb
    import numpy as np

    chain_ids = chain_map(arr.chain_id)
    res_names = res_name_map(arr.res_name)
    if chain_ids or res_names:
        arr = arr.copy()
        if chain_ids:
            arr.chain_id = np.array([chain_ids[str(c)] for c in arr.chain_id])
        if res_names:
            arr.res_name = np.array([res_names.get(str(r), str(r)) for r in arr.res_name])
    pf = _pdb.PDBFile()
    pf.set_structure(arr)
    pf.lines[0:0] = remarks(chain_ids, res_names)
    pf.write(str(path))
