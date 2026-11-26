"""Receptor structure parsing for DiffDock-L, from PDB or mmCIF.

Upstream reads the receptor with ProDy (`prody.parsePDB`), takes its C-alpha
subset, and indexes residues by ProDy's running residue index. tt-bio does not
depend on ProDy, so this module reproduces that selection on top of gemmi, which
reads mmCIF as well as PDB. The three ProDy behaviours that change the result and
are reproduced exactly:

* only the first model is read;
* an atom is kept when its altloc is blank or the first altloc seen for that
  atom name in that residue (ProDy's `altloc='A'` default, which keeps the
  alternate labelled A and otherwise the first one present);
* a residue becomes a receptor node when it has an atom named `CA` and its
  residue name is one of ProDy's 49 amino acids (`AMINO_ACID_RESNAMES`).

Residues come out in file order, which is the order upstream's
`sorted(set(pdb.ca.getResindices()))` produces, and is the order the receptor
node features, the k-NN graph and the per-chain ESM-2 embeddings are all indexed
in. Equality with ProDy 2.6.1 is tested over the 427-complex PoseBusters set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import AMINO_ACID_RESNAMES, RESNAME_TO_LETTER, aa_long2short, atom_order

#: Atoms per residue in the fixed 14-slot layout `atom_order` defines.
MAX_RESIDUE_ATOMS = 14


@dataclass(frozen=True)
class Receptor:
    """One parsed receptor, in the residue order the model consumes.

    ``coords`` is ``(n_res, 14, 3)`` with NaN in every slot the structure does
    not provide, matching upstream's `datasets.parse_chi.get_coords`.
    ``sequence`` is one letter per residue, ``chain_keys`` the segment+chain
    label upstream concatenates to separate chains.
    """

    sequence: str
    coords: np.ndarray
    chain_keys: np.ndarray
    resnames: tuple[str, ...]

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    def chain_sequences(self) -> tuple[list[str], np.ndarray]:
        """Per-chain sequences and the per-residue chain index.

        Chains are numbered by `np.unique` over the segment+chain labels, which
        is alphabetical and not file order. That is upstream's ordering
        (`datasets/process_mols.py:140`) and the ESM-2 embeddings are
        concatenated in it, so it has to be reproduced rather than tidied.
        """
        ids = np.unique(self.chain_keys)
        chain_ids = np.zeros(len(self.chain_keys), dtype=np.int64)
        sequences = []
        letters = np.asarray(list(self.sequence))
        for i, cid in enumerate(ids):
            sel = self.chain_keys == cid
            chain_ids[sel] = i
            sequences.append("".join(letters[sel]))
        return sequences, chain_ids


def _read_gemmi(path: Path):
    import gemmi

    st = gemmi.read_structure(str(path))
    if len(st) == 0:
        raise ValueError(f"{path} contains no model")
    # ProDy reads one model. gemmi exposes mmCIF auth_* names as the default,
    # which is what a PDB file carries, so no renaming is needed.
    st.setup_entities()
    return st[0]


def _residue_atoms(residue) -> dict:
    """The residue's atoms by name, with ProDy's altloc rule applied.

    ProDy's `altloc='A'` default keeps an atom whose altloc field is blank or
    `A` and drops the other alternates. A duplicate name after that filter keeps
    the first occurrence, which is what indexing ProDy's selection gives.
    """
    kept = {}
    for atom in residue:
        # gemmi stores "no alternate location" as NUL, not as a space.
        altloc = (atom.altloc or "").strip().strip("\x00")
        if altloc not in ("", "A"):
            continue
        kept.setdefault(atom.name.strip(), atom)
    return kept


def parse_receptor(path: str | Path) -> Receptor:
    """Parse a receptor file into the residue table the graph builder needs.

    Accepts anything gemmi reads: `.pdb`, `.cif`/`.mmcif`, and their `.gz`
    forms. Upstream accepts PDB only; mmCIF is a tt-bio extension (D-dfd-input-2),
    and the two give the same receptor for the same structure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"receptor file not found: {path}")
    model = _read_gemmi(path)

    letters, coords, chain_keys, resnames = [], [], [], []
    for chain in model:
        for residue in chain:
            resname = residue.name.strip()
            if resname not in AMINO_ACID_RESNAMES:
                continue
            by_name = _residue_atoms(residue)
            if "CA" not in by_name:
                continue
            slot = np.full((MAX_RESIDUE_ATOMS, 3), np.nan)
            names = atom_order[aa_long2short.get(resname, "X")]
            for j, name in enumerate(names):
                atom = by_name.get(name)
                if atom is not None:
                    slot[j] = (atom.pos.x, atom.pos.y, atom.pos.z)
            letters.append(RESNAME_TO_LETTER.get(resname, "X"))
            coords.append(slot)
            # Upstream's chain key is segment identifier + chain identifier
            # (PDB columns 73-76 and 22). mmCIF has no segid, so it contributes
            # nothing there, exactly as a PDB without SEGRES columns.
            chain_keys.append(f"{(residue.segment or '').strip()}{chain.name.strip()}")
            resnames.append(resname)

    if not letters:
        raise ValueError(f"{path} has no amino-acid residue with a CA atom")
    return Receptor(
        sequence="".join(letters),
        coords=np.asarray(coords),
        chain_keys=np.asarray(chain_keys),
        resnames=tuple(resnames),
    )
