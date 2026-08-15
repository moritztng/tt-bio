#!/usr/bin/env python3
"""Does the per-residue RDKit conformer work parallelise across threads?

If RDKit's EmbedMolecule releases the GIL, the 512 per-residue embeddings can run on a
thread pool and the host leg gets most of lever H's win WITHOUT changing any conformer,
which would make it bit-exact where the memo is not. If it does not release the GIL, that
whole route is dead and no seed plumbing is worth designing.

Host only, no device.
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tt_bio._vendor.openfold3.core.data.primitives.structure.query import (  # noqa: E402
    atom_array_from_ccd_code, processed_reference_molecule_from_atom_array)
from tt_bio._vendor.openfold3.core.data.resources.residues import (  # noqa: E402
    MOLECULE_TYPE_TO_LEAVING_ATOMS)
from tt_bio._vendor.openfold3.core.data.primitives.structure.labels import MoleculeType  # noqa: E402

N = 64
CODES = ["ALA", "GLY", "SER", "VAL", "LEU", "ILE", "THR", "ASN"] * (N // 8)
POLY = MoleculeType.PROTEIN
LEAVING = MOLECULE_TYPE_TO_LEAVING_ATOMS[POLY]

arrays = [atom_array_from_ccd_code(c, chain_id="A", res_id=i + 1, molecule_type=POLY)
          for i, c in enumerate(CODES)]


def one(i):
    return processed_reference_molecule_from_atom_array(arrays[i], atoms_to_mask=LEAVING)


t0 = time.perf_counter()
for i in range(N):
    one(i)
seq = time.perf_counter() - t0

for w in (4, 8, 16):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=w) as ex:
        list(ex.map(one, range(N)))
    par = time.perf_counter() - t0
    print(f"{N} residues: sequential {seq*1e3:8.1f} ms | {w:2d} threads {par*1e3:8.1f} ms "
          f"| {seq/par:5.2f}x", flush=True)
