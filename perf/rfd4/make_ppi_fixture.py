"""Turn a monomer design into a two-chain binder design with a conditioned target.

The binder metrics (`refolded_target_aligned_binder_rmsd`,
`refolded_target_backbone_aligned_binder_rmsd`) and the `ppi` pass criterion are the half of the
refolding config that only fires on a design with a chain of interest and a `C_CRD` target, and no
shipped example has both without unresolved atoms -- upstream's own aligner asserts on a NaN
coordinate (`utils/align.py:166-172`), so an input structure cannot be used as the design.

This splits a clean monomer at a residue boundary: the first part keeps its chain, the second
becomes chain B and is handed back to the model as a fixed target, i.e. every one of its atoms
carries its own coordinate in `_condition_coordinate_1_atom` with `CoordinateGroup` 0 (absolute).
That is what `sa_mask('C_CRD')` reads, and what makes chain A the unambiguous chain of interest.

    python3 perf/rfd4/make_ppi_fixture.py <monomer.cif> <out.cif> [--split 50]
"""

import argparse
from pathlib import Path

import numpy as np
from biotite.structure import get_residue_positions, residue_iter
from biotite.structure.io.pdbx import CIFCategory, CIFFile, get_structure, set_structure

CRDGRP_ABSOLUTE = 0
CRDGRP_UNKNOWN = -1


def split_chains(atoms, split_after):
    """Residues after `split_after` become chain B, moved clear of chain A so they do not clash."""
    out = atoms.copy()
    residue_index = np.zeros(out.array_length(), dtype=int)
    for i, residue in enumerate(residue_iter(out)):
        residue_index[np.isin(out.res_id, residue.res_id[0]) & (out.chain_id == residue.chain_id[0])] = i
    target = residue_index >= split_after
    out.chain_id = np.where(target, "B", "A")
    extent = np.nanmax(out.coord, axis=0) - np.nanmin(out.coord, axis=0)
    out.coord[target] += np.array([extent[0] + 8.0, 0.0, 0.0])
    return out, target


def write(atoms, target, path):
    """Write the design with the two conditioning annotations the C_CRD mask is read from."""
    cif = CIFFile()
    set_structure(cif, atoms)
    block = cif.block
    n = atoms.array_length()
    block["atom_site"]["condition_CoordinateGroup_1_atom"] = np.where(
        target, CRDGRP_ABSOLUTE, CRDGRP_UNKNOWN
    ).astype(str)
    idx = np.nonzero(target)[0]
    block["condition_coordinate_1_atom"] = CIFCategory(
        {
            "idx0": idx.astype(str),
            **{f"val{k}": np.array([f"{v:.3f}" for v in atoms.coord[idx, k]]) for k in range(3)},
        }
    )
    cif.write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", type=int, default=50, help="residues kept in chain A")
    args = ap.parse_args()

    atoms = get_structure(CIFFile.read(str(args.design)), model=1, include_bonds=True)
    assert np.isfinite(atoms.coord).all(), "a design with a NaN coordinate breaks upstream's aligner"
    design, target = split_chains(atoms, args.split)
    write(design, target, args.out)
    print(f"{args.out}: chain A {int((~target).sum())} atoms, chain B {int(target.sum())} conditioned")


if __name__ == "__main__":
    main()
