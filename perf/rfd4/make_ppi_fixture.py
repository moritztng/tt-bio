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
    python3 perf/rfd4/make_ppi_fixture.py <monomer.cif> <out.cif> --homodimer

`--homodimer` instead copies the whole monomer into chain B, moved clear and conditioned: two
identical subunits, the case upstream's subunit matching exists for.
"""

import argparse
from pathlib import Path

import numpy as np
from biotite.structure import residue_iter
from biotite.structure.io.pdbx import CIFCategory, CIFFile, get_structure, set_structure

CRDGRP_ABSOLUTE = 0
CRDGRP_UNKNOWN = -1


def split_chains(atoms, split_after, move_clear=True):
    """Residues after `split_after` become chain B, by default moved clear of chain A.

    `move_clear=False` keeps the two halves where the fold put them, in contact, which is what a
    hotspot fixture needs: a hotspot the binder never comes near only tests the zero branch."""
    out = atoms.copy()
    residue_index = np.zeros(out.array_length(), dtype=int)
    for i, residue in enumerate(residue_iter(out)):
        residue_index[np.isin(out.res_id, residue.res_id[0]) & (out.chain_id == residue.chain_id[0])] = i
    target = residue_index >= split_after
    out.chain_id = np.where(target, "B", "A")
    if move_clear:
        extent = np.nanmax(out.coord, axis=0) - np.nanmin(out.coord, axis=0)
        out.coord[target] += np.array([extent[0] + 8.0, 0.0, 0.0])
    return out, target, residue_index


def homodimer(atoms):
    """The monomer as chain A and an identical copy, moved clear, as chain B."""
    from biotite.structure import concatenate

    first, second = atoms.copy(), atoms.copy()
    first.chain_id[:] = "A"
    second.chain_id[:] = "B"
    extent = np.nanmax(atoms.coord, axis=0) - np.nanmin(atoms.coord, axis=0)
    second.coord += np.array([extent[0] + 8.0, 0.0, 0.0])
    out = concatenate([first, second])
    return out, out.chain_id == "B"


def write(atoms, target, path, hotspot=None):
    """Write the design with the conditioning annotations the C_CRD (and C_HOT) masks read."""
    cif = CIFFile()
    set_structure(cif, atoms)
    block = cif.block
    if hotspot is not None:
        block["atom_site"]["condition_hotspot_1_atom"] = np.where(hotspot, "True", "False")
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
    ap.add_argument("--in-contact", action="store_true", help="leave chain B where the fold put it")
    ap.add_argument("--homodimer", action="store_true", help="chain B is a copy of the whole monomer")
    ap.add_argument("--hotspots", default="", help="comma-separated 0-based residue indices on chain B")
    args = ap.parse_args()

    atoms = get_structure(CIFFile.read(str(args.design)), model=1, include_bonds=True)
    assert np.isfinite(atoms.coord).all(), "a design with a NaN coordinate breaks upstream's aligner"
    if args.homodimer:
        design, target = homodimer(atoms)
        write(design, target, args.out)
        print(f"{args.out}: chains A and B, {int(target.sum())} atoms each, B conditioned")
        return
    design, target, residue_index = split_chains(atoms, args.split, move_clear=not args.in_contact)
    hotspot = None
    if args.hotspots:
        hotspot = np.isin(residue_index, [int(i) for i in args.hotspots.split(",")]) & target
    write(design, target, args.out, hotspot)
    print(f"{args.out}: chain A {int((~target).sum())} atoms, chain B {int(target.sum())} conditioned")


if __name__ == "__main__":
    main()
