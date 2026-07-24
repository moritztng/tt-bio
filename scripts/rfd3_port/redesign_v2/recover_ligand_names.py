"""Recover real CCD atom names for RFD3's generic-element ligand output.

RFD3's CIF writer names every atom by its element symbol (P, O, C, N, ...)
regardless of chemical identity, so BoltzGen's real-chemistry parser can't
recognize a designed structure's ligand as NAI/ACT. But RFD3 keeps ligand
atoms in the SAME ORDER they were read from the input motif-scaffolding
fixture -- verified here by comparing the element sequence of design.cif's
ligand chain against the original PDB 1MG5 HETATM records: both residues
(NAI: 44 atoms, ACT: 4 atoms) match element-for-element in order. So the
real atom names can be recovered by direct positional correspondence, no
geometry matching needed.

Usage:
    python3 recover_ligand_names.py <design.cif> <chainA_only.pdb> <input_pdb> <out.pdb>
"""
import sys

import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb_io

NAI_ATOMS = [
    "PA", "O1A", "O2A", "O5B", "C5B", "C4B", "O4B", "C3B", "O3B", "C2B", "O2B",
    "C1B", "N9A", "C8A", "N7A", "C5A", "C6A", "N6A", "N1A", "C2A", "N3A", "C4A",
    "O3", "PN", "O1N", "O2N", "O5D", "C5D", "C4D", "O4D", "C3D", "O3D", "C2D",
    "O2D", "C1D", "N1N", "C2N", "C3N", "C7N", "O7N", "N7N", "C4N", "C5N", "C6N",
]
ACT_ATOMS = ["C", "O", "OXT", "CH3"]


def recover(design_cif, chainA_pdb, input_pdb, out_pdb):
    cf = pdbx.CIFFile.read(design_cif)
    arr = pdbx.get_structure(cf, model=1)
    lig = arr[arr.chain_id == "B"]

    res_ids = sorted(set(lig.res_id.tolist()))
    assert len(res_ids) == 2, f"expected 2 ligand residues, got {res_ids}"
    r0, r1 = res_ids
    n0 = int((lig.res_id == r0).sum())
    n1 = int((lig.res_id == r1).sum())
    assert n0 == 44 and n1 == 4, f"unexpected ligand residue sizes: {n0}, {n1}"

    elem0 = lig.element[lig.res_id == r0].tolist()
    elem1 = lig.element[lig.res_id == r1].tolist()
    pdb_elem_nai = [a[0] if a[0] != "P" else "P" for a in NAI_ATOMS]
    # element is first letter of the CCD name except 2-letter names (none here)
    for name, e in zip(NAI_ATOMS, elem0):
        expect = name[0]
        assert e.upper() == expect.upper(), f"NAI element mismatch: {name} vs {e}"
    for name, e in zip(ACT_ATOMS, elem1):
        expect = name[0]
        assert e.upper() == expect.upper(), f"ACT element mismatch: {name} vs {e}"

    # Reassign names/res_name in-place on the ligand subarray, in file order.
    idx0 = [i for i in range(len(lig)) if lig.res_id[i] == r0]
    idx1 = [i for i in range(len(lig)) if lig.res_id[i] == r1]
    for j, i in enumerate(idx0):
        lig.atom_name[i] = NAI_ATOMS[j]
        lig.res_name[i] = "NAI"
    for j, i in enumerate(idx1):
        lig.atom_name[i] = ACT_ATOMS[j]
        lig.res_name[i] = "ACT"

    with open(chainA_pdb) as f:
        protein_lines = [l for l in f if l.startswith(("ATOM", "HETATM"))]

    het_lines = []
    serial = len(protein_lines) + 1
    for i in range(len(lig)):
        name = lig.atom_name[i]
        resn = lig.res_name[i]
        resid = int(lig.res_id[i]) + 850  # keep original-style numbering, arbitrary but unique
        x, y, z = lig.coord[i]
        elem = lig.element[i]
        line = (
            f"HETATM{serial:5d} {name:<4s} {resn:>3s} B{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
        )
        het_lines.append(line)
        serial += 1

    with open(out_pdb, "w") as f:
        f.writelines(protein_lines)
        f.write("TER\n")
        f.writelines(het_lines)
        f.write("END\n")

    print(f"wrote {out_pdb}: {len(protein_lines)} protein atoms + {len(het_lines)} ligand atoms")
    print("NAI atoms:", NAI_ATOMS[:5], "...")
    print("ACT atoms:", ACT_ATOMS)


if __name__ == "__main__":
    recover(*sys.argv[1:5])
