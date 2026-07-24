"""Basic geometry/clash sanity check for an RFD3-generated structure (mmCIF).

Not a re-implementation of a proper Ramachandran/MolProbity-style validator --
just the two cheapest, most damning checks a real structural biologist would
run first: are the backbone bond lengths/peptide bonds chemically plausible,
and do any two heavy atoms (protein backbone or ligand) interpenetrate.
Restricted to REAL backbone atoms (N/CA/C/O) + ligand atoms; RFD3's CB and
virtual sidechain atoms (V0..V8) are excluded because for tokens whose amino
acid identity is still undetermined (`restype`=UNK, the common case for any
truly *designed*, non-motif position) they are not yet a committed geometry
-- see state/tt-bio-rfdiffusion3-port-p1.md and rfd3-release-video's own
notes (a real CA-CB distance check on those atoms is meaningless: they sit
~0.05-0.1A from CA, not the ~1.53A a real bond would need).

Usage:
    python3 scripts/rfd3_port/design_geometry_check.py <path/to/design.cif>
"""
import sys

import numpy as np
import biotite.structure.io.pdbx as pdbx
from scipy.spatial import cKDTree

BOND_REF = {("N", "CA"): 1.46, ("CA", "C"): 1.52, ("C", "O"): 1.23}
PEPTIDE_REF = 1.33
BOND_TOL = 0.4
CLASH_R = 1.6


def check(path, protein_chains=("A", "B"), ligand_chains=("C", "D")):
    cf = pdbx.CIFFile.read(path)
    arr = pdbx.get_structure(cf, model=1)

    is_bb = np.isin(arr.atom_name, ["N", "CA", "C", "O"]) & np.isin(arr.chain_id, protein_chains)
    is_lig = np.isin(arr.chain_id, ligand_chains)
    core = arr[is_bb | is_lig]

    bad_bonds = []
    n_bonds_checked = 0
    for chain in protein_chains:
        c = core[core.chain_id == chain]
        resids = sorted(set(c.res_id.tolist()))
        by_res = {r: {} for r in resids}
        for i in range(len(c)):
            by_res[int(c.res_id[i])][c.atom_name[i]] = c.coord[i]
        for r in resids:
            atoms = by_res[r]
            for (a, b), ref in BOND_REF.items():
                if a in atoms and b in atoms:
                    n_bonds_checked += 1
                    d = float(np.linalg.norm(atoms[a] - atoms[b]))
                    if abs(d - ref) > BOND_TOL:
                        bad_bonds.append((chain, r, a, b, round(d, 2), ref))
            if r + 1 in by_res and "C" in atoms and "N" in by_res[r + 1]:
                n_bonds_checked += 1
                d = float(np.linalg.norm(atoms["C"] - by_res[r + 1]["N"]))
                if abs(d - PEPTIDE_REF) > BOND_TOL:
                    bad_bonds.append((chain, r, "C", "N(+1)", round(d, 2), PEPTIDE_REF))

    coords, chain_id, res_id, atom_name = core.coord, core.chain_id, core.res_id, core.atom_name
    tree = cKDTree(coords)
    clashes = []
    for i, j in tree.query_pairs(r=CLASH_R):
        if chain_id[i] == chain_id[j] and res_id[i] == res_id[j]:
            continue
        if chain_id[i] == chain_id[j] and abs(int(res_id[i]) - int(res_id[j])) == 1 and \
           {atom_name[i], atom_name[j]} == {"C", "N"}:
            continue
        d = float(np.linalg.norm(coords[i] - coords[j]))
        clashes.append((chain_id[i], int(res_id[i]), atom_name[i], chain_id[j], int(res_id[j]), atom_name[j], round(d, 2)))

    result = {
        "n_backbone_bonds_checked": n_bonds_checked,
        "n_bond_outliers": len(bad_bonds),
        "bond_outliers_sample": bad_bonds[:15],
        "n_clashes": len(clashes),
        "clashes_sample": clashes[:15],
    }
    if len(protein_chains) >= 2:
        ca_a = core.coord[(core.chain_id == protein_chains[0]) & (core.atom_name == "CA")]
        ca_b = core.coord[(core.chain_id == protein_chains[1]) & (core.atom_name == "CA")]
        if len(ca_a) and len(ca_b):
            d_ab = np.linalg.norm(ca_a[:, None, :] - ca_b[None, :, :], axis=-1)
            result["chainA_chainB_min_CA_dist"] = round(float(d_ab.min()), 2)
            result["chainA_chainB_mean_CA_dist"] = round(float(d_ab.mean()), 2)
    return result


if __name__ == "__main__":
    r = check(sys.argv[1])
    print(f"backbone bonds checked: {r['n_backbone_bonds_checked']}, "
          f"outliers (>{BOND_TOL}A from ideal): {r['n_bond_outliers']}")
    for b in r["bond_outliers_sample"]:
        print("  ", b)
    print(f"clashes (<{CLASH_R}A, excl. bonded/same-residue): {r['n_clashes']}")
    for c in r["clashes_sample"]:
        print("  ", c)
    if "chainA_chainB_min_CA_dist" in r:
        print(f"chain A/B CA-CA distance: min={r['chainA_chainB_min_CA_dist']}A "
              f"mean={r['chainA_chainB_mean_CA_dist']}A")
