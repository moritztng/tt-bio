"""Ligand-enclosure gate for an RFD3 motif-scaffolded design.

For every ligand atom, compute the distance to its nearest protein atom. A
defensible "designed active site" needs the bulk of the ligand buried against
the scaffolded protein, not floating free. The existing design_geometry_check.py
only validated protein backbone bond lengths + a single ligand atom's local
packing; this checks WHOLE-ligand enclosure against the protein surface.

Reference cloud = protein backbone (N/CA/C/O) of the protein chains. RFD3's
designed positions are UNK with un-committed CB/virtual sidechain geometry, so
sidechains are excluded (same rationale as design_geometry_check.py); the
backbone fold is what encloses the ligand at this stage of the pipeline. An
optional --use-all-protein-atoms mode adds every protein-chain heavy atom
present in the CIF (motif sidechains are real) for a more permissive second
opinion.

Per ligand residue (by chain_id + res_id) reports: max and mean
atom-to-nearest-protein distance, fraction of ligand atoms within 4 A and 5 A,
and the worst-enclosed atoms. Exit code 0 if the gate PASSES, 1 if it FAILS.

Gate (default, backbone reference):
  PASS  if  mean <= MEAN_MAX  AND  frac_within_5 >= FRAC5  AND  max <= MAX_MAX
  with defaults MEAN_MAX=4.0 A, FRAC5=0.80, MAX_MAX=7.0 A.
  (max is allowed to be looser than mean: a single solvent-exposed tail atom is
  fine; a floating bulk is not. The mean + frac_within_5 pair catches the latter.)

Usage:
  python3 scripts/rfd3_port/ligand_enclosure_check.py <design.cif>       [--protein-chains A] [--ligand-chains B]       [--mean-max 4.0] [--frac5 0.8] [--max-max 7.0] [--use-all-protein-atoms]
"""
import argparse
import sys
import numpy as np
import biotite.structure.io.pdbx as pdbx
from scipy.spatial import cKDTree

BB_NAMES = np.array(["N", "CA", "C", "O"])


def load_atoms(path):
    cf = pdbx.CIFFile.read(path)
    return pdbx.get_structure(cf, model=1)


def enclosure(arr, protein_chains, ligand_chains, use_all_protein_atoms=False):
    if use_all_protein_atoms:
        is_ref = np.isin(arr.chain_id, list(protein_chains))
    else:
        is_ref = np.isin(arr.atom_name, ["N", "CA", "C", "O"]) & np.isin(
            arr.chain_id, list(protein_chains))
    is_lig = np.isin(arr.chain_id, list(ligand_chains))
    ref = arr[is_ref]
    lig = arr[is_lig]
    if len(ref) == 0:
        raise ValueError(f"no protein reference atoms found (chains={protein_chains})")
    if len(lig) == 0:
        raise ValueError(f"no ligand atoms found (chains={ligand_chains})")
    tree = cKDTree(ref.coord)
    dists, _ = tree.query(lig.coord, k=1)
    # group by (chain_id, res_id) -> per-residue stats
    keys = list(zip(lig.chain_id.tolist(), lig.res_id.tolist()))
    residues = {}
    order = []
    for i, k in enumerate(keys):
        if k not in residues:
            residues[k] = []
            order.append(k)
        residues[k].append((lig.atom_name[i], float(dists[i])))
    per_res = {}
    for k in order:
        ds = [d for _, d in residues[k]]
        arr_d = np.array(ds)
        worst = sorted(residues[k], key=lambda t: -t[1])[:5]
        per_res[k] = {
            "n_atoms": len(ds),
            "max": round(float(arr_d.max()), 2),
            "mean": round(float(arr_d.mean()), 2),
            "frac_within_4": round(float((arr_d <= 4.0).mean()), 3),
            "frac_within_5": round(float((arr_d <= 5.0).mean()), 3),
            "worst_atoms": [(n, round(d, 2)) for n, d in worst],
        }
    overall = {
        "n_ligand_atoms": int(len(dists)),
        "n_ref_atoms": int(len(ref)),
        "max": round(float(dists.max()), 2),
        "mean": round(float(dists.mean()), 2),
        "frac_within_4": round(float((dists <= 4.0).mean()), 3),
        "frac_within_5": round(float((dists <= 5.0).mean()), 3),
        "per_residue": per_res,
    }
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("--protein-chains", default="A")
    ap.add_argument("--ligand-chains", default="B")
    ap.add_argument("--mean-max", type=float, default=4.0)
    ap.add_argument("--frac5", type=float, default=0.8)
    ap.add_argument("--max-max", type=float, default=7.0)
    ap.add_argument("--use-all-protein-atoms", action="store_true")
    args = ap.parse_args()
    pc = tuple(args.protein_chains.split(","))
    lc = tuple(args.ligand_chains.split(","))
    arr = load_atoms(args.cif)
    r = enclosure(arr, pc, lc, use_all_protein_atoms=args.use_all_protein_atoms)
    print(f"ligand enclosure: {r['n_ligand_atoms']} ligand atoms vs {r['n_ref_atoms']} protein ref atoms")
    print(f"  overall: max={r['max']} A  mean={r['mean']} A  "
          f"frac<=4A={r['frac_within_4']}  frac<=5A={r['frac_within_5']}")
    for k, v in r["per_residue"].items():
        print(f"  residue {k[0]}{k[1]}: n={v['n_atoms']} max={v['max']} mean={v['mean']} "
              f"frac<=4A={v['frac_within_4']} frac<=5A={v['frac_within_5']}")
        print(f"      worst: {v['worst_atoms']}")
    passed = (r["mean"] <= args.mean_max and r["frac_within_5"] >= args.frac5
              and r["max"] <= args.max_max)
    mode = "all-protein" if args.use_all_protein_atoms else "backbone"
    print(f"GATE ({mode} ref, mean<={args.mean_max} & frac5>={args.frac5} & max<={args.max_max}): "
          f"{'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
