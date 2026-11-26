"""Does torch_cluster's neighbour cap bite on real DiffDock ligands?

The static-graph port replaces ``radius_graph(lig.pos, 5)`` and ``radius(lig.pos, bond_pos,
5)`` with a complete edge set plus a distance mask. That is only equivalent if upstream's
calls return *every* pair inside the cutoff, and both run at torch_cluster's default
``max_num_neighbors=32``: a ligand atom with more than 32 neighbours inside 5 A would have
its edge list silently truncated upstream, and the mask would then produce edges upstream
does not have.

So count, over the whole PoseBusters benchmark set (428 ligands), the largest number of
heavy-atom neighbours any atom has inside 5 A -- on the crystal pose and on the RDKit start
conformer DiffDock actually docks -- and the same for bond-centre -> atom.

    ~/scratch/dfd/venv-ref/bin/python scripts/diffdock_port/probe_radius_truncation.py \
        --out /tmp/dfd-radius-truncation.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

PB = Path.home() / "scratch/dfd/pb/posebusters_benchmark_set"
CUTOFF = 5.0
CAP = 32


def heavy_coords_and_bonds(path):
    mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=True)
    if mol is None or mol.GetNumConformers() == 0:
        return None, None
    pos = mol.GetConformer().GetPositions()
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    index = {old: new for new, old in enumerate(keep)}
    bonds = [(index[b.GetBeginAtomIdx()], index[b.GetEndAtomIdx()])
             for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in index and b.GetEndAtomIdx() in index]
    return pos[keep], np.asarray(bonds, dtype=int).reshape(-1, 2)


def max_neighbours(pos, other=None):
    """Largest row count of pairs inside the cutoff (self-pairs excluded when other is None)."""
    ref = pos if other is None else other
    d = np.linalg.norm(ref[None, :, :] - pos[:, None, :], axis=-1)
    inside = d <= CUTOFF
    if other is None:
        np.fill_diagonal(inside, False)
    return int(inside.sum(axis=1).max()) if inside.size else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")

    rows, skipped = [], []
    for d in sorted(PB.iterdir()):
        if not d.is_dir():
            continue
        cid = d.name
        rec = {"id": cid}
        for tag, name in (("crystal", f"{cid}_ligand.sdf"),
                          ("start_conf", f"{cid}_ligand_start_conf.sdf")):
            pos, bonds = heavy_coords_and_bonds(d / name)
            if pos is None:
                skipped.append(f"{cid}/{tag}")
                continue
            rec[f"{tag}_atoms"] = len(pos)
            rec[f"{tag}_max_atom_neighbours"] = max_neighbours(pos)
            if len(bonds):
                centres = (pos[bonds[:, 0]] + pos[bonds[:, 1]]) / 2
                rec[f"{tag}_max_bond_neighbours"] = max_neighbours(centres, other=pos)
        rows.append(rec)

    def top(key):
        have = [r for r in rows if key in r]
        have.sort(key=lambda r: -r[key])
        return [{"id": r["id"], key: r[key], "atoms": r[key.split("_max")[0] + "_atoms"]}
                for r in have[:5]]

    summary = {
        "n_ligands": len(rows), "cutoff_a": CUTOFF, "torch_cluster_cap": CAP,
        "skipped": skipped,
        "worst": {k: top(k) for k in ("crystal_max_atom_neighbours",
                                      "start_conf_max_atom_neighbours",
                                      "crystal_max_bond_neighbours",
                                      "start_conf_max_bond_neighbours")},
    }
    summary["cap_reached"] = {
        k: [r["id"] for r in rows if r.get(k, 0) >= CAP]
        for k in ("crystal_max_atom_neighbours", "start_conf_max_atom_neighbours",
                  "crystal_max_bond_neighbours", "start_conf_max_bond_neighbours")}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
