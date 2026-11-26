"""Capture upstream DiffDock-L input-pipeline goldens.

Run with the upstream reference venv, not tt-bio's:

    ~/scratch/dfd/venv-ref/bin/python tests/diffdock_input_capture.py \
        --upstream ~/scratch/dfd/upstream \
        --complexes ~/scratch/dfd/data/posebusters_dl/sel \
        --out ~/scratch/dfd/goldens-input

It writes one .npz per complex per graph kind (``score``: all_atoms=False,
``conf``: all_atoms=True), holding exactly what ``InferenceDataset.get`` puts on
the HeteroData, minus the ESM embedding (``lm_embeddings=None``), which this row
does not own. The ligand conformer is embedded with a fixed ETKDG seed so the
positions are reproducible on both sides; upstream leaves the seed at -1.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--complexes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated complex ids")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    up = Path(args.upstream).expanduser()
    sys.path.insert(0, str(up))
    os.chdir(up)

    from rdkit import Chem
    from rdkit.Chem import AddHs, AllChem, MolFromSmiles, RemoveHs
    from torch_geometric.data import HeteroData

    from datasets.process_mols import (
        get_lig_graph_with_matching,
        moad_extract_receptor_structure,
        read_molecule,
    )

    seed = args.seed

    def embed(mol):
        ps = AllChem.ETKDGv2()
        ps.randomSeed = seed
        assert AllChem.EmbedMolecule(mol, ps) != -1, "ETKDG failed at the fixed seed"

    sel = Path(args.complexes).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.name for p in sel.iterdir() if p.is_dir())
    if args.only:
        want = set(args.only.split(","))
        ids = [i for i in ids if i in want]

    for cid in ids:
        pdb = sel / cid / f"{cid}_protein.pdb"
        sdf = sel / cid / f"{cid}_ligand.sdf"
        if not pdb.exists() or not sdf.exists():
            print(f"skip {cid}: missing input")
            continue
        for kind, all_atoms in (("score", False), ("conf", True)):
            dst = out / f"{cid}.{kind}.npz"
            if dst.exists():
                print(f"have {dst.name}")
                continue
            try:
                rec = {}
                g = HeteroData()
                # ligand, exactly InferenceDataset.get's file branch
                mol = read_molecule(str(sdf), remove_hs=False, sanitize=True)
                assert mol is not None
                mol.RemoveAllConformers()
                mol = AddHs(mol)
                embed(mol)
                get_lig_graph_with_matching(
                    mol, g, popsize=None, maxiter=None, matching=False,
                    keep_original=False, num_conformers=1, remove_hs=True)
                moad_extract_receptor_structure(
                    path=str(pdb), complex_graph=g,
                    neighbor_cutoff=15.0, max_neighbors=24,
                    lm_embeddings=None, knn_only_graph=True,
                    all_atoms=all_atoms, atom_cutoff=5, atom_max_neighbors=8)

                rec["lig_x"] = g["ligand"].x.numpy()
                rec["lig_pos"] = g["ligand"].pos.numpy()
                rec["lig_edge_index"] = g["ligand", "lig_bond", "ligand"].edge_index.numpy()
                rec["lig_edge_attr"] = g["ligand", "lig_bond", "ligand"].edge_attr.numpy()
                rec["lig_edge_mask"] = g["ligand"].edge_mask.numpy()
                rec["lig_mask_rotate"] = np.asarray(g["ligand"].mask_rotate)
                rec["rec_x"] = g["receptor"].x.numpy()
                rec["rec_pos"] = g["receptor"].pos.numpy()
                rec["rec_side_chain_vecs"] = g["receptor"].side_chain_vecs.numpy()
                rec["rec_edge_index"] = g["receptor", "rec_contact", "receptor"].edge_index.numpy()
                rec["rec_chain_ids"] = g["receptor"].chain_ids.numpy()
                rec["rec_sequence"] = np.asarray(g["receptor"].sequence, dtype=object)
                if all_atoms:
                    rec["atom_x"] = g["atom"].x.numpy()
                    rec["atom_pos"] = g["atom"].pos.numpy()
                    rec["atom_edge_index"] = g["atom", "atom_contact", "atom"].edge_index.numpy()
                    rec["atom_rec_edge_index"] = g["atom", "atom_rec_contact", "receptor"].edge_index.numpy()
                rec["smiles"] = np.asarray(Chem.MolToSmiles(RemoveHs(mol)))
                rec["seed"] = np.asarray(seed)
                np.savez_compressed(dst, **rec)
                print(f"wrote {dst.name}  rec_nodes={len(rec['rec_pos'])} "
                      f"lig_atoms={len(rec['lig_x'])} rec_edges={rec['rec_edge_index'].shape[1]}")
            except Exception as e:  # noqa: BLE001 - a capture failure is per complex
                print(f"FAIL {cid} {kind}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
