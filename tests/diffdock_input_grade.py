"""Grade tt-bio's DiffDock input pipeline against upstream over a whole set.

    python3 tests/diffdock_input_grade.py \
        --goldens ~/scratch/dfd/goldens-input \
        --complexes ~/scratch/dfd/input-sel \
        --json state/dfd/evidence/input-golden-match.json

The committed pytest covers three complexes on every checkout. This grades every
complex a capture produced, and is what state/dfd/INPUT.md's agreement numbers
come from. Each field is reported as the number of complexes where it matched
exactly; edge indices are reported both exactly and as edge sets, because two
nodes exactly equidistant from a third land in either order and the model
scatter-sums over edges.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

#: Fields graded with exact equality.
EXACT = ("rec_x", "rec_pos", "rec_chain_ids", "atom_x", "atom_pos", "atom_rec_edge_index",
         "lig_x", "lig_edge_index", "lig_edge_attr", "lig_edge_mask", "lig_mask_rotate")
#: Fields graded to a float tolerance.
CLOSE = {"rec_side_chain_vecs": 1e-4}
#: Edge indices, graded exactly and as sets.
EDGES = ("rec_edge_index", "atom_edge_index")


def grade(goldens: Path, complexes: Path, only: list[str] | None = None) -> dict:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rdkit.Chem import AddHs

    from tt_bio.diffdock.ligand import build_ligand_graph, generate_conformer, read_molecule
    from tt_bio.diffdock.receptor import build_receptor_graph
    from tt_bio.diffdock.structure import parse_receptor

    passed = defaultdict(int)
    total = defaultdict(int)
    errors: list[str] = []
    graded: list[str] = []

    ids = sorted({p.name.split(".")[0] for p in goldens.glob("*.score.npz")})
    if only:
        ids = [i for i in ids if i in set(only)]

    for cid in ids:
        mine: dict = {}
        try:
            receptor = parse_receptor(complexes / cid / f"{cid}_protein.pdb")
            for kind, all_atoms in (("score", False), ("conf", True)):
                if not (goldens / f"{cid}.{kind}.npz").exists():
                    continue
                out = build_receptor_graph(
                    receptor, c_alpha_max_neighbors=24,
                    all_atoms=all_atoms, atom_max_neighbors=8)
                if kind == "conf":
                    mine["atom_x"] = out["atom"]["x"].numpy()
                    mine["atom_pos"] = out["atom"]["pos"].numpy()
                    mine["atom_edge_index"] = out["atom"]["edge_index"].numpy()
                    mine["atom_rec_edge_index"] = out["atom"]["atom_rec_edge_index"].numpy()
                else:
                    mine["rec_x"] = out["x"].numpy()
                    mine["rec_pos"] = out["pos"].numpy()
                    mine["rec_chain_ids"] = out["chain_ids"].numpy()
                    mine["rec_side_chain_vecs"] = out["side_chain_vecs"].numpy()
                    mine["rec_edge_index"] = out["edge_index"].numpy()

            g = np.load(goldens / f"{cid}.score.npz", allow_pickle=True)
            mol = read_molecule(complexes / cid / f"{cid}_ligand.sdf", sanitize=True)
            mol.RemoveAllConformers()
            mol = AddHs(mol)
            generate_conformer(mol, seed=int(g["seed"]))
            lig = build_ligand_graph(mol, remove_hs=True)
            mine["lig_x"] = lig["x"].numpy()
            mine["lig_edge_index"] = lig["edge_index"].numpy()
            mine["lig_edge_attr"] = lig["edge_attr"].numpy()
            mine["lig_edge_mask"] = lig["edge_mask"].numpy()
            mine["lig_mask_rotate"] = np.asarray(lig["mask_rotate"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cid}: {type(exc).__name__}: {exc}")
            continue

        graded.append(cid)
        golden = {}
        for kind in ("score", "conf"):
            f = goldens / f"{cid}.{kind}.npz"
            if f.exists():
                golden.update(dict(np.load(f, allow_pickle=True)))

        for key in EXACT:
            if key not in mine or key not in golden:
                continue
            total[key] += 1
            passed[key] += int(np.array_equal(mine[key], golden[key]))
        for key, tol in CLOSE.items():
            if key not in mine or key not in golden:
                continue
            total[key] += 1
            passed[key] += int(np.allclose(mine[key], golden[key], atol=tol, equal_nan=True))
        for key in EDGES:
            if key not in mine or key not in golden:
                continue
            total[key] += 1
            passed[key] += int(np.array_equal(mine[key], golden[key]))
            total[key + "_as_set"] += 1
            passed[key + "_as_set"] += int(
                set(map(tuple, mine[key].T.tolist())) == set(map(tuple, golden[key].T.tolist())))
        # Sequence comparison, kept out of the arrays because it is text.
        total["rec_sequence"] += 1
        seq_mine = "".join(parse_receptor(
            complexes / cid / f"{cid}_protein.pdb").chain_sequences()[0])
        passed["rec_sequence"] += int(seq_mine == "".join(golden["rec_sequence"].tolist()))

    return {
        "graded": len(graded),
        "errors": errors,
        "match": {k: [passed[k], total[k]] for k in sorted(total)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--complexes", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    result = grade(Path(args.goldens).expanduser(), Path(args.complexes).expanduser(),
                   args.only.split(",") if args.only else None)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        Path(args.json).expanduser().write_text(text + "\n")


if __name__ == "__main__":
    main()
