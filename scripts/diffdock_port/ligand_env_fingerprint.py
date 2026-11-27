"""Fingerprint the ligand graph and the seed conformer an environment produces.

Four PoseBusters complexes blow the score head to 1e15 in one venv and dock cleanly in
another (`state/dfd/BLOWUP-2026-11-27.md`).  The venvs differ in rdkit and numpy, and
`state/dfd/NAN-2026-11-27.md` already measured that rdkit/numpy, not torch, choose the seed
conformer: `lig_pos` was the only score-model input that moved between two venvs.

This prints, per complex, everything about the ligand side of the input that an environment
can change, with no network, no checkpoint and no device: whether the SDF reads at all, the
canonical SMILES, the graph (atom count, bond count, the rotatable-bond mask) and the seed
conformer's own geometry.  The conformer is reported by its *internal* geometry -- the sorted
pairwise distance vector -- because that is invariant to the random placement the sampler
applies afterwards and so compares across runs.

Run the same command in each environment and diff the json:

    PYTHONPATH=$PWD python3 scripts/diffdock_port/ligand_env_fingerprint.py --out pc.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: The four complexes that die in the Kabsch SVD (D-dfd-orch-87).
BLOWUP = ["7AVI_S2Q", "7TBU_S3P", "7X5N_5M5", "7ZCC_OGA"]

PB = Path.home() / "scratch/dfd/pb/posebusters_benchmark_set"


def versions() -> dict:
    import numpy
    import rdkit
    import torch

    return {
        "python": platform.python_version(),
        "host": platform.node(),
        "torch": torch.__version__,
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
    }


def digest(a) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def internal_geometry(pos) -> dict:
    """The conformer's shape, independent of where the sampler later puts it."""
    import numpy as np

    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu = np.triu_indices(len(pos), k=1)
    dist = np.sort(d[iu])
    return {
        "n_atoms": int(len(pos)),
        "radius_of_gyration": float(np.sqrt(((pos - pos.mean(0)) ** 2).sum(1).mean())),
        "max_pair_dist": float(dist[-1]) if dist.size else 0.0,
        "sorted_pair_dist_sha": digest(dist),
        "sorted_pair_dist_head": [round(float(x), 6) for x in dist[:5]],
    }


def fingerprint(cid: str, seed: int) -> dict:
    """Everything the ligand side of the score-model input depends on, or why it failed."""
    from rdkit import Chem

    out: dict = {"complex": cid}
    sdf = PB / cid / f"{cid}_ligand.sdf"
    if not sdf.exists():
        return {**out, "error": f"no such file: {sdf}"}

    # 1. does this environment's rdkit read the reference ligand at all?
    try:
        mol = next(iter(Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=True)), None)
        if mol is None:
            return {**out, "error": "SDMolSupplier returned None"}
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception as exc:                                     # noqa: BLE001
        return {**out, "error": f"{type(exc).__name__}: {exc}"}
    out["smiles"] = smiles
    out["smiles_sha"] = hashlib.sha256(smiles.encode()).hexdigest()[:16]

    # 2. the graph and the seed conformer the dock actually builds, from that SMILES
    from tt_bio.diffdock import ligand as lig

    try:
        mol2 = lig.read_ligand(smiles, seed=seed)
    except Exception as exc:                                     # noqa: BLE001
        return {**out, "error": f"read_ligand: {type(exc).__name__}: {exc}"}

    graph = lig.build_ligand_graph(mol2)
    mask = graph.get("edge_mask")
    out["graph"] = {
        "n_atoms": int(graph["x"].shape[0]),
        "n_edge_cols": int(graph["edge_index"].shape[1]),
        "n_rotatable": int(mask.sum()) if mask is not None else None,
        "atom_feature_sha": digest(graph["x"].numpy()),
        "edge_index_sha": digest(graph["edge_index"].numpy()),
    }
    if "pos" not in graph:
        return {**out, "error": "no conformer embedded"}
    out["conformer"] = internal_geometry(graph["pos"].numpy())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="-")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--complexes", nargs="*", default=BLOWUP)
    a = ap.parse_args()

    report = {
        "versions": versions(),
        "seed": a.seed,
        "complexes": [fingerprint(c, a.seed) for c in a.complexes],
    }
    text = json.dumps(report, indent=1)
    if a.out == "-":
        print(text)
    else:
        Path(a.out).expanduser().write_text(text + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
