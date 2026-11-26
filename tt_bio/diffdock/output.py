"""Where a docking run's poses land, and in what files.

Upstream's layout byte for byte (D-dfd-10), because screening pipelines parse the filenames:

    <out_dir>/<name>/rank1.sdf
    <out_dir>/<name>/rank1_confidence-0.42.sdf
    ...
    <out_dir>/<name>/rank1_reverseprocess.pdb     # --save_trajectory only
    <out_dir>/failures.csv                        # a CSV or directory run that lost a row

plus one file nothing upstream writes: ``confidence.json``, so a caller reads the ranking
instead of parsing floats out of filenames (D-dfd-9). It is additive, so no upstream
consumer breaks.

Without a confidence model there are no confidence-suffixed files and the ranking is sample
order. Upstream crashes in that case (it formats ``confidence[rank]`` whatever the flags
said, and the per-complex handler counts a failure), so this is the one place the layout
deviates, and only where upstream has no output at all.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def pose_filename(rank: int, confidence: float | None) -> str:
    """Upstream's name for the pose at ``rank`` (1-based)."""
    return f"rank{rank}.sdf" if confidence is None else f"rank{rank}_confidence{confidence:.2f}.sdf"


def write_poses(write_dir, mol, positions, confidences=None, *, trajectories=None) -> list[Path]:
    """Write one complex's ranked poses. ``positions`` is already confidence-ordered.

    ``mol`` is the RDKit molecule the poses are conformers of; it is written once per rank
    with that rank's coordinates, as upstream's ``write_mol_with_coords`` does.
    """
    from rdkit import Chem

    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rows = []
    for i, pos in enumerate(positions):
        conf = None if confidences is None else float(confidences[i])
        paths = [write_dir / pose_filename(i + 1, conf)]
        if i == 0 and conf is not None:
            # Upstream writes rank 1 twice, bare and suffixed. Scripts in the wild read both.
            paths.insert(0, write_dir / "rank1.sdf")
        for path in paths:
            _write_mol_with_coords(Chem, mol, pos, path)
            written.append(path)
        rows.append({"rank": i + 1, "confidence": conf, "sdf": paths[-1].name})

    for i, pdb_text in enumerate(trajectories or []):
        path = write_dir / f"rank{i + 1}_reverseprocess.pdb"
        path.write_text(pdb_text)
        written.append(path)

    conf_path = write_dir / "confidence.json"
    conf_path.write_text(json.dumps(rows, indent=2) + "\n")
    written.append(conf_path)
    return written


def _write_mol_with_coords(Chem, mol, pos, path: Path) -> None:
    from rdkit.Chem import Conformer

    mol = Chem.Mol(mol)
    conf = Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = (float(v) for v in pos[i])
        conf.SetAtomPosition(i, (x, y, z))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def write_failures(out_dir, failures: list[tuple[str, str]]) -> Path | None:
    """One row per complex that failed, with its error. A bad ligand in a 10,000-row screen
    must not end the screen, and a count in a log is not an answer (FEATURE-PLAN conflict 1).
    """
    if not failures:
        return None
    path = Path(out_dir) / "failures.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["complex_name", "error"])
        w.writerows(failures)
    return path
