"""Assemble the committed DiffDock input fixtures from a capture directory.

    python3 tests/diffdock_input_fixtures.py \
        --goldens ~/scratch/dfd/goldens-input \
        --complexes ~/scratch/dfd/input-sel

Copies the chosen complexes' receptor, ligand and captured goldens into
`tests/fixtures/diffdock/` and writes INDEX.json. The set is deliberately small:
it is the regression signal that runs on every checkout, not the grading set.
The full PoseBusters grading lives in state/dfd/INPUT.md.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

#: Smallest single-chain, a larger ligand, and a two-chain receptor.
CHOSEN = ("6Z5Z_BDF", "7BJ6_TVK", "7BNH_BEZ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--complexes", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "fixtures" / "diffdock"))
    args = ap.parse_args()

    goldens, complexes = Path(args.goldens).expanduser(), Path(args.complexes).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    entries = []
    for cid in CHOSEN:
        for suffix in ("_protein.pdb", "_ligand.sdf"):
            shutil.copy(complexes / cid / f"{cid}{suffix}", out / f"{cid}{suffix}")
        for kind in ("score", "conf"):
            shutil.copy(goldens / f"{cid}.{kind}.npz", out / f"{cid}.{kind}.npz")
        g = np.load(out / f"{cid}.score.npz", allow_pickle=True)
        entries.append({
            "id": cid,
            "residues": int(len(g["rec_pos"])),
            "chains": int(len(np.unique(g["rec_chain_ids"]))),
            "lig_heavy_atoms": int(len(g["lig_x"])),
            "conformer_seed": int(g["seed"]),
        })

    (out / "INDEX.json").write_text(json.dumps({
        "source": "PoseBusters Benchmark Set, Zenodo record 8278563",
        "upstream": "gcorso/DiffDock @ 85c49b60, released v1.1 checkpoints",
        "captured_by": "tests/diffdock_input_capture.py",
        "complexes": entries,
    }, indent=2) + "\n")
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
