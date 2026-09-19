#!/usr/bin/env python3
"""Round-trip `tt_bio/abodybuilder3_output.py`'s PDB writer through B1's instrument.

The port's structure numbers will be produced by writing a PDB and handing it to
`tt_bio/antibody_rmsd.py`, so the writer sits between the model and every accuracy claim. What can
go wrong there is not arithmetic, it is convention: an atom named into the wrong slot, a chain
break numbered differently from theirs, a masked atom written as a zero. None of that shows up as
an error -- it shows up as a plausible RMSD.

So this reads THEIR released truth structures with B1's own reader, rebuilds atom14 tensors from
them, writes them back out through OUR writer, and scores our file against their file with B1's
scorer. Every region must come back 0.000 A. Anything else is a convention bug, and the size of it
is the size of the error it would have hidden in a real evaluation.

Residues are placed as alanine, whose atom14 names are `['N', 'CA', 'C', 'O', 'CB', ...]`, so the
four scored atoms land in slots 0, 1, 2 and 4 -- the round trip tests the naming, not a residue
table. Only structures whose chains are complete are used: their PDB fixer drops residues with no
resolved atoms, and our writer numbers contiguously because a prediction has every residue the
region list names.

Run: PYTHONPATH=$PWD python3 scripts/abb3_port/output_gate.py <output-dir-from-zenodo>
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from tt_bio import antibody_rmsd
from tt_bio.abodybuilder3_output import score_fv, write_fv_pdb

#: atom14 slot for each atom the published evaluation scores, for alanine.
SLOT = {"N": 0, "CA": 1, "C": 2, "CB": 4}
ALA = 0


def rebuild(true_pdb: Path, regions: list[str]):
    """Their truth structure as `(aatype, atom14, atom_mask, n_heavy)`, or None if it is short."""
    atoms = antibody_rmsd.read_atoms(true_pdb, tuple(SLOT))
    n_heavy = sum(1 for r in regions if r in antibody_rmsd.HEAVY_REGIONS)
    heavy = [k for k in atoms if k[0] == "H"]
    light = [k for k in atoms if k[0] != "H"]
    if len(heavy) != n_heavy or len(light) != len(regions) - n_heavy:
        return None
    ordered = sorted(heavy, key=lambda k: (k[1], k[2])) + sorted(light, key=lambda k: (k[1], k[2]))
    n_res = len(ordered)
    atom14 = torch.zeros(n_res, 14, 3, dtype=torch.float64)
    mask = torch.zeros(n_res, 14)
    for i, key in enumerate(ordered):
        for name, xyz in atoms[key].items():
            atom14[i, SLOT[name]] = torch.tensor(xyz, dtype=torch.float64)
            mask[i, SLOT[name]] = 1.0
    return torch.full((n_res,), ALA), atom14, mask, n_heavy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--variant", default="base-loss")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    names = sorted(p.stem for p in (args.root / args.variant / "plddt").glob("*.pt"))
    worst, checked, skipped = 0.0, 0, 0
    worst_where = ""
    with tempfile.TemporaryDirectory() as tmp:
        for name in names[: args.limit]:
            regions = list(torch.load(args.root / args.variant / "plddt" / f"{name}.pt",
                                      weights_only=False)["region"])
            true_pdb = args.root / args.variant / "true" / f"{name}.pdb"
            built = rebuild(true_pdb, regions)
            if built is None:
                skipped += 1
                continue
            ours = write_fv_pdb(Path(tmp) / f"{name}.pdb", *built)
            scored = score_fv(true_pdb, ours, regions)
            for region, value in scored.items():
                if value > worst:
                    worst, worst_where = value, f"{name} {region}"
            checked += 1
    print(f"  round-tripped {checked} structures ({skipped} skipped as short), "
          f"worst region RMSD {worst:.4f} A at {worst_where or 'nowhere'}")
    # 5e-4 and not 0: the PDB carries three decimals, so a coordinate round trip is exact to 1e-3
    # and an RMSD over a few hundred atoms cannot exceed that.
    ok = checked > 0 and worst <= 5e-4
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
