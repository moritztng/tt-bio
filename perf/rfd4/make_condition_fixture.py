"""Add RASA and chain-centroid conditioning to a design, so the two adherence metrics have
something to score.

`condition_rasa-binned_1_atom` (C_RAS, RasaEnum ints) is set on a seeded quarter of the atoms,
spread over buried, intermediate and exposed, and `condition_ChainCentroid_1_atom` (C_CENTROID,
one xyz per atom) gives every atom of the first chain a target 2.5 A off its actual centroid.

    python3 perf/rfd4/make_condition_fixture.py <design.cif> <out.cif>
"""

import argparse
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFCategory, CIFFile, get_structure

SEED = 20251021


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    cif = CIFFile.read(str(args.design))
    atoms = get_structure(cif, model=1)
    n = atoms.array_length()
    rng = np.random.default_rng(SEED)
    rasa = np.zeros(n, dtype=int)
    picked = rng.choice(n, size=n // 4, replace=False)
    rasa[picked] = rng.integers(1, 4, size=len(picked))
    cif.block["atom_site"]["condition_rasa-binned_1_atom"] = rasa.astype(str)

    chain = atoms.chain_id == atoms.chain_id[0]
    target = atoms.coord[chain].mean(axis=0) + np.array([2.0, -1.5, 0.0])
    idx = np.nonzero(chain)[0]
    cif.block["condition_ChainCentroid_1_atom"] = CIFCategory(
        {"idx0": idx.astype(str), **{f"val{k}": np.full(len(idx), f"{target[k]:.3f}") for k in range(3)}}
    )
    cif.write(str(args.out))
    print(f"{args.out}: {len(picked)} RASA-conditioned atoms, centroid target on chain {atoms.chain_id[0]}")


if __name__ == "__main__":
    main()
