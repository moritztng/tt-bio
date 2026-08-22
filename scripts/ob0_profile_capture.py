"""Call `calculate_profile` from whichever openfold3 tree is on sys.path, on a fixed
numpy MSA, and dump the result. One subprocess per tree (both expose a top-level
`openfold3`), so the comparison can be bit-exact.

    OF3_TREE=/tmp/pin_of3              OUT=/tmp/prof_pin.npy  <python> this.py
    OF3_TREE=/home/ttuser/ob0_upstream OUT=/tmp/prof_v050.npy <python> this.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ["OF3_TREE"])

import numpy as np  # noqa: E402


def main() -> None:
    from openfold3.core.data.primitives.sequence.msa import calculate_profile
    from openfold3.core.data.resources.residues import MoleculeType

    # Deliberately n_rows != n_cols and n_rows > 1: np.tile and np.repeat agree at
    # n_rows == 1, and agree trivially when the two extents coincide.
    rng = np.random.default_rng(0)
    n_rows, n_cols = 7, 24
    msa = rng.integers(0, 21, size=(n_rows, n_cols), dtype=np.int64)

    out = calculate_profile(msa_array=msa, molecule_type=MoleculeType.PROTEIN,
                            chunk_size=1000)
    np.save(os.environ["OUT"], np.asarray(out))
    print(f"{os.environ['OF3_TREE']}: profile {np.asarray(out).shape} "
          f"sum={np.asarray(out).sum():.6f} -> {os.environ['OUT']}")


if __name__ == "__main__":
    main()
