"""Time and profile the CIF write of one design, card-free, on a stand-in output with real shapes.

    python perf/rfd4/host3/write_prof.py --spec <spec.yaml> [--reps 3] [--profile out.pstats] [--out dir]

The stand-in keeps the input's residue where C_SEQ conditions it and draws a residue elsewhere,
with the decoder's atom37 mask for that residue and the input's CA positions, so the writer takes
the same paths (matching residue copy, reconciled residue) a real design does.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch


def stand_in(features, atom_array, seed=0):
    import numpy as np
    from tt_bio._vendor.rfd4_prep._cif.utils_structure_export import UNIFIED_ATOM37_ENCODING
    from tt_bio.rfd4 import ae_fold as AF
    tab = AF.tables()
    idx = UNIFIED_ATOM37_ENCODING.token_to_idx
    n = features["padding_mask"].shape[1]
    g = torch.Generator().manual_seed(seed)
    rt = torch.randint(1, 21, (1, n), generator=g)  # the 20 standard residues, never <M>
    _, first = np.unique(atom_array.token_id, return_index=True)
    for i, name in enumerate(atom_array.res_name[first][:n]):
        if name in idx and name != "<M>":
            rt[0, i] = idx[name]  # the target keeps its residue, as a design does
    class Out:
        pass
    o = Out()
    o.residue_type = rt
    o.atom_mask = tab["atom_mask"][rt]
    o.coords_nm = features["coords"][:, :, :, :].clone().reshape(1, n, 37, 3) / 10.0
    o.coords_nm = o.coords_nm + 0.01 * torch.randn(o.coords_nm.shape, generator=g)
    o.insertion_logits = torch.zeros(1, n, n)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile")
    ap.add_argument("--out", default="/tmp/host3_write")
    a = ap.parse_args()
    from tt_bio import rfd4
    out = Path(a.out)
    designs = rfd4.prepare_examples(a.spec, out / "inputs")
    features, atom_array = rfd4.featurize(designs[0].cif, seed=0)
    o = stand_in(features, atom_array)
    for i in range(a.reps):
        prof = None
        if a.profile and i == a.reps - 1:
            import cProfile
            prof = cProfile.Profile()
            prof.enable()
        t = time.perf_counter()
        rfd4.write_design(o, features, atom_array, out / f"w{time.time_ns()}.cif")
        dt = time.perf_counter() - t
        if prof:
            prof.disable()
            prof.dump_stats(a.profile)
        print(f"write {i} {dt:.3f} s", flush=True)


if __name__ == "__main__":
    main()
