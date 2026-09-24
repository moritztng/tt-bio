#!/usr/bin/env python3
"""Price the device gradient step at the state BindCraft 2 actually hands the predictor.

`bcx-e2e` measured its step at n=256, a different draw of the same campaign. The PD-L1
seed-0 trajectory is a 77-residue binder padded to 96 against the 115-residue target, so
the predictor sees n=211. 211 is not a multiple of the 32 token bucket; 224 is the same
state rounded up to it. Both are registered with `bcx-e2e`'s own harness and run by its
`time` command, changing nothing else: same trunk, same seam, same arms.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "bcx_e2e"))

import e2e                                                            # noqa: E402

# (n_target, n_binder_padded, n_binder_real). 211 is what BindCraft 2's own
# pad_design_chains produces (perf/bcx_predictor/state_shape.json); 224 is 211 rounded up
# to the 32 bucket, the same design with the binder padded 13 residues further.
e2e.STATES[211] = (115, 96, 77)
e2e.STATES[224] = (115, 109, 77)

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + (sys.argv[1:] or ["time", "--n", "211", "--reps", "3"])
    e2e.main()
