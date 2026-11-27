"""r4t-step: the RFD4-Proteina training step harness. The step itself lives in `tt_bio.train.rfd4_run`
(moved there by r4t D56); this caller keeps the perf tools it can reach on sys.path and the old command line.
`python3 perf/r4t_step/trunk_step.py --help` lists the flags."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "r4t_perf"))   # --pcie-meter
sys.path.insert(0, os.path.join(_HERE, "..", ".."))           # the tree this file is in, as before the move

from tt_bio.train.rfd4_run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
