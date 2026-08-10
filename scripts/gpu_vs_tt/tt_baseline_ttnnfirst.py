#!/usr/bin/env python3
"""tt_baseline with ttnn imported before anything else, and nothing else changed.

Bisects why the tt_baseline CLI path hangs pc card 0 at 298 aa while the z-size
sweep harness folds the same input/flags/card (gpu-vs-tt-precision-fairness).
The sweep imports ttnn first; the CLI path imports torch before ttnn via
build_fold. Symbol interposition between libtorch and ttnn's bundled libs is
order-sensitive. Exonerated already: input, flags, ttnn/KMD versions, grid,
MSA dir contents, and the set_arm cache clears (tt_baseline_cacheclear.py).
"""
import sys
from pathlib import Path

import ttnn  # noqa: F401 -- must precede torch/tt_bio imports; this IS the experiment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import tt_baseline as B

if __name__ == "__main__":
    sys.exit(B.main())
