#!/usr/bin/env python3
"""`duo_round.py` with `autograd.RELU_BW_GATED` set from BCW_RELU_GATED (1 gated, 0 composed).
tt_bio is not edited; the attribute is read at every ReLU backward."""
import os, pathlib, runpy, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt_bio import autograd
autograd.RELU_BW_GATED = os.environ.get("BCW_RELU_GATED", "1") == "1"
print(f"bcw-callcut: RELU_BW_GATED={autograd.RELU_BW_GATED}", flush=True)
sys.argv[0] = str(ROOT / "perf/bcx_p10_duotraj/duo_round.py")
runpy.run_path(sys.argv[0], run_name="__main__")
