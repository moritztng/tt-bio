#!/usr/bin/env python3
"""tt_baseline with a pass-through wrapper on _transpose_memory_config only.

Second half of the enqueue-race bisect (gpu-vs-tt-precision-fairness): wrapping
BOTH config functions with no-op pass-throughs already prevents the pc 298-aa
wedge. This variant wraps only the transpose side (1208 of the 1732 wrapped
calls per fold); tt_baseline_onlyln.py wraps only the layer_norm side (524).
Whichever single wrapper still folds names the call path whose enqueue timing
the race lives in.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import tt_bio.tenstorrent as T

ORIG_TMC = T._transpose_memory_config


def tmc(t):
    return ORIG_TMC(t)


T._transpose_memory_config = tmc

import tt_baseline as B

if __name__ == "__main__":
    sys.exit(B.main())
