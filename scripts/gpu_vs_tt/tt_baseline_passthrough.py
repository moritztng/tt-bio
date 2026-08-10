#!/usr/bin/env python3
"""tt_baseline with pure pass-through wrappers (no recording overhead).

Sharpens the enqueue-race repro (gpu-vs-tt-precision-fairness pass 5): the
sweep's recording wrappers on _transpose_memory_config/_l1_layer_norm make the
hang disappear. This variant wraps the same two functions but does no work --
no shape formatting, no dict updates -- so only a function-call frame or two of
per-call overhead is added. If this folds, the race window is microseconds
wide; if it hangs, avoiding the wedge needs the recording's heavier delay.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import tt_bio.tenstorrent as T
import tt_bio.protenix as P

ORIG_TMC = T._transpose_memory_config
ORIG_LN = T._l1_layer_norm


def tmc(t):
    return ORIG_TMC(t)


def ln(x, headroom, **kw):
    return ORIG_LN(x, headroom, **kw)


T._transpose_memory_config = tmc
T._l1_layer_norm = ln
P._l1_layer_norm = ln

import tt_baseline as B

if __name__ == "__main__":
    sys.exit(B.main())
