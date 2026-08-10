#!/usr/bin/env python3
"""tt_baseline with a pass-through wrapper on _l1_layer_norm only.

Second half of the enqueue-race bisect (gpu-vs-tt-precision-fairness): wrapping
both config functions prevents the pc 298-aa wedge; wrapping only
_transpose_memory_config still hangs. This variant wraps only the layer_norm
side (524 of the 1732 wrapped calls per fold). If it folds, the race lives in
the h=1.5 L1 layer_norm enqueue path -- the same class the z-size-robustness
leg root-caused for qb1's [385,506] CB-clash band.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import tt_bio.tenstorrent as T
import tt_bio.protenix as P

ORIG_LN = T._l1_layer_norm


def ln(x, headroom, **kw):
    return ORIG_LN(x, headroom, **kw)


T._l1_layer_norm = ln
P._l1_layer_norm = ln

import tt_baseline as B

if __name__ == "__main__":
    sys.exit(B.main())
