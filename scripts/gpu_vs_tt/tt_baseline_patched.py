#!/usr/bin/env python3
"""tt_baseline with the z-size sweep's recording monkeypatches, verbatim.

The last untested discriminator between the sweep harness (folds 298 aa on pc
card 0, 3/3) and the tt_baseline CLI path (hangs 6/6): the sweep wraps
`_transpose_memory_config` and `_l1_layer_norm` with recording pass-throughs.
Their per-call Python overhead slows the host enqueue rate, so a fold through
this shim tells apart "the wrappers' existence matters" from a host-device
enqueue race that only a slower host path avoids. Exonerated already: input,
flags, ttnn/KMD versions, grid, MSA dir, cache clears, import order.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import ttnn
import tt_bio.tenstorrent as T
import tt_bio.protenix as P

ORIG_TMC = T._transpose_memory_config
ORIG_LN = T._l1_layer_norm
DEC = defaultdict(Counter)


def _shp(t):
    return "x".join(str(int(d)) for d in t.shape)


def tmc(t):
    mc = ORIG_TMC(t)
    DEC["transpose|" + _shp(t)]["L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"] += 1
    return mc


def ln(x, headroom, **kw):
    out, in_l1 = ORIG_LN(x, headroom, **kw)
    DEC["layer_norm|h=%s|%s" % (headroom, _shp(x))]["L1" if in_l1 else "DRAM"] += 1
    return out, in_l1


T._transpose_memory_config = tmc
T._l1_layer_norm = ln
P._l1_layer_norm = ln

import tt_baseline as B

if __name__ == "__main__":
    sys.exit(B.main())
