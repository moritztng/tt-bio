#!/usr/bin/env python3
"""What (q_chunk, k_chunk) would the fused kernel's OWN order offer at 832?

`_tri_att_fused_large_s` is the route that serves OpenFold3's trunk at 1088, 1216 and 1472. Its
whole point, in its own words, is that "its pair is a NARROW q against a WIDE k and the stock
ladder never offers that". That is exactly the pairing 832 needs and is denied -- not because it
does not fit, but because the route is gated to q_len > _Q_SPLIT_MAX_S and 832 < 1024.

This prints the pairs the route would try, host-side. 1088 is the control: it must reproduce the
q64 k1088 that production already serves.
"""
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing")
import ttnn                                                        # noqa: E402
from tt_bio import tenstorrent as T                                # noqa: E402
from tt_bio import triatt_sdpa as TS                               # noqa: E402

HEADS, HEAD_DIM, DTYPE = 4, 32, ttnn.bfloat16
CORES = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
print("cap _Q_SPLIT_MAX_S = %d, cores = %d" % (TS._Q_SPLIT_MAX_S, CORES))
for n in (704, 832, 864, 1088, 1216, 1472):
    pairs = list(TS.fused_pairs(n, HEADS, HEAD_DIM, CORES, DTYPE))
    print("  n=%-5d eligible_today=%-5s pairs=%s"
          % (n, n > TS._Q_SPLIT_MAX_S, pairs[:8]))
