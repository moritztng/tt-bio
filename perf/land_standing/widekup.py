#!/usr/bin/env python3
"""Does TT_BIO_SDPA_WIDE_K_UP ever offer a rung? It is default-off with a measured 1.1554x.

The charter's rule is that a merged lever defaulting off is not a landed win, so a default-off
lever with a real number attached is a candidate. But its own comment says the upward k rung
exists only "under bfp8", and this row has been bitten before by levers that are on and still
never fire (`a-defects-reachability-must-be-counted-on-the-real-model`). So count the reach
before spending any device time on a fold A/B.

`_k_chunks_up` is consumed only by `_tri_att_k_chunks`, i.e. the STOCK ladder, and it returns
dividing k values WIDER than the shipped pick whose CBs fit L1 at the widest dividing q. Empty
tuple means the lever cannot change anything at that shape.
"""
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing")
import ttnn                                                        # noqa: E402
from tt_bio import tenstorrent as T                                # noqa: E402

TILE = T.SDPA_CHUNK_TILE
# Head configurations this file's own comments name for the triangle-attention path.
CONFIGS = [(4, 32), (8, 32)]
DTYPES = [("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)]

print("grid %s, cap %s" % (T.COMPUTE_GRID_MAIN, "n/a"))
for heads, head_dim in CONFIGS:
    for dname, dt in DTYPES:
        fires, checked = [], 0
        for n in range(TILE, 1536 + TILE, TILE):
            checked += 1
            try:
                up = T._k_chunks_up(n, n, heads, head_dim, dt)
            except Exception as exc:                               # noqa: BLE001
                up = ("ERR:%s" % type(exc).__name__,)
            if up:
                fires.append((n, up[:3]))
        print("\nheads=%d head_dim=%d dtype=%-5s : %d of %d tile-aligned lengths get an upward rung"
              % (heads, head_dim, dname, len(fires), checked))
        for n, up in fires[:12]:
            shipped = T._sdpa_chunks_shipped(n, n)
            print("    n=%-5d shipped(q,k)=%-12s upward k offered=%s" % (n, shipped, up))
        if len(fires) > 12:
            print("    ... and %d more" % (len(fires) - 12))
