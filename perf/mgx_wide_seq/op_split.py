"""A tt_bio CLI run with every pair op's calls timed between device syncs, by op and pair shape.

    TT_VISIBLE_DEVICES=<c> ... python perf/mgx_wide_seq/op_split.py <out.json> -- <tt_bio.main args>

Wraps TriangleMultiplication, TriangleAttention and Transition `__call__` (start/end kept apart),
syncs the device before and after each call, and writes {op: [calls, seconds]} for pairs past
`concat_host_bytes()` plus the in-place counters. The syncs cost a little wall; compare two runs
of this script, never this against a plain fold.
"""
import json
import sys
import time
from collections import defaultdict

import ttnn

from tt_bio import tenstorrent as tt

out, args = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
T = defaultdict(lambda: [0, 0.0])


def wrap(cls, name):
    orig = cls.__call__

    def call(self, x, *a, **k):
        big = x.logical_volume() * 2 > tt.concat_host_bytes()
        if not big:
            return orig(self, x, *a, **k)
        dev = tt.get_device()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = orig(self, x, *a, **k)
        ttnn.synchronize_device(dev)
        key = f"{name}{'-end' if getattr(self, 'ending', False) else ''}:{'x'.join(map(str, x.shape))}"
        T[key][0] += 1
        T[key][1] += time.perf_counter() - t0
        return r
    cls.__call__ = call


for cls in (tt.TriangleMultiplication, tt.TriangleAttention, tt.Transition):
    wrap(cls, cls.__name__)

from tt_bio.main import cli  # noqa: E402

try:
    cli.main(args, standalone_mode=False)
finally:
    with open(out, "w") as f:
        json.dump({"ops": {k: [n, round(s, 3)] for k, (n, s) in sorted(T.items())},
                   "total_s": round(sum(s for _, s in T.values()), 3),
                   "pair_inplace": tt.PAIR_INPLACE_STATS}, f, indent=1)
