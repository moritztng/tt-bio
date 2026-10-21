#!/usr/bin/env python3
"""Which column-group plan triatt_ln picks over a whole fold, and the free L1 it picked it against.

`triatt_ln._plan` sizes its plan to the largest contiguous L1 block free at call time, so a buffer
left resident by another op can push the fused LayerNorm + qkvgb projection onto two column groups
(x read and normed twice) with no change to its own code. This wraps `_plan`, then runs
xmodel_ab.py unchanged with the remaining arguments.

    python tl_plan_probe.py OUT.json -- <xmodel_ab.py args, --tree first>
"""
import atexit, collections, json, runpy, sys
from pathlib import Path

out, rest = Path(sys.argv[1]), sys.argv[sys.argv.index("--") + 1:]
tree = Path(rest[rest.index("--tree") + 1]).resolve()
sys.path.insert(0, str(tree))
import ttnn  # noqa: E402
import tt_bio.triatt_ln as TL  # noqa: E402

SEEN = collections.Counter()
_plan = TL._plan


def _logged(kt, nw, tr, dev, ncores):
    lcf = ttnn.get_memory_view(dev, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    lcf = min(lcf) if isinstance(lcf, (list, tuple)) else lcf
    got = _plan(kt, nw, tr, dev, ncores)
    SEEN[json.dumps([kt, nw, int(bool(tr)), ncores, int(lcf), got])] += 1
    return got


TL._plan = _logged
atexit.register(lambda: out.write_text(json.dumps(
    {"rejects": {str(k): v for k, v in TL.REJECTS.items()}, "stats": list(TL.STATS),
     "plans": [[json.loads(k), n] for k, n in SEEN.most_common()]}, indent=1)))
sys.argv = [str(tree / "perf" / "of3_4xpd" / "xmodel_ab.py")] + rest
runpy.run_path(sys.argv[0], run_name="__main__")
