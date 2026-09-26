#!/usr/bin/env python3
"""One exactness-OFF full step, through `of3t-restep`'s own arm wrapper. This row's A/B arm.

`fullstep.py` run directly is an exactness-ON step: `502ed112e` made exact softmax and layer
norm the tape's default, so the host float64 path fires on every attention block and the run
costs ~25x. Every OFF reading this campaign quotes went through `steparms.arm`, which wraps
`fullstep.main()` in `ag.exact_training(False)` and stamps `host_quiet` either side of it.
Same wrapper here, so this row's arms are the same scope as the baseline they are compared
to -- and so `host_quiet` is recorded rather than asserted afterwards.

Cost of getting this wrong, measured on this row: four minutes of card on an arm that was
still in the trunk forward of rep 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf/of3t_restep"))

import steparms as A                                                # noqa: E402

if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2])
    raise SystemExit(A.arm(name, False, sys.argv[3:], out))
