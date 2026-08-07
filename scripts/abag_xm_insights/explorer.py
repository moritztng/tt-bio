"""Per-target data for the site's explorer -- one real 64-sample fold per (model, target).

For each target we ship chunk 0: a genuine 64-sample fold, its samples ordered by the
model's own confidence (index 0 is the sample the model would hand you, the maximum is
what was in the pool). Everything the explorer draws -- the delivered pick, the oracle, the
spread, the rank of the best sample -- derives from that one array, so the page carries no
number the parquets do not.
"""

from __future__ import annotations

import numpy as np

import core

CHUNK = 0
DEPTH = 64


def run() -> dict:
    out = {"depth": DEPTH, "chunk": CHUNK, "models": {}}
    for m in core.MODELS:
        rows = {}
        for t, p in core.pools(m).items():
            q = p[p.chunk == CHUNK]
            if len(q) != DEPTH:
                continue
            o = core.rank_order(q.selector.to_numpy(), q.dockq.to_numpy())[::-1]
            rows[t] = [round(float(x), 3) for x in q.dockq.to_numpy()[o]]
        out["models"][m] = rows
    return out
