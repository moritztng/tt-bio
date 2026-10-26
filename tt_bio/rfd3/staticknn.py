"""RFD3's atom k-nearest query with the atoms that do not move answered once per design.

In a binder or motif design the target's atoms carry no noise and come back from every step unchanged, bit for bit
(the page fixture: 4651 of 6051 atoms over all 199 steps). Their rows of the k-nearest query only change when a
moving atom comes within the row's k-th distance. So the static rows' k nearest among the static atoms are found
once (`StaticKnn.query`), and each step only asks a tree over the movers for the few that come within each static
row's k-th static distance, merged into the cached row by distance (`_merge`). Moving rows, rows reached by more
movers than the bounded query returns and rows with two equal distances among their first k + 1 are queried on the
full tree, exactly as before, so every row equals cKDTree(x).query(x, k) and its tie order. Distances on both sides
of a merge come from cKDTree's own float64 metric, so they compare exactly as the full tree compares them.
"""

from __future__ import annotations

import threading

import numba
import numpy as np
from scipy.spatial import cKDTree

#: Static atoms needed before the split pays for its bookkeeping; below this a step queries every row.
MIN_STATIC = 0.25
#: Movers merged into a static row per step; a row reached by more is queried on the full tree.
MAX_MOVERS = 16


@numba.njit(cache=True, nogil=True)
def _merge(sn, sd, mn, md, out, ok):
    """Row-wise merge of the static (sn, sd) and mover (mn, md; inf = none) lists, both ascending, into out[:, :k].

    ok[r] is cleared when a row saw MAX_MOVERS movers (more may exist) or two equal distances up to its k + 1-th.
    """
    h, k = sn.shape
    q = mn.shape[1]
    for r in range(h):
        if md[r, q - 1] < np.inf:
            ok[r] = False
            continue
        i = j = 0
        prev = -1.0
        good = True
        for c in range(k + 1):
            if i < k and (j >= q or sd[r, i] < md[r, j]):
                d = sd[r, i]; v = sn[r, i]; i += 1
            elif j < q and md[r, j] < np.inf:
                d = md[r, j]; v = mn[r, j]; j += 1
            elif i < k:
                d = sd[r, i]; v = sn[r, i]; i += 1
            else:
                break
            if d == prev or (i < k and j < q and sd[r, i] == md[r, j]):
                good = False
            prev = d
            if c < k:
                out[r, c] = v
        ok[r] = good


class StaticKnn:
    """One per design stream; reused across steps while the static atoms' coordinates stay identical."""

    def __init__(self):
        self.last = None      # previous call's coordinates
        self.ids = None       # static atom ids the cache was built on
        self.xs = None        # their coordinates at build time
        self.nn = self.dk = self.clean = None
        self.stats = [0, 0]   # rows answered, rows queried on the full tree
        self.lock = threading.Lock()  # the index thread and a synchronous first step can share one cache

    def _build(self, x, still, k, workers):
        ids = np.flatnonzero(still)
        xs = x[ids]
        kk = min(k + 1, len(ids))
        d, j = cKDTree(xs).query(xs, k=kk, workers=workers)
        d, j = d.reshape(len(ids), kk), j.reshape(len(ids), kk)
        self.ids, self.xs, self.nn, self.d = ids, xs, ids[j[:, :k]], np.ascontiguousarray(d[:, :k])
        self.dk = d[:, k - 1].copy()
        # a row whose first k + 1 distances are not strictly increasing has an order the tree picks: query it
        self.clean = (np.diff(d, axis=1) > 0).all(1) if kk > k else np.zeros(len(ids), bool)

    def query(self, x: np.ndarray, k: int, workers: int) -> np.ndarray:
        """x float64 [L, 3] -> nn [L, k] int64, equal to cKDTree(x).query(x, k=k)[1]."""
        with self.lock:
            return self._query(x, k, workers)

    def _query(self, x, k, workers):
        L = len(x)
        tree = cKDTree(x)
        last, self.last = self.last, x
        valid = self.ids is not None and self.ids[-1] < L and np.array_equal(x[self.ids], self.xs)
        if not valid:
            self.ids = None
            still = (x == last).all(1) if last is not None and last.shape == x.shape else None
            if still is not None and still.sum() >= max(MIN_STATIC * L, k + 1):
                self._build(x, still, k, workers)
        if self.ids is None:
            _, nn = tree.query(x, k=k, workers=workers)
            self.stats[0] += L; self.stats[1] += L
            return nn.reshape(L, k)
        moving = np.ones(L, bool); moving[self.ids] = False
        mid = np.flatnonzero(moving)
        nn = np.empty((L, k), np.int64)
        ok = self.clean.copy()
        if len(mid):
            q = min(MAX_MOVERS, len(mid))
            # every mover up to a row's k-th static distance, a hair past it so rounding cannot drop one at exactly
            # that distance (the merge then sees the tie and sends the row to the full tree); scipy before 1.9 takes
            # only a scalar bound, so query to the largest and mask each row to its own
            bound = self.dk * (1 + 1e-9)
            md, mj = cKDTree(x[mid]).query(self.xs, k=q, distance_upper_bound=float(bound.max()), workers=workers)
            md, mj = md.reshape(len(self.ids), q), mj.reshape(len(self.ids), q)
            md[md > bound[:, None]] = np.inf
            hit = np.flatnonzero(ok & (md[:, 0] < np.inf))
            if len(hit):
                mn = mid[np.minimum(mj[hit], len(mid) - 1)]
                out, good = np.empty((len(hit), k), np.int64), np.empty(len(hit), np.bool_)
                _merge(self.nn[hit], self.d[hit], mn, np.ascontiguousarray(md[hit]), out, good)
                ok[hit] = good
                nn[self.ids[hit[good]]] = out[good]
                ok_plain = ok.copy(); ok_plain[hit] = False
            else:
                ok_plain = ok
        else:
            ok_plain = ok
        nn[self.ids[ok_plain]] = self.nn[ok_plain]
        redo = np.concatenate([mid, self.ids[~ok]])
        if len(redo):
            _, nb = tree.query(x[redo], k=k, workers=workers)
            nn[redo] = nb.reshape(len(redo), k)
        self.stats[0] += L; self.stats[1] += len(redo)
        return nn
