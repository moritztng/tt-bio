"""RFD3's atom k-nearest list built ahead of the step that needs it, then made exact.

The atom index (model._neighbours_kdtree) is a k-nearest query over the step's noisy coordinates, and the step waits
on it before the atom encoder. Recycle 0 of the previous step already predicts those coordinates closely: the next
step's X_noisy differs from that prediction only by the recycle-1 correction of the update. So the query runs during
the previous step's last recycle, on the prediction, for k' > k candidates (`candidates`), and the step itself only
re-ranks each row's candidates on the real coordinates (`rescore`).

A row is exact when its k-th re-ranked distance D_k is below d'(r) - |dx_r| - max_j |dx_j|, where d'(r) is its k'-th
predicted distance and dx the prediction error: every atom outside the candidates is then farther than D_k on the
real coordinates. A row is also refused when two of its first k + 1 re-ranked distances are equal, so its order
cannot depend on how ties are broken. Refused rows are queried for real. The result equals cKDTree's query row for
row, which tests/test_rfd3_specidx.py checks against it.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, nogil=True)
def _rescore(x, cand, dlast, dx, k, nn, ok):
    L, kp = cand.shape
    dmax = 0.0
    for i in range(L):
        dmax = max(dmax, dx[i])
    d2 = np.empty(kp)
    for r in range(L):
        for c in range(kp):
            j = cand[r, c]
            a = x[r, 0] - x[j, 0]
            b = x[r, 1] - x[j, 1]
            e = x[r, 2] - x[j, 2]
            d2[c] = a * a + b * b + e * e
        o = np.argsort(d2)
        good = np.sqrt(d2[o[k - 1]]) < dlast[r] - dx[r] - dmax
        for c in range(1, min(k + 1, kp)):
            if d2[o[c]] == d2[o[c - 1]]:
                good = False
        ok[r] = good
        for c in range(k):
            nn[r, c] = cand[r, o[c]]


def candidates(x_spec: np.ndarray, kp: int, workers: int):
    """k' nearest of the predicted coordinates (float64 [L, 3]): (candidate ids [L, k'], k'-th distance [L])."""
    from scipy.spatial import cKDTree
    kp = min(kp, len(x_spec))
    d, c = cKDTree(x_spec).query(x_spec, k=kp, workers=workers)
    return c.reshape(len(x_spec), kp).astype(np.int64), d.reshape(len(x_spec), kp)[:, -1].copy()


def rescore(x: np.ndarray, x_spec: np.ndarray, cand: np.ndarray, dlast: np.ndarray, k: int, workers: int):
    """The k nearest of every atom of x (float64 [L, 3]) in ascending distance, as cKDTree(x).query(x, k) returns.

    Rows the prediction cannot certify are queried for real; returns (nn [L, k] int64, number of such rows).
    """
    L = len(x)
    if k > cand.shape[1] or k == L:
        from scipy.spatial import cKDTree
        _, nn = cKDTree(x).query(x, k=k, workers=workers)
        return nn.reshape(L, k), L
    dx = np.sqrt(((x - x_spec) ** 2).sum(1))
    nn = np.empty((L, k), np.int64)
    ok = np.empty(L, np.bool_)
    _rescore(x, cand, dlast, dx, k, nn, ok)
    bad = np.flatnonzero(~ok)
    if len(bad):
        from scipy.spatial import cKDTree
        _, nb = cKDTree(x).query(x[bad], k=k, workers=workers)
        nn[bad] = nb.reshape(len(bad), k)
    return nn, len(bad)
