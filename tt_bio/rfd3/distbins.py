"""RFD3's distogram bin index (bucketize of the pairwise CA distances) in one numba pass, equal to the torch chain.

Per pair the kernel takes the float32 distance, a direct index on the uniform edges and a one-step fix-up against
the real float32 edges. It does not reproduce torch.linalg.norm's rounding, so any pair within a relative 1e-5 of an
edge (a few hundred of 469k at 685 tokens) is recomputed with torch.linalg.norm + torch.bucketize on the gathered
pairs, which gives the same bits as the full call. Pairs past the last edge, 77-81 % of them, cost one compare.

The kernel is not parallel=True. That makes numba launch its OpenMP pool at one thread per logical core whatever
torch was given, and the pool's spinning threads then slow every later torch CPU op in the process about 20x: on qb1
one call turned rfd3's 0.1 s DiT weight fold into 2.1 s, +3 s a design. Fully serial it cost WH +11.6 s a design
instead (Galaxy, 1000 MHz, RFD3 page cell 57.0 -> 68.6 s). So it is a nogil kernel over row ranges, run on a few
Python threads that block (not spin) between calls; every row is computed exactly as before, so the bits are too.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import torch

_TOL = 1e-5
_MAX_THREADS = 8
_ROWS_PER_THREAD = 128
_pool = None


@numba.njit(cache=True, nogil=True)
def _kernel(R, edges, tol, out, flag, r0, r1):
    N = R.shape[1]
    E = edges.shape[0]
    lo = edges[0]
    inv = np.float32((E - 1) / (edges[E - 1] - edges[0]))
    far = edges[E - 1] + tol[E - 1]
    nflag = 0
    for bi in range(r0, r1):
        b, i = bi // N, bi % N
        c = 0
        for j in range(N):
            dx = R[b, i, 0] - R[b, j, 0]
            dy = R[b, i, 1] - R[b, j, 1]
            dz = R[b, i, 2] - R[b, j, 2]
            d = np.sqrt(np.float32(dx * dx + dy * dy + dz * dz))
            if d > far:
                out[b, i, j] = E
                flag[b, i, j] = False
                continue
            k = min(max(int((d - lo) * inv) + 1, 0), E)
            while k > 0 and not edges[k - 1] < d:
                k -= 1
            while k < E and edges[k] < d:
                k += 1
            f = (k > 0 and d - edges[k - 1] <= tol[k - 1]) or (k < E and edges[k] - d <= tol[k])
            out[b, i, j] = k
            flag[b, i, j] = f
            c += f
        nflag += c
    return nflag


def bins(R_L: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """R_L [B, N, 3] float32 (host), edges [E] float32 -> int32 [B, N, N], == bucketize(linalg.norm(...), edges)."""
    R = R_L.detach().contiguous()
    e = edges.numpy()
    B, N = R.shape[:2]
    out = np.empty((B, N, N), np.int32)
    flag = np.empty((B, N, N), np.bool_)
    args = (R.numpy(), e, (np.abs(e) * _TOL).astype(np.float32), out, flag)
    rows = B * N
    t = max(1, min(_MAX_THREADS, torch.get_num_threads(), rows // _ROWS_PER_THREAD))
    if t == 1:
        n = _kernel(*args, 0, rows)
    else:
        global _pool
        if _pool is None:
            _pool = ThreadPoolExecutor(_MAX_THREADS, thread_name_prefix="rfd3-distbins")
        cut = [rows * k // t for k in range(t + 1)]
        n = sum(_pool.map(lambda k: _kernel(*args, cut[k], cut[k + 1]), range(t)))
    out = torch.from_numpy(out)
    if n:
        b, i, j = torch.from_numpy(flag).nonzero(as_tuple=True)
        out[b, i, j] = torch.bucketize(torch.linalg.norm(R[b, i] - R[b, j], dim=-1), edges).to(torch.int32)
    return out
