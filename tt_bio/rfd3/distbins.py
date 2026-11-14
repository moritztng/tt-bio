"""RFD3's distogram bin index (bucketize of the pairwise CA distances) in one numba pass, equal to the torch chain.

Per pair the kernel takes the float32 distance, a direct index on the uniform edges and a one-step fix-up against
the real float32 edges. It does not reproduce torch.linalg.norm's rounding, so any pair within a relative 1e-5 of an
edge (a few hundred of 469k at 685 tokens) is recomputed with torch.linalg.norm + torch.bucketize on the gathered
pairs, which gives the same bits as the full call. Pairs past the last edge, 77-81 % of them, cost one compare.

Numba's own threads are off on purpose. A parallel=True kernel makes numba launch its OpenMP pool at one thread per
logical core whatever torch was given, and that pool's spinning threads then slow every later torch CPU op in the
process about 20x: on qb1 one call turned rfd3's 0.1 s DiT weight fold into 2.1 s, +3 s a design, and sent the tfg
tests sharing a pytest process past the gate's timeout. Instead the kernel releases the GIL and up to four Python
threads each take a share of the upper triangle, mirroring every result (the distance is symmetric to the bit). At
704 tokens on pc that is 0.27 ms of kernel against 1.27 ms serial over the full matrix. The call sits right before
the token encoder is enqueued, twice a step, so the card waits for it.
"""

from __future__ import annotations

import numba
import numpy as np
import torch

_TOL = 1e-5
# Python threads over row ranges, the kernel releasing the GIL: numba's own pool stays unlaunched (see above).
_THREADS = max(1, min(4, torch.get_num_threads()))


@numba.njit(cache=True, nogil=True)
def _kernel(R, edges, tol, out, flag, b, i0, i1):
    """Rows [i0, i1) of design b against every j >= i, each result written to (i, j) and (j, i): the distance is
    symmetric to the bit (R_j - R_i is R_i - R_j negated, so the squares agree). Returns the flagged entries."""
    N = R.shape[1]
    E = edges.shape[0]
    lo = edges[0]
    inv = np.float32((E - 1) / (edges[E - 1] - edges[0]))
    far = edges[E - 1] + tol[E - 1]
    c = 0
    for i in range(i0, i1):
        for j in range(i, N):
            dx = R[b, i, 0] - R[b, j, 0]
            dy = R[b, i, 1] - R[b, j, 1]
            dz = R[b, i, 2] - R[b, j, 2]
            d = np.sqrt(np.float32(dx * dx + dy * dy + dz * dz))
            if d > far:
                out[b, i, j] = E
                out[b, j, i] = E
                flag[b, i, j] = False
                flag[b, j, i] = False
                continue
            k = min(max(int((d - lo) * inv) + 1, 0), E)
            while k > 0 and not edges[k - 1] < d:
                k -= 1
            while k < E and edges[k] < d:
                k += 1
            f = (k > 0 and d - edges[k - 1] <= tol[k - 1]) or (k < E and edges[k] - d <= tol[k])
            out[b, i, j] = k
            out[b, j, i] = k
            flag[b, i, j] = f
            flag[b, j, i] = f
            c += f
    return c


_POOL = None


def _chunks(N, n):
    """n row ranges holding about equal shares of the upper triangle (row i owns N - i pairs)."""
    w = np.cumsum(np.arange(N, 0, -1))
    cut = np.searchsorted(w, w[-1] * np.arange(1, n) / n) + 1
    edges = [0, *sorted(set(int(c) for c in cut if 0 < c < N)), N]
    return list(zip(edges[:-1], edges[1:]))


def bins(R_L: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """R_L [B, N, 3] float32 (host), edges [E] float32 -> int32 [B, N, N], == bucketize(linalg.norm(...), edges)."""
    R = R_L.detach().contiguous()
    e = edges.numpy()
    B, N = R.shape[:2]
    out = np.empty((B, N, N), np.int32)
    flag = np.empty((B, N, N), np.bool_)
    args = (R.numpy(), e, (np.abs(e) * _TOL).astype(np.float32), out, flag)
    jobs = [(b, i0, i1) for b in range(B) for i0, i1 in _chunks(N, _THREADS if N >= 128 else 1)]
    if len(jobs) == 1:
        n = _kernel(*args, *jobs[0])
    else:
        global _POOL
        if _POOL is None:
            from concurrent.futures import ThreadPoolExecutor
            _POOL = ThreadPoolExecutor(_THREADS, thread_name_prefix="rfd3-distbins")
        n = sum(_POOL.map(lambda j: _kernel(*args, *j), jobs))
    out = torch.from_numpy(out)
    if n:
        b, i, j = torch.from_numpy(flag).nonzero(as_tuple=True)
        out[b, i, j] = torch.bucketize(torch.linalg.norm(R[b, i] - R[b, j], dim=-1), edges).to(torch.int32)
    return out
