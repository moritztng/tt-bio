"""RFD3's distogram bin index (bucketize of the pairwise CA distances) in one numba pass, equal to the torch chain.

Per pair the kernel takes the float32 distance, a direct index on the uniform edges and a one-step fix-up against
the real float32 edges. It does not reproduce torch.linalg.norm's rounding, so any pair within a relative 1e-5 of an
edge (a few hundred of 469k at 685 tokens) is recomputed with torch.linalg.norm + torch.bucketize on the gathered
pairs, which gives the same bits as the full call. Pairs past the last edge, 77-81 % of them, cost one compare.

The kernel is serial on purpose. A parallel=True kernel makes numba launch its OpenMP pool at one thread per logical
core whatever torch was given, and that pool's spinning threads then slow every later torch CPU op in the process
about 20x: on qb1 one call turned rfd3's 0.1 s DiT weight fold into 2.1 s, +3 s a design, and sent the tfg tests
sharing a pytest process past the gate's timeout. Serial it is 3 ms against torch's 2 ms at 685 tokens on a busy
host and faster at larger B x N, where the parallel kernel was slower than both.
"""

from __future__ import annotations

import numba
import numpy as np
import torch

_TOL = 1e-5


@numba.njit(cache=True)
def _kernel(R, edges, tol, out, flag):
    B, N = R.shape[0], R.shape[1]
    E = edges.shape[0]
    lo = edges[0]
    inv = np.float32((E - 1) / (edges[E - 1] - edges[0]))
    far = edges[E - 1] + tol[E - 1]
    nflag = np.zeros(B * N, np.int64)
    for bi in range(B * N):
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
        nflag[bi] = c
    return nflag.sum()


def bins(R_L: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """R_L [B, N, 3] float32 (host), edges [E] float32 -> int32 [B, N, N], == bucketize(linalg.norm(...), edges)."""
    R = R_L.detach().contiguous()
    e = edges.numpy()
    B, N = R.shape[:2]
    out = np.empty((B, N, N), np.int32)
    flag = np.empty((B, N, N), np.bool_)
    n = _kernel(R.numpy(), e, (np.abs(e) * _TOL).astype(np.float32), out, flag)
    out = torch.from_numpy(out)
    if n:
        b, i, j = torch.from_numpy(flag).nonzero(as_tuple=True)
        out[b, i, j] = torch.bucketize(torch.linalg.norm(R[b, i] - R[b, j], dim=-1), edges).to(torch.int32)
    return out
