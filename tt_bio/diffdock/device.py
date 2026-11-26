"""DiffDock's edge convolution on a Tenstorrent card: FC block plus tensor product as GEMMs.

The host module (:mod:`tt_bio.diffdock.tensor_product`) contracts four small tensors per path
per edge. That is the right golden and the wrong device program: a per-edge einsum over
3 x 1 x 3 blocks is nothing a tile engine can run. Here the same arithmetic is rewritten so
that every op is a row-wise GEMM over the edge axis against a constant matrix, or an
elementwise product of two ``[E, width]`` blocks, all widths padded to a multiple of 32.

Every path ``(i, j, k)`` of the released checkpoint has ``v = 1`` (the edge harmonics carry
one copy of each ``l``), so for one edge

    out[w, k] = pw * sum_u W[u, w] * Y[u, k],     Y[u, k] = sum_{i,j} a[u, i] b[j] C[i, j, k]

and per path the device runs:

1. ``Y`` for every ``(u, k)``: ``((a @ G) * (b @ E)) @ Q``. ``G`` holds the 3j entries,
   ``E`` copies each harmonic component across its block, ``Q`` sums over ``j`` and
   repeats ``Y[u, k]`` across ``w`` so the result lands on the term axis ``(k, u, w)``.
2. The per-edge weights straight onto the same term axis: ``h @ W2d + b2d``, where ``W2d`` is
   the checkpoint's second FC layer with its columns reordered, repeated over ``k`` and
   scaled by the path normalisation. The weight block upstream materialises is never built
   in upstream's layout, and the reorder costs nothing at run time.
3. ``(weights * Y) @ S``, where ``S`` sums over ``u`` and drops each ``(w, k)`` into e3nn's
   output column ``w * (2l + 1) + k``.

The constants are built in float64 from the shipped 3j table and cast once. A pad column of
a constant matrix is zero, so padding never reaches a real output column.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tt_bio.diffdock.tensor_product import FullyConnectedTensorProduct

TILE = 32


def pad32(n: int) -> int:
    return ((n + TILE - 1) // TILE) * TILE


class _Path:
    """The host-side float64 constants of one tensor-product path."""

    def __init__(self, tp: FullyConnectedTensorProduct, n: int, w2: Tensor, b2: Tensor,
                 weight_start: int):
        i, j, k = tp.paths[n]
        ti, tj, tk = tp.in_irreps.terms[i], tp.sh_irreps.terms[j], tp.out_irreps.terms[k]
        U, V, Wm = tp.weight_shapes[n]
        if V != 1:
            raise NotImplementedError(
                f"path {n} has sh multiplicity {V}; the device form assumes v = 1, which every "
                f"DiffDock-L signature satisfies")
        I, J, K = 2 * ti.l + 1, 2 * tj.l + 1, 2 * tk.l + 1
        C = tp._w3j64[n]                                   # [I, J, K], float64
        in_off = tp._in_slices[i].start
        sh_off = tp._sh_slices[j].start
        out_off = tp._out_slices[k].start
        pw = tp.path_weights[n]

        in_pad, sh_pad, out_pad = pad32(tp.in_irreps.dim), pad32(tp.sh_irreps.dim), pad32(tp.out_irreps.dim)
        uk = pad32(U * K)
        T = pad32(K * U * Wm)
        f64 = torch.float64

        G = torch.zeros(in_pad, J * uk, dtype=f64)
        E = torch.zeros(sh_pad, J * uk, dtype=f64)
        Q = torch.zeros(J * uk, T, dtype=f64)
        S = torch.zeros(T, out_pad, dtype=f64)
        W2d = torch.zeros(w2.shape[1], T, dtype=f64)
        b2d = torch.zeros(T, dtype=f64)
        for jj in range(J):
            E[sh_off + jj, jj * uk:jj * uk + U * K] = 1.0
            for uu in range(U):
                for ii in range(I):
                    for kk in range(K):
                        G[in_off + uu * I + ii, jj * uk + uu * K + kk] = C[ii, jj, kk]
        for kk in range(K):
            for uu in range(U):
                for ww in range(Wm):
                    t = (kk * U + uu) * Wm + ww
                    for jj in range(J):
                        Q[jj * uk + uu * K + kk, t] = 1.0
                    S[t, out_off + ww * K + kk] = 1.0
                    col = weight_start + uu * Wm + ww
                    W2d[:, t] = pw * w2[col].to(f64)
                    b2d[t] = pw * b2[col].to(f64)
        self.G, self.E, self.Q, self.S, self.W2d, self.b2d = G, E, Q, S, W2d, b2d


def path_constants(tp: FullyConnectedTensorProduct, fc) -> list[_Path]:
    """Per-path constants for *tp* fed by the two-layer FC block *fc* (``fc[0]``, ``fc[3]``)."""
    w2, b2 = fc[-1].weight.detach(), fc[-1].bias.detach()
    out, start = [], 0
    for n, (u, v, w) in enumerate(tp.weight_shapes):
        out.append(_Path(tp, n, w2, b2, start))
        start += u * v * w
    if start != tp.weight_numel:
        raise AssertionError(f"walked {start} weights, tp has {tp.weight_numel}")
    return out


def host_emulation(tp, fc, edge_attr: Tensor, x1: Tensor, x2: Tensor) -> Tensor:
    """The device program run in float64 torch: proves the rewrite before a card is involved."""
    f64 = torch.float64
    paths = path_constants(tp, fc)
    a = torch.nn.functional.pad(x1.to(f64), (0, pad32(x1.shape[1]) - x1.shape[1]))
    b = torch.nn.functional.pad(x2.to(f64), (0, pad32(x2.shape[1]) - x2.shape[1]))
    h = torch.relu(edge_attr.to(f64) @ fc[0].weight.to(f64).T + fc[0].bias.to(f64))
    out = 0
    for p in paths:
        y = ((a @ p.G) * (b @ p.E)) @ p.Q
        out = out + ((h @ p.W2d + p.b2d) * y) @ p.S
    return out[:, :tp.out_irreps.dim]


class DeviceEdgeTP:
    """``(edge_attr, x1, x2) -> tp(x1, x2, fc(edge_attr))`` on a ttnn device, per edge.

    *x1* is the destination node's features gathered per edge, *x2* the edge harmonics, both
    already on the device as ``[E, pad32(width)]`` TILE tensors; the result is
    ``[E, pad32(out_dim)]`` with zeros in the pad columns.
    """

    def __init__(self, device, tp: FullyConnectedTensorProduct, fc, *, dtype=None,
                 weight_dtype=None):
        import ttnn
        self.ttnn = ttnn
        self.device = device
        self.tp = tp
        self.dtype = dtype or ttnn.float32
        wdt = weight_dtype or self.dtype
        self.in_pad = pad32(tp.in_irreps.dim)
        self.sh_pad = pad32(tp.sh_irreps.dim)
        self.out_pad = pad32(tp.out_irreps.dim)
        self.hidden = fc[0].weight.shape[0]
        self.n_edge_features = fc[0].weight.shape[1]

        def up(t: Tensor, dt=wdt):
            t = t.to(torch.float32)
            if t.dim() == 1:
                t = t.reshape(1, -1)
            return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

        # The edge-feature and hidden widths (144 at ns=48) are not tile multiples; zero rows
        # and columns keep the pad lanes at relu(0) = 0, so they never feed a weight.
        hp, fp = pad32(self.hidden), pad32(self.n_edge_features)
        self.w1 = up(_pad2(fc[0].weight.detach().T, fp, hp))
        self.b1 = up(_pad2(fc[0].bias.detach().reshape(1, -1), 1, hp))
        # The 0/1 placement matrices are exact in bf16; the 3j values and the FC weights are not.
        self.paths = [
            dict(G=up(p.G), E=up(p.E, ttnn.bfloat16), Q=up(p.Q, ttnn.bfloat16),
                 S=up(p.S, ttnn.bfloat16), W2d=up(_pad2(p.W2d, hp, p.W2d.shape[1])),
                 b2d=up(p.b2d))
            for p in path_constants(tp, fc)
        ]
        self.ckc = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)

    def __call__(self, edge_attr, x1, x2):
        ttnn = self.ttnn
        mm = lambda x, w: ttnn.matmul(x, w, compute_kernel_config=self.ckc, dtype=self.dtype)
        h = ttnn.relu(ttnn.linear(edge_attr, self.w1, bias=self.b1,
                                  compute_kernel_config=self.ckc, dtype=self.dtype))
        out = None
        for p in self.paths:
            y = mm(ttnn.multiply(mm(x1, p["G"]), mm(x2, p["E"])), p["Q"])
            wt = ttnn.linear(h, p["W2d"], bias=p["b2d"], compute_kernel_config=self.ckc,
                             dtype=self.dtype)
            part = mm(ttnn.multiply(wt, y), p["S"])
            out = part if out is None else ttnn.add(out, part)
        return out


def _pad2(t: Tensor, rows: int, cols: int) -> Tensor:
    return torch.nn.functional.pad(t, (0, cols - t.shape[1], 0, rows - t.shape[0]))


def to_device(device, x: Tensor, dtype=None):
    """``[E, n]`` host block -> ``[E, pad32(n)]`` TILE tensor on *device*."""
    import ttnn
    x = torch.nn.functional.pad(x.to(torch.float32), (0, pad32(x.shape[1]) - x.shape[1]))
    return ttnn.from_torch(x, dtype=dtype or ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=device)
