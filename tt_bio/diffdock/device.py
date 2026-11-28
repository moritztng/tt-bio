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


def edge_bucket(n: int) -> int:
    """Padded length of a pose-dependent edge set: the next power of two, at least a tile.

    A radius or cross edge set has a different size on every pose, and at a multiple of 32
    nearly every pose is a new shape, so a new set of device programs. On this ladder the
    poses of one ligand, and ligands of similar size, share them. The padding is masked
    edges, at most as many as the real ones.
    """
    return max(TILE, 1 << (n - 1).bit_length())


def exact01_dtype(device, working):
    """The dtype of a 0/1 matrix that multiplies a *working*-dtype operand on *device*.

    0/1 is exact in bf16, and Blackhole keeps the other operand of an fp32 x bf16 matmul at full
    width, so bf16 is free there. Wormhole runs that operand at bf16 width: the tensor product's
    error went from Blackhole's 4e-3 to 1.1e-2 (state/dfd/WH.md), so there the matrix carries the
    working dtype.
    """
    import ttnn
    return ttnn.bfloat16 if device.arch() == ttnn.Arch.BLACKHOLE else working


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
        pdt = exact01_dtype(device, self.dtype)
        self.paths = [
            dict(G=up(p.G), E=up(p.E, pdt), Q=up(p.Q, pdt),
                 S=up(p.S, pdt), W2d=up(_pad2(p.W2d, hp, p.W2d.shape[1])),
                 b2d=up(p.b2d))
            for p in path_constants(tp, fc)
        ]
        self.ckc = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)

    def __call__(self, edge_attr, x1, x2, h=None):
        """*h*, when given, is the FC block's hidden layer already built (``[E, pad32(hidden)]``)
        and *edge_attr* is not read."""
        ttnn = self.ttnn
        mm = lambda x, w: ttnn.matmul(x, w, compute_kernel_config=self.ckc, dtype=self.dtype)
        if h is None:
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


def fc1_blocks(fc, ns: int, in_pad: int, plan=None):
    """``(W1a, W1s, W1d, b1)`` in float64 with ``cat([ea, x[src, :ns], x[dst, :ns]]) @ W1 + b =
    ea @ W1a + x[src] @ W1s + x[dst] @ W1d + b1`` for zero-padded ``ea`` and node rows ``x`` of
    width *in_pad*. With an edge-tile *plan*, the hidden width and bias are the kernel's (its
    folded-bias lane included); otherwise ``pad32(hidden)`` and the FC bias.
    """
    f64 = torch.float64
    W = fc[0].weight.detach().to(f64).T                 # [F, hidden]
    F, hidden = W.shape
    a = F - 2 * ns
    if a <= 0:
        raise ValueError(f"edge features {F} cannot hold two node blocks of {ns}")
    hp = plan.hidden_pad if plan is not None else pad32(hidden)
    b1 = torch.zeros(hp, dtype=f64)
    if plan is not None:
        b1[:] = plan.b1
    else:
        b1[:hidden] = fc[0].bias.detach().to(f64)
    blocks = []
    for rows, lo, hi in ((pad32(a), 0, a), (in_pad, a, a + ns), (in_pad, a + ns, F)):
        w = torch.zeros(rows, hp, dtype=f64)
        w[:hi - lo, :hidden] = W[lo:hi]
        blocks.append(w)
    return (*blocks, b1.reshape(1, -1))


def to_device(device, x: Tensor, dtype=None):
    """``[E, n]`` host block -> ``[E, pad32(n)]`` TILE tensor on *device*."""
    import ttnn
    x = torch.nn.functional.pad(x.to(torch.float32), (0, pad32(x.shape[1]) - x.shape[1]))
    return ttnn.from_torch(x, dtype=dtype or ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=device)


def gather_rows(table, idx, n: int):
    """``table[idx]`` for a TILE ``[N, d]`` table and a ``[1, n]`` uint32 row index.

    ``ttnn.embedding`` only takes a bf16 table. A bf16 table goes straight through; a float32
    one is split into three bf16 slices, ``hi + mid + lo``, each the rounding of what the ones
    before it left, so 3 x 8 mantissa bits carry all 24 of float32 and the gather stays an exact
    copy. The split runs on the card, so a table the previous layer produced never leaves it.
    """
    import ttnn
    bf, rm = ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
    if table.dtype == bf:
        return ttnn.reshape(ttnn.embedding(idx, ttnn.to_layout(table, rm), layout=ttnn.TILE_LAYOUT),
                            (n, table.shape[-1]))
    out, rest = None, table
    for k in range(3):
        part = ttnn.typecast(rest, bf)
        if k < 2:
            rest = ttnn.subtract(rest, ttnn.typecast(part, ttnn.float32))
        got = ttnn.embedding(idx, ttnn.to_layout(part, rm), layout=ttnn.TILE_LAYOUT)
        got = ttnn.typecast(ttnn.reshape(got, (n, table.shape[-1])), ttnn.float32)
        out = got if out is None else ttnn.add(out, got)
    return out


class DeviceConvLayer:
    """:class:`tt_bio.diffdock.conv.TensorProductConvLayer` on the card, score-network order.

    ``gather node_attr[edge_dst] -> per-group FC + tensor product -> mask -> scatter-mean onto
    edge_src -> batch norm -> residual``. The scatter is a dense ``[N, E]`` one-hot matmul
    built once per static graph (tt-atom's form, ``tt_atom/model.py``); the mask scales the
    messages and is what the divisor sums, so a node with every edge masked gets
    ``0 / eps = 0`` rather than NaN, which is the all-masked rule of D-dfd-orch-10.

    *edge_tp* picks the per-edge FC + tensor product: ``"gemm"`` (:class:`DeviceEdgeTP`) or
    ``"tile"`` (:class:`~tt_bio.diffdock.edge_tile.DeviceEdgeTile`, one fused Blackhole kernel).
    The tile kernel writes its own column order; the 0/1 permutation back to e3nn's commutes
    with the scatter, so it is applied on node rows after it.
    """

    def __init__(self, device, layer, *, dtype=None, edge_tp: str = "gemm"):
        import ttnn
        if layer.residual_first:
            raise NotImplementedError("the confidence model's residual-first order is not ported yet")
        self.ttnn, self.device, self.layer = ttnn, device, layer
        self.dtype = dtype or ttnn.float32
        self.fcs = fcs = list(layer.fc) if layer.edge_groups > 1 else [layer.fc]
        self.in_dim, self.out_dim = layer.in_irreps.dim, layer.out_irreps.dim
        self.out_pad = pad32(self.out_dim)
        self.perm = None
        self.x1_bf16 = edge_tp == "tile"
        if edge_tp == "gemm":
            self.tps = [DeviceEdgeTP(device, layer.tp, fc, dtype=self.dtype) for fc in fcs]
            self.ckc = self.tps[0].ckc
        elif edge_tp == "tile":
            from tt_bio.diffdock.edge_tile import DeviceEdgeTile
            if device.arch() != ttnn.Arch.BLACKHOLE:
                raise NotImplementedError("the fused edge-tile kernel is Blackhole only")
            self.tps = [DeviceEdgeTile(device, layer.tp, fc) for fc in fcs]
            perm = self.tps[0].plan.perm
            if any(not torch.equal(t.plan.perm, perm) for t in self.tps[1:]):
                raise ValueError("edge groups disagree on the kernel column order")
            self.perm = ttnn.from_torch(_pad2(perm, perm.shape[0], self.out_pad).float(),
                                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=device)
            self.ckc = ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)
        else:
            raise ValueError(f"edge_tp must be gemm or tile, not {edge_tp!r}")
        bn = layer.batch_norm
        scale = torch.ones(self.out_pad, dtype=torch.float64)
        shift = torch.zeros(self.out_pad, dtype=torch.float64)
        if bn is not None:
            im = iv = 0
            for term, sl in zip(bn.irreps, bn.irreps.slices()):
                norm = (bn.running_var[iv:iv + term.mul].double() + bn.eps).pow(-0.5)
                if bn.affine:
                    norm = norm * bn.weight[iv:iv + term.mul].double()
                s = norm.repeat_interleave(2 * term.l + 1)
                scale[sl] = s
                if term.l == 0 and term.p == 1:
                    b = bn.bias[im:im + term.mul].double() if bn.affine else 0.0
                    shift[sl] = b - bn.running_mean[im:im + term.mul].double() * norm
                    im += term.mul
                iv += term.mul
        up = lambda t: ttnn.from_torch(t.float().reshape(1, -1), dtype=self.dtype,
                                       layout=ttnn.TILE_LAYOUT, device=device)
        self.bn_scale, self.bn_shift = up(scale), up(shift)
        self._split = {}

    def graph(self, edge_index: Tensor, n_out: int, n_in: int):
        """Upload the loop-invariant half of a static graph: the gather index and the scatter."""
        ttnn = self.ttnn
        src, dst = edge_index
        E = edge_index.shape[1]
        if E % TILE or n_out % TILE:
            raise ValueError(f"edges {E} and nodes {n_out} must be multiples of {TILE}")
        S = torch.zeros(n_out, E)
        S[src, torch.arange(E)] = 1.0
        idx = ttnn.from_torch(dst.to(torch.int32).reshape(1, E), dtype=ttnn.uint32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        # 0/1, and the scatter is a sum of exact products.
        S = ttnn.from_torch(S, dtype=exact01_dtype(self.device, self.dtype),
                            layout=ttnn.TILE_LAYOUT, device=self.device)
        return dict(idx=idx, S=S, E=E, n_out=n_out, edge_index=edge_index)

    def fc1_split(self, ns: int):
        """The first FC layer of every edge group cut into its three input blocks.

        The score network's edge input is ``cat([edge_attr, x[src, :ns], x[dst, :ns]])``, so
        ``e @ W1 = edge_attr @ W1a + (x @ W1s)[src] + (x @ W1d)[dst]``: the node halves are
        multiplied on node rows and gathered, and the concatenated edge block never exists.
        ``W1s``/``W1d`` take the whole padded node row and are zero past ``ns``. The bias is the
        edge TP's own, including the tile kernel's folded-bias lane.
        """
        if ns not in self._split:
            up = lambda t: self.ttnn.from_torch(t.float(), dtype=self.dtype,
                                                layout=self.ttnn.TILE_LAYOUT, device=self.device)
            self._split[ns] = [
                tuple(up(t) for t in fc1_blocks(fc, ns, pad32(self.in_dim),
                                                getattr(tp, "plan", None)))
                for fc, tp in zip(self.fcs, self.tps)]
        return self._split[ns]

    def _group_index(self, g, sizes):
        """Per edge group ``(src, dst)`` uint32 gather indices, uploaded once per graph."""
        key = ("groups", sizes)
        if key not in g:
            ttnn, idx, lo = self.ttnn, [], 0
            for n in sizes:
                pair = [ttnn.from_torch(r[lo:lo + n].to(torch.int32).reshape(1, n),
                                        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                        device=self.device)
                        for r in g["edge_index"]]
                idx.append(pair)
                lo += n
            g[key] = idx
        return g[key]

    def inv_degree(self, g, edge_mask):
        """``1 / max(sum of the mask over a node's edges, eps)``, ``[n_out, 1]``.

        It depends on the graph and the mask only, so layers that share both share it.
        """
        ttnn = self.ttnn
        div = ttnn.matmul(g["S"], edge_mask, compute_kernel_config=self.ckc, dtype=self.dtype)
        return ttnn.reciprocal(ttnn.clamp(div, min=float(torch.finfo(torch.float32).eps)))

    def __call__(self, node_attr, g, edge_attr_groups, edge_sh, edge_mask, ns=None,
                 out_attr=None, inv_degree=None):
        """*node_attr* is ``[N, pad32(in_dim)]`` TILE.

        *edge_attr_groups* are TILE ``[E_g, pad32(F)]`` blocks partitioning the edges in order,
        *edge_sh* ``[E, pad32(sh)]`` TILE and *edge_mask* ``[E, 1]`` TILE. With *ns* set they
        are the bare edge attributes instead, and the node halves of the edge input are built
        here from *node_attr* (:meth:`fc1_split`), so a stack of layers never leaves the card.
        *out_attr* is the table the messages land in when it is not *node_attr* (a cross group
        of the confidence model): the ``x[src]`` half of the edge input is read from it.
        *inv_degree* is :meth:`inv_degree` of this graph and mask when the caller already has it.
        """
        ttnn = self.ttnn
        ckc = self.ckc
        # The tile kernel reads x1 at bf16, so the table is rounded before the gather: the same
        # bits the kernel gets from rounding an exact fp32 copy, for one slice instead of three.
        x1 = gather_rows(ttnn.typecast(node_attr, ttnn.bfloat16) if self.x1_bf16 else node_attr,
                         g["idx"], g["E"])
        if ns is not None:
            sizes = tuple(int(ea.shape[0]) for ea in edge_attr_groups)
            index = self._group_index(g, sizes)
            mm = lambda x, w: ttnn.matmul(x, w, compute_kernel_config=ckc, dtype=self.dtype)
        msgs, lo = [], 0
        for gi, (tp, ea) in enumerate(zip(self.tps, edge_attr_groups)):
            n = ea.shape[0]
            h = None
            if ns is not None:
                w1a, w1s, w1d, b1 = self.fc1_split(ns)[gi]
                src, dst = index[gi]
                h = ttnn.linear(ea, w1a, bias=b1, compute_kernel_config=ckc, dtype=self.dtype)
                out_t = node_attr if out_attr is None else out_attr
                h = ttnn.add(h, gather_rows(mm(out_t, w1s), src, n))
                h = ttnn.relu(ttnn.add(h, gather_rows(mm(node_attr, w1d), dst, n)))
            msgs.append(tp(ea, x1[lo:lo + n, :], edge_sh[lo:lo + n, :], h=h))
            lo += n
        msg = msgs[0] if len(msgs) == 1 else ttnn.concat(msgs, dim=0)
        msg = ttnn.multiply(msg, edge_mask)
        s = ttnn.matmul(g["S"], msg, compute_kernel_config=ckc, dtype=self.dtype)
        if self.perm is not None:
            s = ttnn.matmul(s, self.perm, compute_kernel_config=ckc, dtype=self.dtype)
        out = ttnn.multiply(s, self.inv_degree(g, edge_mask) if inv_degree is None else inv_degree)
        if self.layer.batch_norm is not None:
            out = ttnn.add(ttnn.multiply(out, self.bn_scale), self.bn_shift)
        if self.layer.residual:
            res = node_attr
            if res.shape[-1] != self.out_pad:
                res = ttnn.pad(res, [(0, 0), (0, self.out_pad - res.shape[-1])], value=0.0)
            out = ttnn.add(out, res)
        return out


def pose_blocks(edge_index: Tensor, lo: int, hi: int, node_sets, n_poses: int):
    """Edges ``[lo, hi)`` as *n_poses* copies of one graph, or None when they are not.

    The static graph is one pose tiled over the poses, so the dense ``[N, E]`` scatter of a
    block is the same ``[n, e]`` matrix *n_poses* times on the diagonal, and the gather its
    transpose. This finds that one matrix. *node_sets* are ``(start, n)`` per node type: pose
    ``p`` owns nodes ``start + p * n .. start + (p + 1) * n``. Edges that touch a node outside
    every set are the layout's dead padding and are left out.

    Returns ``rows`` (``[n_poses, e]``, the edge rows of each pose in order), the pose-local
    ``src`` and ``dst`` (``[e]``) and the node set each indexes, ``out_set`` and ``in_set``.
    """
    def locate(nodes):
        which = torch.full_like(nodes, -1)
        pose, local = torch.zeros_like(nodes), torch.zeros_like(nodes)
        for k, (start, n) in enumerate(node_sets):
            hit = (nodes >= start) & (nodes < start + n * n_poses)
            which[hit] = k
            pose[hit] = (nodes[hit] - start) // n
            local[hit] = (nodes[hit] - start) % n
        return which, pose, local

    (ws, ps, ls), (wd, pd, ld) = locate(edge_index[0, lo:hi]), locate(edge_index[1, lo:hi])
    real = (ws >= 0) & (wd >= 0)
    if not real.any() or (ps != pd)[real].any():
        return None
    out_set, in_set = int(ws[real][0]), int(wd[real][0])
    if (ws[real] != out_set).any() or (wd[real] != in_set).any():
        return None
    at = torch.nonzero(real).reshape(-1)
    order = at[torch.sort(ps[at], stable=True).indices]
    if order.numel() % n_poses:
        return None
    rows = order.reshape(n_poses, -1)
    src, dst = ls[rows], ld[rows]
    if (ps[rows] != torch.arange(n_poses)[:, None]).any() or (src != src[0]).any() \
            or (dst != dst[0]).any():
        return None
    return dict(rows=rows + lo, src=src[0], dst=dst[0], out_set=out_set, in_set=in_set)


def stack_layout(edge_index: Tensor, n_p: int, plan, n_in_p: int | None = None):
    """The padded edge layout a stack of convolutions runs over: ``(ei, pos)``.

    A layer sees edges ``[:cut]`` split into FC groups at *splits*, and every block a layer
    reads has to start on a multiple of 32, so the blocks between consecutive cut/split
    boundaries are each padded out with dead ``pad -> pad`` edges. *pos* maps an unpadded
    boundary to its row in the padded layout; *ei* is the padded edge index.

    A dead row points at the pad node of each side: *n_p* out, *n_in_p* in, which differ only
    for a convolution whose destination node set is not its source (the centre head scatters
    into poses, the torsion head into bonds). *n_in_p* defaults to *n_p*.

    Shared by :meth:`DeviceConvModule.run_stack`, which uploads host edge features into this
    layout, and :class:`tt_bio.diffdock.device_loop.ResidentStack`, which keeps them on the
    card -- the two cannot drift apart because the layout is computed in one place.
    """
    n_in_p = n_p if n_in_p is None else n_in_p
    E = edge_index.shape[1]
    bounds = sorted({0, E} | {c for c, _ in plan} | {x for _, sp in plan for x in sp or ()})
    pos, cols, blocks, at = {}, [], [], 0
    for lo, hi in zip(bounds, bounds[1:]):
        pos[lo] = at
        n = hi - lo
        extra = pad32(n) - n
        cols.append(torch.cat([edge_index[:, lo:hi],
                               torch.tensor([[n_p - 1], [n_in_p - 1]]).expand(2, extra)], 1))
        blocks.append((lo, hi, extra))
        at += n + extra
    pos[E] = at
    return torch.cat(cols, 1), pos, blocks
class GraphCache:
    """Uploaded static graphs (gather index + dense ``[n_out, E]`` bf16 scatter), keyed by shape
    and checked by content, least recently used out first.

    The static network reuses one edge index across layers and steps (masks carry what moves),
    so the conv modules of one network share a cache and each graph crosses to the card once
    per rollout instead of once per call. A rollout touches six: the receptor, ligand and main
    stacks, the main stack's last layer and the two heads. A screening run brings new shapes
    with every complex, so past *size* entries the oldest is freed on the card.
    """

    def __init__(self, size: int = 8):
        from collections import OrderedDict
        self.size, self.entries = size, OrderedDict()

    def get(self, key, edge_index: Tensor, build) -> dict:
        hit = self.entries.get(key)
        if hit is None or not torch.equal(hit[0], edge_index):
            if hit is not None:
                _free(hit[1])
            hit = self.entries[key] = (edge_index, build())
        self.entries.move_to_end(key)
        while len(self.entries) > self.size:
            _free(self.entries.popitem(last=False)[1][1])
        return hit[1]

    def nbytes(self) -> int:
        """Card bytes held: per graph the bf16 scatter, its uint32 index and any per-group
        ``(src, dst)`` indices :meth:`DeviceConvLayer._group_index` added."""
        total = 0
        for _, g in self.entries.values():
            groups = sum(1 for k in g if isinstance(k, tuple))
            total += g["n_out"] * g["E"] * 2 + g["E"] * 4 * (1 + 2 * groups)
        return total


def _free(g: dict) -> None:
    import ttnn
    for v in g.values():
        for t in (v if isinstance(v, list) else [v]):
            for u in (t if isinstance(t, list) else [t]):
                if isinstance(u, ttnn.Tensor):
                    ttnn.deallocate(u)


class DeviceConvModule(torch.nn.Module):
    """A host :class:`~tt_bio.diffdock.conv.TensorProductConvLayer` whose arithmetic runs on
    the card: same call signature, so it drops into :class:`~tt_bio.diffdock.score_model.ScoreModel`
    in place of the host layer and the rest of the network does not change.

    Each call pads the node axes to :func:`~tt_bio.diffdock.graph.pad_count` and every edge
    group to a multiple of 32 with dead edges ``pad -> pad`` (mask 0), uploads, runs
    :class:`DeviceConvLayer` and reads the result back. With *grade* set it also runs the host
    layer in float64 on the same inputs and appends ``{pcc, max_rel_err, rel_frobenius}`` to
    :attr:`log`, so a full forward reports every convolution against its own golden.
    """

    def __init__(self, device, layer, *, dtype=None, grade: bool = False, name: str = "",
                 graphs: GraphCache | None = None, edge_tp: str = "gemm"):
        super().__init__()
        import copy
        self.name, self.log = name, []
        #: uploaded static graphs, shared by the modules built together (:class:`GraphCache`)
        self.graphs = GraphCache() if graphs is None else graphs
        #: wall seconds spent inside :meth:, summed over calls. Upload, the device ops
        #: and the read-back: everything this module owns. The rest of a step is host glue, so
        #: 1 - sum(secs) / step is the card's idle fraction in the untraced form.
        self.secs = 0.0
        #: Per-layer device seconds and rows, filled only when :attr: is on.
        self.layer_secs, self.layer_rows = 0.0, 0
        self.host = copy.deepcopy(layer).double() if grade else None
        #: :meth:`run_stack` may carry this layer. Grading needs every layer's input on the host.
        self.resident = not grade
        self.layer = layer
        self.dev = DeviceConvLayer(device, copy.deepcopy(layer).float(), dtype=dtype,
                                   edge_tp=edge_tp)

    #: Set by an attribution script: time and synchronise around every layer of a stack, so
    #: ``layer_secs`` is that layer's own device time rather than the stack's. Off by default,
    #: because the synchronisation is itself a cost no shipped rollout should pay.
    time_layers = False

    def _graph(self, ei: Tensor, n_out_p: int, n_in_p: int) -> dict:
        return self.graphs.get((tuple(ei.shape), n_out_p, n_in_p), ei,
                               lambda: self.dev.graph(ei, n_out_p, n_in_p))

    @staticmethod
    def run_stack(mods, node_attr, edge_index, edge_attr, edge_sh, edge_mask, ns, plan):
        """A stack of convolutions with the node features on the card from first to last.

        The host form (:func:`tt_bio.diffdock.static_net.run_stack`) concatenates
        ``[edge_attr, x[src, :ns], x[dst, :ns]]`` per layer, pads it, uploads it and reads the
        nodes back. Here the edge attributes, harmonics and mask cross once per stack, the edge
        input is built on the card (:meth:`DeviceConvLayer.fc1_split`) and one read-back ends
        it. *plan* is ``[(cut, splits)]`` per layer: the layer sees edges ``[:cut]``, split into
        FC groups at the offsets *splits* (None: one group). Out nodes = in nodes.
        """
        import time
        import ttnn
        from tt_bio.diffdock.graph import pad_count
        t0 = time.perf_counter()
        E, N = edge_index.shape[1], node_attr.shape[0]
        n_p = pad_count(N)
        mask = edge_mask.reshape(E).float()
        ei, pos, blocks = stack_layout(edge_index, n_p, plan)
        attrs, shs, masks = [], [], []
        for lo, hi, extra in blocks:
            attrs.append(torch.nn.functional.pad(edge_attr[lo:hi].float(), (0, 0, 0, extra)))
            shs.append(torch.nn.functional.pad(edge_sh[lo:hi].float(), (0, 0, 0, extra)))
            masks.append(torch.nn.functional.pad(mask[lo:hi], (0, extra)))
        d = mods[0].dev.device
        ea, sh = to_device(d, torch.cat(attrs)), to_device(d, torch.cat(shs))
        m = ttnn.from_torch(torch.cat(masks).reshape(-1, 1), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=d)
        x = to_device(d, torch.nn.functional.pad(node_attr.float(), (0, 0, 0, n_p - N)))
        rows, inv = {}, {}
        rows_of = lambda a, b: rows.setdefault((a, b), (ea[a:b, :], sh[a:b, :], m[a:b, :]))
        for mod, (cut, splits) in zip(mods, plan):
            c = pos[cut]
            edges = [0, *(pos[x] for x in splits or ()), c]
            if len(edges) - 1 != len(mod.dev.tps):
                raise ValueError(f"{mod.name}: {len(edges) - 1} edge groups for "
                                 f"{len(mod.dev.tps)} FC blocks")
            g = mod._graph(ei[:, :c], n_p, n_p)
            t1 = time.perf_counter() if DeviceConvModule.time_layers else None
            if c not in inv:  # layers over the same cut share the graph and the mask
                inv[c] = mod.dev.inv_degree(g, rows_of(0, c)[2])
            x = mod.dev(x, g, [rows_of(a, b)[0] for a, b in zip(edges, edges[1:])],
                        rows_of(0, c)[1], rows_of(0, c)[2], ns=ns, inv_degree=inv[c])
            if t1 is not None:
                # attribution only: without the sync a layer is charged for queueing, not work
                ttnn.synchronize_device(d)
                mod.layer_secs += time.perf_counter() - t1
                mod.layer_rows += c
        out = ttnn.to_torch(x)[:N, :mods[-1].layer.out_irreps.dim].to(node_attr.dtype)
        mods[0].secs += time.perf_counter() - t0
        return out

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None,
                reduce="mean", edge_weight=1.0, edge_mask=None, edge_chunk=None):
        import time
        import ttnn
        from tt_bio.diffdock.graph import pad_count
        _t0 = time.perf_counter()
        if reduce != "mean" or not (isinstance(edge_weight, float) and edge_weight == 1.0):
            raise NotImplementedError("the card layer is the released model's: mean, unweighted")
        E = edge_index.shape[1]
        if E == 0:
            return self.layer(node_attr, edge_index, edge_attr, edge_sh, out_nodes=out_nodes)
        N = node_attr.shape[0]
        n_out = out_nodes or N
        n_in_p, n_out_p = pad_count(N), pad_count(n_out)
        groups = list(edge_attr) if isinstance(edge_attr, (list, tuple)) else [edge_attr]
        mask = torch.ones(E) if edge_mask is None else edge_mask.reshape(E).float()
        cols, attrs, shs, masks, lo = [], [], [], [], 0
        for a in groups:
            n = a.shape[0]
            extra = pad32(n) - n
            ei = torch.cat([edge_index[:, lo:lo + n],
                            torch.tensor([[n_out_p - 1], [n_in_p - 1]]).expand(2, extra)], 1)
            cols.append(ei)
            attrs.append(torch.nn.functional.pad(a.float(), (0, 0, 0, extra)))
            shs.append(torch.nn.functional.pad(edge_sh[lo:lo + n].float(), (0, 0, 0, extra)))
            masks.append(torch.nn.functional.pad(mask[lo:lo + n], (0, extra)))
            lo += n
        ei, sh, m = torch.cat(cols, 1), torch.cat(shs), torch.cat(masks)
        d = self.dev.device
        table = torch.nn.functional.pad(node_attr.float(),
                                        (0, pad32(node_attr.shape[1]) - node_attr.shape[1],
                                         0, n_in_p - N))
        g = self._graph(ei, n_out_p, n_in_p)
        out = self.dev(ttnn.from_torch(table, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=d),
                       g, [to_device(d, a) for a in attrs], to_device(d, sh),
                       ttnn.from_torch(m.reshape(-1, 1), dtype=ttnn.float32,
                                       layout=ttnn.TILE_LAYOUT, device=d))
        out = ttnn.to_torch(out)[:n_out, :self.layer.out_irreps.dim].to(node_attr.dtype)
        self.secs += time.perf_counter() - _t0
        if self.host is not None:
            with torch.no_grad():
                ref = self.host(node_attr.double(), edge_index,
                                [a.double() for a in groups] if len(groups) > 1
                                else groups[0].double(), edge_sh.double(), out_nodes=out_nodes,
                                edge_mask=edge_mask)
            r, o = ref.flatten(), out.double().flatten()
            self.log.append(dict(
                edges=E, nodes=N, out_nodes=n_out,
                pcc=float(torch.corrcoef(torch.stack([r, o]))[0, 1]) if r.numel() > 1 else None,
                max_rel_err=float((o - r).abs().max() / r.abs().max().clamp_min(1e-30)),
                rel_frobenius=float((o - r).norm() / r.norm().clamp_min(1e-30))))
        return out


def convs_on_device(model, device, *, dtype=None, grade: bool = False,
                    edge_tp: str = "gemm", gemm_modules=(), graphs: int = 8) -> dict:
    """Swap every convolution in *model* for a :class:`DeviceConvModule`, in place.

    *edge_tp* is the edge tensor-product form of every convolution except those whose path ends
    in one of *gemm_modules*, which keep the fp32 GEMM form. Returns ``{module path:
    DeviceConvModule}`` so a caller can read each one's grade log. *graphs* sizes the shared
    :class:`GraphCache`.
    """
    from tt_bio.diffdock.conv import TensorProductConvLayer
    swapped, graphs = {}, GraphCache(graphs)
    for path, mod in list(model.named_modules()):
        for name, child in list(mod._modules.items()):
            if isinstance(child, TensorProductConvLayer):
                full = f"{path}.{name}" if path else name
                mod._modules[name] = swapped[full] = DeviceConvModule(
                    device, child, dtype=dtype, grade=grade, name=full, graphs=graphs,
                    edge_tp="gemm" if full.endswith(tuple(gemm_modules)) else edge_tp)
    return swapped


#: Convolutions that keep the fp32 GEMM form when the rest run the edge-tile kernel. The torsion
#: head's bf16 kernel error broke the SE(3) check at step 18 (2 of 51 legs against a float64
#: reference); on GEMM every leg passes (state/dfd/PERF.md, D-dfd-orch-19).
TILE_GEMM_MODULES = ("tor_bond_conv",)


class DeviceScoreNetwork:
    """The score network with its 19 convolutions on a card, every pose of a step in one forward.

    Same interface as :class:`tt_bio.diffdock.score_model.HostScoreNetwork` (``config``,
    ``score_fn``, ``confidence``), so the rollout does not know a card is involved. Two things
    differ from the host class: the convolutions run through :class:`DeviceConvModule`, as one
    device-resident stack per edge set, and a step is one :func:`~tt_bio.diffdock.static_net.static_forward`
    over fixed-shape graphs instead of a loop over poses, which is what makes the shapes the same
    every step. The confidence model's 45 convolutions are swapped the same way, so the ranking
    pass runs the same device layers; its host glue (graph build, encoders, head) is unchanged.
    """

    on_device = True

    def __init__(self, host, device, *, edge_tp: str = "gemm"):
        self.host, self.device = host, device
        self.model, self.config = host.model, host.config
        form = dict(edge_tp=edge_tp, gemm_modules=TILE_GEMM_MODULES if edge_tp == "tile" else ())
        self.convs = convs_on_device(host.model, device, **form)
        # The ranking pass takes the same form as the rollout: on the tile kernel its card time
        # is 2.1-2.5x lower and it ranked three graded pose sets exactly as the host does.
        self.confidence_convs = ({} if host.confidence_model is None
                                 else convs_on_device(host.confidence_model, device,
                                                      graphs=CONFIDENCE_GRAPHS, **form))
        self._confidence = (None if host.confidence_model is None
                            else DeviceConfidence(host.confidence_model, self.confidence_convs))

    def score_fn(self, graph):
        from tt_bio.diffdock.conv import pad_nodes
        from tt_bio.diffdock.static_net import StaticComplex, static_forward

        lig, rec = graph.ligand, graph.receptor
        n_lig = lig["pos"].shape[0]
        rec_x = self.host._receptor_x(rec)
        built = {}

        def score(step):
            P = step.lig_pos.shape[0]
            if P not in built:      # one layout per batch size; the rollout keeps it fixed
                built[P] = StaticComplex.build(lig, rec_x, rec["pos"].float(),
                                               rec["edge_index"], P)
            c = built[P]
            lp = pad_nodes(step.lig_pos.reshape(P * n_lig, 3).float(), c.sg.n_lig_padded)
            with torch.no_grad():
                return static_forward(self.model, c, lp, step.t_tr, step.tr_sigma,
                                      step.rot_sigma, step.tor_sigma,
                                      crop_cutoff=step.crop_cutoff)

        return score

    def trace_body(self, graph, start: Tensor, cfg, bounds, noise):
        """``(body, read_poses)`` for a whole rollout on the card: what a trace capture records.

        ``body(t)`` issues step *t* -- score, guard, perturb and pose update -- into one pose
        buffer on the card and reads nothing back; ``read_poses()`` is the single read after the
        last step. *start* is the ``[n_poses, n_lig, 3]`` randomised start and *noise* the
        sampler's generator, drawn here in `sampling.sample`'s order, so a replay and
        ``sample(self.score_fn(graph), ...)`` take the same trajectory.
        """
        from tt_bio.diffdock.device_loop import DeviceRollout
        from tt_bio.diffdock.static_net import StaticComplex

        lig, rec = graph.ligand, graph.receptor
        c = StaticComplex.build(lig, self.host._receptor_x(rec), rec["pos"].float(),
                                rec["edge_index"], start.shape[0])
        r = DeviceRollout(self.device, self.model, c, self.convs, start, lig["mask_rotate"],
                          cfg, bounds, noise, self.config.crop_beyond)
        return r.body, r.read_poses

    def rollout(self, graph, start: Tensor, cfg, bounds, noise):
        """`sampling.sample` on the card: every step of :meth:`trace_body`, then one read.

        Poses, edge features and noise stay on the card for the whole rollout, where
        ``sample(self.score_fn(graph), ...)`` uploads every pose's edge features each step.
        Same seed, same trajectory. A pose that comes back non-finite is marked failed and
        handed back at its start coordinates, the last finite ones the host holds.
        """
        from tt_bio.diffdock.sampling import Rollout, _nonfinite_rows

        body, read_poses = self.trace_body(graph, start, cfg, bounds, noise)
        for t in range(cfg.steps):
            body(t)
        poses = read_poses()
        failed = _nonfinite_rows(poses, poses.shape[0])
        return Rollout(poses=torch.where(failed[:, None, None], start.float(), poses),
                       failed=failed)

    def confidence(self, graph, poses: Tensor):
        """The host ranking pass with its interaction layers in :class:`DeviceConfidence`."""
        return self.host.confidence(graph, poses, convolve=self._confidence)


#: The confidence edge sets a pose changes: the ligand's radius graph and its two cross sets.
#: The receptor's three are the same for every pose and keep their exact padded size.
POSE_EDGES = ("lig", "lr", "la")

#: A confidence pose touches nine graphs per layer, four of them the receptor's; an LRU of 8
#: would evict each one just before it is needed again.
CONFIDENCE_GRAPHS = 16


class DeviceConfidence:
    """The confidence model's interaction layers with every node table on the card.

    :meth:`ConfidenceModel.convolve <tt_bio.diffdock.confidence_model.ConfidenceModel.convolve>`
    walks the wiring; this object is its executor. Per pose the ligand's node table and its
    three edge sets cross to the card once, the receptor's two node tables and three edge sets
    once per receptor (the same host tensors for every pose and ligand), and one read-back of the
    ligand features ends the pass. The 41 convolutions are the :class:`DeviceConvModule`
    layers :func:`convs_on_device` put on the model, so nothing here is a second implementation.

    A group with no edges contributes zeros, as the host layer does (D-dfd-orch-10).
    """

    def __init__(self, model, convs: dict):
        from tt_bio.diffdock.confidence_model import WIRING
        self.model = model
        self.mods = [convs[f"conv_layers.{i}"] for i in range(len(model.conv_layers))]
        self.device = self.mods[0].dev.device
        #: edge set -> (out set, source set) in its stored direction
        self.ends = {es: (out, src) for out, src, es, flip in WIRING if not flip}
        self._held = {}
        self.first_layer = len(WIRING)
        self._same = {}                   # conv -> (out, src, edges, update), see _conv

    def _keep(self, key, src: Tensor, build):
        """*build()*'s upload for *key*, rebuilt only when *src* is a different host tensor."""
        hit = self._held.get(key)
        if hit is None or hit[0] is not src:
            if hit is not None and hit[1] is not None:
                _free(hit[1] if isinstance(hit[1], dict) else {"t": hit[1]})
            hit = self._held[key] = (src, build())
        return hit[1]

    def _edges(self, es, edges, rows):
        import ttnn
        ei, attr, sh = edges
        E = ei.shape[1]
        if E == 0:
            return None
        a, b = self.ends[es]
        extra = (edge_bucket(E) if es in POSE_EDGES else pad32(E)) - E
        pad = lambda t: torch.nn.functional.pad(t.float(), (0, 0, 0, extra))
        return dict(
            ei=torch.cat([ei, torch.tensor([[rows[a] - 1], [rows[b] - 1]]).expand(2, extra)], 1),
            attr=to_device(self.device, pad(attr)), sh=to_device(self.device, pad(sh)),
            mask=ttnn.from_torch(pad(torch.ones(E, 1)), dtype=ttnn.float32,
                                 layout=ttnn.TILE_LAYOUT, device=self.device))

    def _conv(self, i, out, src, e, flip):
        """Convolution *i*, or the update it gave last time for these same card tensors.

        Only the first layer can repeat: its receptor groups read the receptor's own tables and
        edges, which :meth:`_keep` holds across poses and ligands, so four of its nine
        convolutions (the three largest edge sets) run once per receptor. Every later layer
        reads tables the layer before it made.
        """
        hit = self._same.get(i)
        if hit is None or hit[0] is not out or hit[1] is not src or hit[2] is not e:
            hit = (out, src, e, self._run(i, out, src, e, flip))
            if i < self.first_layer:
                self._same[i] = hit
        return hit[3]

    def _run(self, i, out, src, e, flip):
        import ttnn
        mod = self.mods[i]
        if e is None:
            return ttnn.zeros((int(out.shape[0]), mod.dev.out_pad), dtype=ttnn.float32,
                              layout=ttnn.TILE_LAYOUT, device=self.device)
        ei = torch.flip(e["ei"], dims=[0]) if flip else e["ei"]
        g = mod._graph(ei, int(out.shape[0]), int(src.shape[0]))
        return mod.dev(src, g, [e["attr"]], e["sh"], e["mask"], ns=self.model.ns, out_attr=out)

    @staticmethod
    def _combine(x, updates):
        import ttnn
        width = int(updates[0].shape[-1])
        if int(x.shape[-1]) < width:
            x = ttnn.pad(x, [(0, 0), (0, width - int(x.shape[-1]))], value=0.0)
        for u in updates:
            x = ttnn.add(x, u)
        return x

    def __call__(self, g: dict) -> Tensor:
        """Final ligand features ``[n_lig, width]`` on the host for one pose's graph *g*
        (:meth:`ConfidenceModel.ligand_side`)."""
        import time
        import ttnn
        from tt_bio.diffdock.graph import pad_count
        t0 = time.perf_counter()
        rows = {s: pad_count(x.shape[0]) for s, x in g["nodes"].items()}
        nodes = {s: self._keep(("node", s), x, lambda x=x, s=s: to_device(
                     self.device, torch.nn.functional.pad(x, (0, 0, 0, rows[s] - x.shape[0]))))
                 for s, x in g["nodes"].items()}
        edges = {es: self._keep(("edge", es), e[1], lambda es=es, e=e: self._edges(es, e, rows))
                 for es, e in g["edges"].items()}
        lig = self.model.convolve(nodes, edges, conv=self._conv, combine=self._combine)
        n_lig = g["nodes"]["lig"].shape[0]
        width = self.mods[-1].layer.out_irreps.dim
        out = ttnn.to_torch(lig)[:n_lig, :width].to(g["nodes"]["lig"].dtype)
        self.mods[0].secs += time.perf_counter() - t0
        return out
