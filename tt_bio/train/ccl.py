"""Exact device-side fp32 2D hierarchical CCL collectives over the Blackhole Galaxy Ethernet fabric.

Why built-in ``ttnn.all_reduce`` / ``ttnn.reduce_scatter`` cannot be used for fp32 gradients:
their compiled reduction kernels use 16-bit FPU accumulation (err 5e-3..8e-3 vs fp32 order floor
~1.4e-6). Instead, we move raw 32-bit tiles with ``all_to_all_async_generic(in_dim=1, out_dim=2)``
(each peer's shard lands contiguously along dim 1), slice contiguously in dataflow-only tile mode,
and sum in rank order with exact SFPU fp32 ``ttnn.add``, followed by 2D ``ttnn.all_gather``.

Why ``max_packet_payload_size_bytes = 8192`` is mandatory for float32 ``all_to_all_async_generic``:
``all_to_all_sender_writer.cpp`` packs 2 tiles per fabric packet (``2 * 4096 = 8192 B`` for fp32),
which overflows the default 4352 B ERISC fabric channel buffer and hangs/corrupts the router.
Setting ``FabricRouterConfig.max_packet_payload_size_bytes = 8192`` fits cleanly in Blackhole
ERISC L1 (16 slots/channel) and double-buffers both ``all_to_all_async_generic`` and ``all_gather``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Dict, List, Mapping, Sequence, Tuple

FABRIC_FP32_MAX_PAYLOAD_BYTES = 8192
TILE = 32
DEFAULT_FULL_MESH = (8, 4)

__all__ = [
    "FABRIC_FP32_MAX_PAYLOAD_BYTES",
    "DEFAULT_FULL_MESH",
    "WidthGroupPlan",
    "SmallGroupPlan",
    "ReductionPlan",
    "configure_fp32_ring_fabric",
    "flat_2d_shape",
    "fp32_reduction_order_floor",
    "plan_reduction_groups",
    "exact_reduce_scatter_2d",
    "exact_all_gather_2d",
    "exact_all_reduce_2d",
    "DeviceCCLReducer",
]


def configure_fp32_ring_fabric(max_payload_bytes: int = FABRIC_FP32_MAX_PAYLOAD_BYTES) -> None:
    """Configure FABRIC_1D_RING with a payload budget large enough for 2-tile fp32 packets."""
    if max_payload_bytes < FABRIC_FP32_MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"max_payload_bytes={max_payload_bytes} < {FABRIC_FP32_MAX_PAYLOAD_BYTES} "
            "(2 float32 tiles per all_to_all packet)"
        )
    import ttnn

    rc = ttnn.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = int(max_payload_bytes)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=rc)


def flat_2d_shape(
    num_elements: int,
    mesh_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    cols: int = 10944,
) -> Tuple[int, int, int, int]:
    """Return minimal tile-aligned ``(1, 1, R, C)`` shape divisible by ``mesh_shape`` along dim 2."""
    if num_elements <= 0:
        raise ValueError(f"num_elements must be > 0, got {num_elements}")
    rm, cm = mesh_shape
    if rm <= 0 or cm <= 0:
        raise ValueError(f"invalid mesh_shape {mesh_shape}")
    if cols <= 0 or cols % TILE != 0:
        raise ValueError(f"cols={cols} must be a positive multiple of {TILE}")
    row_quantum = TILE * rm * cm
    rows_needed = math.ceil(num_elements / cols)
    rows = math.ceil(rows_needed / row_quantum) * row_quantum
    return (1, 1, rows, cols)


def fp32_reduction_order_floor(ranks_tensor) -> Tuple[object, float]:
    """Return ``(serial_sum, floor)`` where ``floor`` is max |order_sum - serial_sum|.

    ``ranks_tensor`` has shape ``(N, ...)`` in float32. ``floor`` is taken over reversed rank
    order and balanced pairwise-tree order, matching the D89 pre-registered accuracy bar.
    """
    import torch

    n = int(ranks_tensor.shape[0])
    ser = ranks_tensor[0].clone()
    for i in range(1, n):
        ser.add_(ranks_tensor[i])
    rev = ranks_tensor[n - 1].clone()
    for i in range(n - 2, -1, -1):
        rev.add_(ranks_tensor[i])
    tree = [ranks_tensor[i].clone() for i in range(n)]
    while len(tree) > 1:
        tree = [
            tree[i] + tree[i + 1] if i + 1 < len(tree) else tree[i]
            for i in range(0, len(tree), 2)
        ]
    floor = max(float((rev - ser).abs().max()), float((tree[0] - ser).abs().max()))
    return ser, floor


@dataclass(frozen=True)
class WidthGroupPlan:
    """One tile-aligned width group reduced in a single ``exact_all_reduce_2d`` call."""

    cols: int
    total_rows: int
    padded_rows: int
    entries: Tuple[Tuple[str, Tuple[int, ...], int, int], ...]  # (name, orig_shape, row_off, rows)


@dataclass(frozen=True)
class SmallGroupPlan:
    """One exact ``(m, n)`` group of small/non-32-aligned tensors reduced via 4D ``all_gather+add``."""

    m: int
    n: int
    total_batch: int
    entries: Tuple[Tuple[str, Tuple[int, ...], int, int], ...]  # (name, orig_shape, b_off, b_count)


@dataclass(frozen=True)
class ReductionPlan:
    """Zero-untilize, tile-preserving CCL packing plan for an arbitrary parameter dict."""

    width_groups: Tuple[WidthGroupPlan, ...]
    small_groups: Tuple[SmallGroupPlan, ...]
    skipped: Tuple[str, ...]


def _canonical_small_4d(shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Map any 1D..4D shape to ``(b, m, n)`` so ``[1, b, m, n]`` preserves tile boundaries."""
    if len(shape) == 0:
        return (1, 1, 1)
    if len(shape) == 1:
        return (1, 1, int(shape[0]))
    if len(shape) == 2:
        return (1, int(shape[0]), int(shape[1]))
    b = int(math.prod(shape[:-2]))
    return (b, int(shape[-2]), int(shape[-1]))


def plan_reduction_groups(
    shapes: Mapping[str, Tuple[int, ...]],
    mesh_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    *,
    replicated_names: Collection[str] = (),
    min_2d_group_elems: int = 1_048_576,
) -> ReductionPlan:
    """Group parameter shapes into minimal tile-aligned 2D CCL groups plus small 4D groups.

    Every tensor with ``ndim >= 2`` and ``M % 32 == 0, N % 32 == 0`` is grouped by column width
    ``N`` so ``reshape([1, 1, rows, N]) + concat(dim=2)`` preserves 32x32 tiles with zero untilize.
    Width groups whose total elements meet ``min_2d_group_elems`` (or already span at least one
    ``32 * rm * cm`` row quantum) run through ``exact_all_reduce_2d``; smaller or non-32-aligned
    tensors are grouped by exact ``(m, n)`` along non-tile ``dim=1`` and reduced via 4D
    ``all_gather(dim=0) + exact fp32 add``.
    """
    rm, cm = mesh_shape
    row_quantum = TILE * rm * cm
    rep = set(replicated_names)
    skipped: List[str] = []
    by_cols: Dict[int, List[Tuple[str, Tuple[int, ...], int]]] = {}
    by_small: Dict[Tuple[int, int], List[Tuple[str, Tuple[int, ...], int]]] = {}

    for name in sorted(shapes):
        if name in rep:
            skipped.append(name)
            continue
        shp = tuple(int(d) for d in shapes[name])
        if len(shp) >= 2 and shp[-1] % TILE == 0 and shp[-2] % TILE == 0:
            rows = int(math.prod(shp[:-1]))
            by_cols.setdefault(shp[-1], []).append((name, shp, rows))
        else:
            b, m, n = _canonical_small_4d(shp)
            by_small.setdefault((m, n), []).append((name, shp, b))

    width_groups: List[WidthGroupPlan] = []
    for cols in sorted(by_cols, key=lambda c: -sum(r * c for _, _, r in by_cols[c])):
        items = by_cols[cols]
        total_rows = sum(r for _, _, r in items)
        total_elems = total_rows * cols
        if total_elems >= min_2d_group_elems or total_rows >= row_quantum:
            padded_rows = math.ceil(total_rows / row_quantum) * row_quantum
            entries = []
            off = 0
            for nm, shp, r in items:
                entries.append((nm, shp, off, r))
                off += r
            width_groups.append(
                WidthGroupPlan(
                    cols=cols,
                    total_rows=total_rows,
                    padded_rows=padded_rows,
                    entries=tuple(entries),
                )
            )
        else:
            # Tiny tile-aligned group (e.g. 3x (256, 512)): cheaper via small 4D group than padding to 1024 rows.
            for nm, shp, _r in items:
                b, m, n = _canonical_small_4d(shp)
                by_small.setdefault((m, n), []).append((nm, shp, b))

    small_groups: List[SmallGroupPlan] = []
    for (m, n) in sorted(by_small):
        items = by_small[(m, n)]
        entries = []
        off = 0
        for nm, shp, b in items:
            entries.append((nm, shp, off, b))
            off += b
        small_groups.append(SmallGroupPlan(m=m, n=n, total_batch=off, entries=tuple(entries)))

    return ReductionPlan(
        width_groups=tuple(width_groups),
        small_groups=tuple(small_groups),
        skipped=tuple(skipped),
    )


def _safe_dealloc(t, keep=None) -> None:
    import ttnn

    if t is None:
        return
    if keep is not None:
        if t is keep:
            return
        try:
            if t.buffer_address() == keep.buffer_address():
                return
        except Exception:
            pass
    if t.is_allocated():
        ttnn.deallocate(t)


def _concat_chunked(tensors: Sequence, dim: int, chunk: int = 24):
    """Bit-exact chunked ``ttnn.concat`` along ``dim`` (never deallocates caller inputs)."""
    import ttnn

    if len(tensors) == 1:
        return ttnn.clone(tensors[0])
    if len(tensors) <= chunk:
        return ttnn.concat(list(tensors), dim=dim)
    mid = [_concat_chunked(tensors[i:i + chunk], dim=dim, chunk=chunk) for i in range(0, len(tensors), chunk)]
    out = _concat_chunked(mid, dim=dim, chunk=chunk)
    for t in mid:
        ttnn.deallocate(t)
    return out


def _exact_rs_axis(cur, axis: int, n_axis: int, is_full_ring: bool, num_links: int):
    import ttnn

    _, _, rows, cols = cur.shape
    if rows % (TILE * n_axis) != 0:
        raise ValueError(
            f"dim 2 ({rows}) must be divisible by {TILE * n_axis} for tile-aligned RS on axis {axis}"
        )
    topo = ttnn.Topology.Ring if is_full_ring else ttnn.Topology.Linear
    y = ttnn.experimental.all_to_all_async_generic(
        cur, in_dim=1, out_dim=2, cluster_axis=axis, topology=topo, num_links=num_links
    )
    r_out = rows // n_axis
    acc = ttnn.slice(y, [0, 0, 0, 0], [1, 1, r_out, cols])
    for i in range(1, n_axis):
        sl = ttnn.slice(y, [0, i, 0, 0], [1, i + 1, r_out, cols])
        nxt = ttnn.add(acc, sl)
        acc.deallocate()
        sl.deallocate()
        acc = nxt
    y.deallocate()
    return acc


def exact_reduce_scatter_2d(
    tensor,
    mesh_shape: Tuple[int, int],
    *,
    full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    num_links: int = 2,
    force_linear: bool = False,
):
    """Exact fp32 2D reduce-scatter along axis 0 (size rm) then axis 1 (size cm)."""
    rm, cm = mesh_shape
    full_r, full_c = full_shape
    cur = tensor
    if rm > 1:
        cur = _exact_rs_axis(cur, 0, rm, (rm == full_r) and not force_linear, num_links)
    if cm > 1:
        nxt = _exact_rs_axis(cur, 1, cm, (cm == full_c) and not force_linear, num_links)
        if cur is not tensor:
            cur.deallocate()
        cur = nxt
    return cur


def _ag_axis(
    cur,
    axis: int,
    is_full_ring: bool,
    num_links: int,
    *,
    ag_ring: bool = True,
    chunks_per_sync: int = 20,
    num_workers_per_link: int = 2,
    num_buffers_per_channel: int = 2,
):
    import ttnn

    topo = ttnn.Topology.Ring if (is_full_ring and ag_ring) else ttnn.Topology.Linear
    return ttnn.all_gather(
        cur,
        dim=2,
        cluster_axis=axis,
        topology=topo,
        num_links=num_links,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )


def exact_all_gather_2d(
    tensor,
    mesh_shape: Tuple[int, int],
    *,
    full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    num_links: int = 2,
    ag_ring: bool = True,
    chunks_per_sync: int = 20,
    num_workers_per_link: int = 2,
    num_buffers_per_channel: int = 2,
    deallocate_input: bool = True,
):
    """2D all-gather along axis 1 then axis 0 restoring ``[1, 1, R, C]``."""
    rm, cm = mesh_shape
    full_r, full_c = full_shape
    cur = tensor
    if cm > 1:
        g1 = _ag_axis(
            cur, 1, cm == full_c, num_links,
            ag_ring=ag_ring, chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=num_buffers_per_channel,
        )
        if deallocate_input or cur is not tensor:
            cur.deallocate()
        cur = g1
    if rm > 1:
        g0 = _ag_axis(
            cur, 0, rm == full_r, num_links,
            ag_ring=ag_ring, chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=num_buffers_per_channel,
        )
        if deallocate_input or cur is not tensor:
            cur.deallocate()
        cur = g0
    return cur


def exact_all_reduce_2d(
    tensor,
    mesh_shape: Tuple[int, int],
    *,
    full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    num_links: int = 2,
    ag_ring: bool = True,
    force_linear: bool = False,
):
    """Exact fp32 2D all-reduce (2D RS + 2D AG) over a Blackhole Galaxy mesh."""
    rs = exact_reduce_scatter_2d(
        tensor, mesh_shape, full_shape=full_shape, num_links=num_links, force_linear=force_linear
    )
    return exact_all_gather_2d(
        rs, mesh_shape, full_shape=full_shape, num_links=num_links, ag_ring=ag_ring, deallocate_input=True
    )


def _exact_small_all_reduce_4d(
    tensor_4d,
    mesh_shape: Tuple[int, int],
    *,
    full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    ag_ring: bool = True,
    force_linear: bool = False,
):
    """Exact fp32 2D all-reduce for ``[1, B, M, N]`` tensors via ``all_gather(dim=0) + add``."""
    import ttnn

    rm, cm = mesh_shape
    full_r, full_c = full_shape
    _, b, m, n = (int(d) for d in tensor_4d.shape)
    cur = tensor_4d
    if rm > 1:
        topo0 = ttnn.Topology.Ring if (rm == full_r and ag_ring and not force_linear) else ttnn.Topology.Linear
        g0 = ttnn.all_gather(cur, dim=0, cluster_axis=0, topology=topo0, num_links=1)
        s0 = ttnn.slice(g0, [0, 0, 0, 0], [1, b, m, n])
        for i in range(1, rm):
            sl = ttnn.slice(g0, [i, 0, 0, 0], [i + 1, b, m, n])
            nxt = ttnn.add(s0, sl)
            s0.deallocate()
            sl.deallocate()
            s0 = nxt
        g0.deallocate()
        cur = s0
    if cm > 1:
        topo1 = ttnn.Topology.Ring if (cm == full_c and ag_ring and not force_linear) else ttnn.Topology.Linear
        g1 = ttnn.all_gather(cur, dim=0, cluster_axis=1, topology=topo1, num_links=1)
        s1 = ttnn.slice(g1, [0, 0, 0, 0], [1, b, m, n])
        for i in range(1, cm):
            sl = ttnn.slice(g1, [i, 0, 0, 0], [i + 1, b, m, n])
            nxt = ttnn.add(s1, sl)
            s1.deallocate()
            sl.deallocate()
            s1 = nxt
        g1.deallocate()
        if cur is not tensor_4d:
            cur.deallocate()
        cur = s1
    return cur


class DeviceCCLReducer:
    """Device-resident gradient all-reducer compatible with ``DeviceOptimizer.step(reduce_fn=...)``.

    Groups all tile-aligned 2D+ parameter gradients by column width ``N`` into a handful of flat
    ``[1, 1, R_pad, N]`` buffers (4 calls covering >99% of RFD4's 178.4M parameters) and reduces
    them via :func:`exact_all_reduce_2d`, while grouping smaller / non-32-aligned tensors by exact
    ``(m, n)`` along non-tile ``dim=1`` into exact 4D ``all_gather(dim=0) + fp32 add`` calls.
    Zero host round-trips, zero untilize, exact fp32 accumulation throughout.
    """

    def __init__(
        self,
        mesh_shape: Tuple[int, int],
        *,
        full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
        num_links: int = 2,
        ag_ring: bool = True,
        force_linear: bool = False,
        scale: float = 1.0,
        replicated_names: Collection[str] = (),
        min_2d_group_elems: int = 1_048_576,
    ) -> None:
        self.mesh_shape = tuple(int(v) for v in mesh_shape)
        self.full_shape = tuple(int(v) for v in full_shape)
        self.num_links = int(num_links)
        self.ag_ring = bool(ag_ring)
        self.force_linear = bool(force_linear)
        self.scale = float(scale)
        self.replicated_names = frozenset(replicated_names)
        self.min_2d_group_elems = int(min_2d_group_elems)
        self._plan_cache: Dict[Tuple[Tuple[str, Tuple[int, ...]], ...], ReductionPlan] = {}
        self._pad_zeros: Dict[Tuple[int, int, int], object] = {}

    def _zero_pad(self, device, pad_rows: int, cols: int):
        import ttnn

        key = (id(device), pad_rows, cols)
        z = self._pad_zeros.get(key)
        if z is None or not z.is_allocated():
            z = ttnn.zeros([1, 1, pad_rows, cols], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            self._pad_zeros[key] = z
        return z

    def deallocate_caches(self) -> None:
        import ttnn

        for z in self._pad_zeros.values():
            if z is not None and z.is_allocated():
                ttnn.deallocate(z)
        self._pad_zeros.clear()

    def reduce_tensor(self, tensor):
        import ttnn

        rm, cm = self.mesh_shape
        if rm * cm <= 1:
            return tensor
        orig_shape = tuple(int(d) for d in tensor.shape)
        if (
            len(orig_shape) == 4
            and orig_shape[0] == 1
            and orig_shape[1] == 1
            and orig_shape[2] % (TILE * rm * cm) == 0
            and orig_shape[3] % TILE == 0
        ):
            out = exact_all_reduce_2d(
                tensor,
                self.mesh_shape,
                full_shape=self.full_shape,
                num_links=self.num_links,
                ag_ring=self.ag_ring,
                force_linear=self.force_linear,
            )
        else:
            b, m, n = _canonical_small_4d(orig_shape)
            t4 = ttnn.reshape(tensor, [1, b, m, n])
            r4 = _exact_small_all_reduce_4d(
                t4,
                self.mesh_shape,
                full_shape=self.full_shape,
                ag_ring=self.ag_ring,
                force_linear=self.force_linear,
            )
            _safe_dealloc(t4, keep=tensor)
            out = ttnn.reshape(r4, list(orig_shape))
            _safe_dealloc(r4, keep=out)
        if self.scale != 1.0:
            scaled = ttnn.multiply(out, self.scale)
            out.deallocate()
            out = scaled
        return out

    def _get_plan(self, active_shapes: Mapping[str, Tuple[int, ...]]) -> ReductionPlan:
        key = tuple(sorted(active_shapes.items()))
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = plan_reduction_groups(
                active_shapes,
                self.mesh_shape,
                replicated_names=self.replicated_names,
                min_2d_group_elems=self.min_2d_group_elems,
            )
            self._plan_cache[key] = plan
        return plan

    def __call__(self, grads: Mapping[str, object]) -> Dict[str, object]:
        import ttnn

        rm, cm = self.mesh_shape
        if rm * cm <= 1:
            return dict(grads)

        out: Dict[str, object] = {}
        active_shapes: Dict[str, Tuple[int, ...]] = {}
        for name, g in grads.items():
            if g is None:
                out[name] = None
            else:
                active_shapes[name] = tuple(int(d) for d in g.shape)

        if not active_shapes:
            return out

        plan = self._get_plan(active_shapes)
        for name in plan.skipped:
            out[name] = grads[name]

        # 1. Large tile-aligned width groups: one exact_all_reduce_2d per column width.
        for wg in plan.width_groups:
            views = []
            pieces = []
            dev = None
            for name, _shp, _off, rows in wg.entries:
                g = grads[name]
                if dev is None:
                    dev = g.device()
                v = ttnn.reshape(g, [1, 1, rows, wg.cols])
                views.append((v, g))
                pieces.append(v)
            pad_r = wg.padded_rows - wg.total_rows
            if pad_r > 0:
                pieces.append(self._zero_pad(dev, pad_r, wg.cols))
            packed = _concat_chunked(pieces, dim=2)
            for v, orig in views:
                _safe_dealloc(v, keep=orig)

            reduced = exact_all_reduce_2d(
                packed,
                self.mesh_shape,
                full_shape=self.full_shape,
                num_links=self.num_links,
                ag_ring=self.ag_ring,
                force_linear=self.force_linear,
            )
            ttnn.deallocate(packed)
            if self.scale != 1.0:
                scaled = ttnn.multiply(reduced, self.scale)
                ttnn.deallocate(reduced)
                reduced = scaled

            for name, orig_shp, off, rows in wg.entries:
                if len(wg.entries) == 1 and pad_r == 0:
                    sl = reduced
                    reduced = None
                else:
                    sl = ttnn.slice(reduced, [0, 0, off, 0], [1, 1, off + rows, wg.cols])
                res = ttnn.reshape(sl, list(orig_shp))
                _safe_dealloc(sl, keep=res)
                out[name] = res
            if reduced is not None and reduced.is_allocated():
                ttnn.deallocate(reduced)

        # 2. Small / non-32-aligned groups: one 4D all_gather(dim=0) + fp32 add per (m, n).
        for sg in plan.small_groups:
            views = []
            pieces = []
            for name, _shp, _off, b in sg.entries:
                g = grads[name]
                v = ttnn.reshape(g, [1, b, sg.m, sg.n])
                views.append((v, g))
                pieces.append(v)
            packed = _concat_chunked(pieces, dim=1)
            for v, orig in views:
                _safe_dealloc(v, keep=orig)

            reduced = _exact_small_all_reduce_4d(
                packed,
                self.mesh_shape,
                full_shape=self.full_shape,
                ag_ring=self.ag_ring,
                force_linear=self.force_linear,
            )
            ttnn.deallocate(packed)
            if self.scale != 1.0:
                scaled = ttnn.multiply(reduced, self.scale)
                ttnn.deallocate(reduced)
                reduced = scaled

            for name, orig_shp, off, b in sg.entries:
                if len(sg.entries) == 1:
                    sl = reduced
                    reduced = None
                else:
                    sl = ttnn.slice(reduced, [0, off, 0, 0], [1, off + b, sg.m, sg.n])
                res = ttnn.reshape(sl, list(orig_shp))
                _safe_dealloc(sl, keep=res)
                out[name] = res
            if reduced is not None and reduced.is_allocated():
                ttnn.deallocate(reduced)

        return out
