"""Exact device-side fp32 2D hierarchical CCL collectives over the Blackhole Galaxy Ethernet fabric.

Why built-in ``ttnn.all_reduce`` / ``ttnn.reduce_scatter`` cannot be used for fp32 gradients:
their compiled reduction kernels use 16-bit FPU accumulation (err 5e-3..8e-3 vs fp32 order floor
~1.4e-6). Instead, we move raw 32-bit tiles with ``all_to_all_async_generic(in_dim=1, out_dim=2)``
(each peer's shard lands contiguously along dim 1), slice contiguously in dataflow-only tile mode,
and sum in exact SFPU fp32 binary-tree order with ``ttnn.add``, followed by 2D ``ttnn.all_gather``.

Why ``max_packet_payload_size_bytes = 8192`` is mandatory for float32 ``all_to_all_async_generic``:
``all_to_all_sender_writer.cpp`` packs 2 tiles per fabric packet (``2 * 4096 = 8192 B`` for fp32),
which overflows the default 4352 B ERISC fabric channel buffer and hangs/corrupts the router.
Setting ``FabricRouterConfig.max_packet_payload_size_bytes = 8192`` fits cleanly in Blackhole
ERISC L1 (16 slots/channel) and double-buffers both ``all_to_all_async_generic`` and ``all_gather``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

FABRIC_FP32_MAX_PAYLOAD_BYTES = 8192
TILE = 32
DEFAULT_FULL_MESH = (8, 4)

__all__ = [
    "FABRIC_FP32_MAX_PAYLOAD_BYTES",
    "DEFAULT_FULL_MESH",
    "PaddedSmallSlot",
    "WidthGroupPlan",
    "SmallGroupPlan",
    "ReductionPlan",
    "SPMDMuonBucketSlot",
    "SPMDMuonWidthGroup",
    "SPMDMuonPlan",
    "configure_fp32_ring_fabric",
    "flat_2d_shape",
    "exact_2d_fp32_sum",
    "fp32_reduction_order_floor",
    "plan_reduction_groups",
    "plan_spmd_muon_shards",
    "exact_reduce_scatter_2d",
    "exact_all_gather_axis",
    "exact_all_gather_2d",
    "exact_all_reduce_2d",
    "DeviceCCLReducer",
    "SPMDMuonSharder",
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


def _tree_sum_cpu(x, dim: int):
    """Exact IEEE-754 fp32 halving-tree sum along ``dim`` matching ``_tree_reduce_dim_tt``."""
    import torch

    cur = x
    while int(cur.shape[dim]) > 1:
        n = int(cur.shape[dim])
        half = n // 2
        s = cur.narrow(dim, 0, half) + cur.narrow(dim, half, half)
        if n % 2 == 1:
            s = torch.cat([s, cur.narrow(dim, 2 * half, 1)], dim=dim)
        cur = s
    return cur.squeeze(dim)


def exact_2d_fp32_sum(
    ranks_tensor,
    mesh_shape: Tuple[int, int],
    *,
    tree: bool = True,
):
    """Exact CPU IEEE-754 fp32 2D sum (axis 0 then axis 1) matching hardware to 0.0 ULP.

    ``ranks_tensor`` has shape ``(rm * cm, ...)`` in float32. When ``tree=True`` (matching
    :func:`exact_reduce_scatter_2d`), each axis sums via exact halving pairs; when ``tree=False``,
    each axis sums sequentially ``0..n-1``.
    """
    rm, cm = int(mesh_shape[0]), int(mesh_shape[1])
    if int(ranks_tensor.shape[0]) != rm * cm:
        raise ValueError(
            f"ranks_tensor leading dim {ranks_tensor.shape[0]} != rm*cm ({rm}*{cm}={rm * cm})"
        )
    x2d = ranks_tensor.reshape(rm, cm, *ranks_tensor.shape[1:])
    if tree:
        s0 = _tree_sum_cpu(x2d, 0) if rm > 1 else x2d[0].clone()
        return _tree_sum_cpu(s0, 0) if cm > 1 else s0[0].clone()
    s0 = x2d[0].clone()
    for i in range(1, rm):
        s0 = s0 + x2d[i]
    s1 = s0[0].clone()
    for j in range(1, cm):
        s1 = s1 + s0[j]
    return s1


def fp32_reduction_order_floor(ranks_tensor) -> Tuple[object, float]:
    """Return ``(serial_sum, floor)`` where ``floor`` is max |order_sum - serial_sum|.

    ``ranks_tensor`` has shape ``(N, ...)`` in float32. ``floor`` is taken over four independent
    valid IEEE-754 fp32 addition orders (reversed, pairwise tree, even-then-odd, and stride-4
    grouped) without referencing any hardware order, matching the D89 pre-registered accuracy bar.
    """
    n = int(ranks_tensor.shape[0])
    ser = ranks_tensor[0].clone()
    for i in range(1, n):
        ser.add_(ranks_tensor[i])
    if n <= 1:
        return ser, 0.0

    # 1. Reversed rank order
    rev = ranks_tensor[n - 1].clone()
    for i in range(n - 2, -1, -1):
        rev.add_(ranks_tensor[i])

    # 2. Adjacent pairwise tree order
    tree = [ranks_tensor[i].clone() for i in range(n)]
    while len(tree) > 1:
        tree = [
            tree[i] + tree[i + 1] if i + 1 < len(tree) else tree[i]
            for i in range(0, len(tree), 2)
        ]

    # 3. Even-then-odd order (two interleaved groups)
    evens = ranks_tensor[0].clone()
    for i in range(2, n, 2):
        evens.add_(ranks_tensor[i])
    odds = ranks_tensor[1].clone()
    for i in range(3, n, 2):
        odds.add_(ranks_tensor[i])
    even_odd = evens + odds

    # 4. Stride-4 grouped order (independent of mesh shape, valid for any N >= 4)
    candidates = [rev, tree[0], even_odd]
    if n >= 4:
        g4 = []
        for r in range(4):
            acc = ranks_tensor[r].clone()
            for i in range(r + 4, n, 4):
                acc.add_(ranks_tensor[i])
            g4.append(acc)
        s4 = g4[0]
        for acc in g4[1:]:
            s4 = s4 + acc
        candidates.append(s4)

    floor = max(float((c - ser).abs().max()) for c in candidates)
    return ser, floor


@dataclass(frozen=True)
class PaddedSmallSlot:
    """One ``(m, n)`` small/non-32-aligned group folded into a ``WidthGroupPlan`` via pure-TILE pad."""

    m: int
    n: int
    m_pad: int
    total_batch: int
    row_off: int
    padded_rows: int  # == total_batch * m_pad
    entries: Tuple[Tuple[str, Tuple[int, ...], int, int], ...]  # (name, orig_shape, b_off, b_count)


@dataclass(frozen=True)
class WidthGroupPlan:
    """One tile-aligned width group reduced in a single ``exact_all_reduce_2d`` call."""

    cols: int
    total_rows: int
    padded_rows: int
    entries: Tuple[Tuple[str, Tuple[int, ...], int, int], ...]  # (name, orig_shape, row_off, rows)
    folded_smalls: Tuple[PaddedSmallSlot, ...] = ()


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
    fold_small_into_width: bool = True,
    max_fold_elems: int = 4_194_304,
) -> ReductionPlan:
    """Group parameter shapes into minimal tile-aligned 2D CCL groups plus fallback small 4D groups.

    Every tensor with ``ndim >= 2`` and ``M % 32 == 0, N % 32 == 0`` is grouped by column width
    ``N`` so ``reshape([1, 1, rows, N]) + concat(dim=2)`` preserves 32x32 tiles with zero untilize.
    When ``fold_small_into_width=True`` (default), smaller / non-32-aligned ``(m, n)`` groups whose
    padded footprint fits within ``max_fold_elems`` are folded into the smallest existing width
    group with ``cols >= n`` via pure-TILE ``ttnn.pad`` + zero-copy ``reshape``, collapsing all
    558 RFD4 parameters into just 4 2D CCL calls (cutting cold compile ~5x and warm latency ~40%).
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

    # Mutable accumulators per promoted width group
    wg_builders: List[dict] = []
    for cols in sorted(by_cols, key=lambda c: -sum(r * c for _, _, r in by_cols[c])):
        items = by_cols[cols]
        total_rows = sum(r for _, _, r in items)
        total_elems = total_rows * cols
        if total_elems >= min_2d_group_elems or total_rows >= row_quantum:
            entries = []
            off = 0
            for nm, shp, r in items:
                entries.append((nm, shp, off, r))
                off += r
            wg_builders.append(
                {
                    "cols": cols,
                    "total_rows": total_rows,
                    "entries": tuple(entries),
                    "folded": [],
                }
            )
        else:
            for nm, shp, _r in items:
                b, m, n = _canonical_small_4d(shp)
                by_small.setdefault((m, n), []).append((nm, shp, b))

    # Fold small groups into the narrowest width group with cols >= n (zero extra CCL calls).
    wg_by_cols = {wb["cols"]: wb for wb in wg_builders}
    avail_cols = sorted(wg_by_cols)
    small_groups: List[SmallGroupPlan] = []

    for (m, n) in sorted(by_small):
        items = by_small[(m, n)]
        entries = []
        off = 0
        for nm, shp, b in items:
            entries.append((nm, shp, off, b))
            off += b
        total_batch = off

        folded = False
        if fold_small_into_width and avail_cols:
            m_pad = math.ceil(m / TILE) * TILE
            target_cols = next((c for c in avail_cols if c >= n), None)
            if target_cols is not None:
                slot_rows = total_batch * m_pad
                if slot_rows * target_cols <= max_fold_elems:
                    wb = wg_by_cols[target_cols]
                    row_off = wb["total_rows"]
                    wb["folded"].append(
                        PaddedSmallSlot(
                            m=m,
                            n=n,
                            m_pad=m_pad,
                            total_batch=total_batch,
                            row_off=row_off,
                            padded_rows=slot_rows,
                            entries=tuple(entries),
                        )
                    )
                    wb["total_rows"] = row_off + slot_rows
                    folded = True
        if not folded:
            small_groups.append(
                SmallGroupPlan(m=m, n=n, total_batch=total_batch, entries=tuple(entries))
            )

    width_groups = tuple(
        WidthGroupPlan(
            cols=wb["cols"],
            total_rows=wb["total_rows"],
            padded_rows=math.ceil(wb["total_rows"] / row_quantum) * row_quantum,
            entries=wb["entries"],
            folded_smalls=tuple(wb["folded"]),
        )
        for wb in wg_builders
    )

    return ReductionPlan(
        width_groups=width_groups,
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
    mid = [
        _concat_chunked(tensors[i:i + chunk], dim=dim, chunk=chunk)
        for i in range(0, len(tensors), chunk)
    ]
    out = _concat_chunked(mid, dim=dim, chunk=chunk)
    for t in mid:
        ttnn.deallocate(t)
    return out


def _tree_reduce_dim_tt(tensor_4d, dim: int, n_slices: int):
    """Exact fp32 binary-tree sum along non-tile ``dim`` (0 or 1) of a 4D TILE tensor.

    Caller owns ``tensor_4d``; returns a newly allocated tensor of size 1 along ``dim``.
    """
    import ttnn

    shape = [int(d) for d in tensor_4d.shape]
    if n_slices <= 1:
        return ttnn.clone(tensor_4d)
    cur = tensor_4d
    cur_n = n_slices
    while cur_n > 1:
        half = cur_n // 2
        beg0 = [0, 0, 0, 0]
        end0 = list(shape)
        end0[dim] = half
        beg1 = [0, 0, 0, 0]
        beg1[dim] = half
        end1 = list(shape)
        end1[dim] = 2 * half
        left = ttnn.slice(cur, beg0, end0)
        right = ttnn.slice(cur, beg1, end1)
        s = ttnn.add(left, right)
        left.deallocate()
        right.deallocate()
        if cur_n % 2 == 1:
            begt = [0, 0, 0, 0]
            begt[dim] = 2 * half
            endt = list(shape)
            endt[dim] = cur_n
            tail = ttnn.slice(cur, begt, endt)
            nxt = ttnn.concat([s, tail], dim=dim)
            s.deallocate()
            tail.deallocate()
            s = nxt
            cur_n = half + 1
        else:
            cur_n = half
        if cur is not tensor_4d:
            cur.deallocate()
        cur = s
    return cur


def _exact_rs_axis(cur, axis: int, n_axis: int, is_full_ring: bool, num_links: int):
    import ttnn

    _, _, rows, cols = (int(d) for d in cur.shape)
    if rows % (TILE * n_axis) != 0:
        raise ValueError(
            f"dim 2 ({rows}) must be divisible by {TILE * n_axis} for tile-aligned RS on axis {axis}"
        )
    topo = ttnn.Topology.Ring if is_full_ring else ttnn.Topology.Linear
    y = ttnn.experimental.all_to_all_async_generic(
        cur, in_dim=1, out_dim=2, cluster_axis=axis, topology=topo, num_links=num_links
    )
    acc = _tree_reduce_dim_tt(y, dim=1, n_slices=n_axis)
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


def exact_all_gather_axis(
    tensor,
    axis: int,
    mesh_shape: Tuple[int, int],
    *,
    full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    num_links: int = 2,
    ag_ring: bool = True,
    force_linear: bool = False,
    chunks_per_sync: int = 20,
    num_workers_per_link: int = 2,
    num_buffers_per_channel: int = 2,
    deallocate_input: bool = False,
):
    """Single-axis ``all_gather(dim=2)`` along mesh ``axis`` (0 or 1)."""
    import ttnn

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    n_ax = int(mesh_shape[axis])
    if n_ax <= 1:
        return tensor if deallocate_input else ttnn.clone(tensor)
    is_full = (n_ax == int(full_shape[axis])) and ag_ring and not force_linear
    topo = ttnn.Topology.Ring if is_full else ttnn.Topology.Linear
    out = ttnn.all_gather(
        tensor,
        dim=2,
        cluster_axis=axis,
        topology=topo,
        num_links=num_links,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    if deallocate_input:
        tensor.deallocate()
    return out


def exact_all_gather_2d(
    tensor,
    mesh_shape: Tuple[int, int],
    *,
    full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    num_links: int = 2,
    ag_ring: bool = True,
    force_linear: bool = False,
    chunks_per_sync: int = 20,
    num_workers_per_link: int = 2,
    num_buffers_per_channel: int = 2,
    deallocate_input: bool = True,
):
    """2D all-gather along axis 1 then axis 0 restoring ``[1, 1, R, C]``."""
    rm, cm = mesh_shape
    cur = tensor
    if cm > 1:
        cur = exact_all_gather_axis(
            cur, 1, mesh_shape, full_shape=full_shape, num_links=num_links,
            ag_ring=ag_ring, force_linear=force_linear, chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=num_buffers_per_channel,
            deallocate_input=(deallocate_input or cur is not tensor),
        )
    if rm > 1:
        cur = exact_all_gather_axis(
            cur, 0, mesh_shape, full_shape=full_shape, num_links=num_links,
            ag_ring=ag_ring, force_linear=force_linear, chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=num_buffers_per_channel,
            deallocate_input=(deallocate_input or cur is not tensor),
        )
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
        rs, mesh_shape, full_shape=full_shape, num_links=num_links,
        ag_ring=ag_ring, force_linear=force_linear, deallocate_input=True,
    )


def _exact_small_all_reduce_4d(
    tensor_4d,
    mesh_shape: Tuple[int, int],
    *,
    full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    ag_ring: bool = True,
    force_linear: bool = False,
):
    """Exact fp32 2D all-reduce for ``[1, B, M, N]`` tensors via ``all_gather(dim=0) + tree add``."""
    import ttnn

    rm, cm = mesh_shape
    full_r, full_c = full_shape
    cur = tensor_4d
    if rm > 1:
        topo0 = ttnn.Topology.Ring if (rm == full_r and ag_ring and not force_linear) else ttnn.Topology.Linear
        g0 = ttnn.all_gather(cur, dim=0, cluster_axis=0, topology=topo0, num_links=1)
        s0 = _tree_reduce_dim_tt(g0, dim=0, n_slices=rm)
        g0.deallocate()
        cur = s0
    if cm > 1:
        topo1 = ttnn.Topology.Ring if (cm == full_c and ag_ring and not force_linear) else ttnn.Topology.Linear
        g1 = ttnn.all_gather(cur, dim=0, cluster_axis=1, topology=topo1, num_links=1)
        s1 = _tree_reduce_dim_tt(g1, dim=0, n_slices=cm)
        g1.deallocate()
        if cur is not tensor_4d:
            cur.deallocate()
        cur = s1
    return cur


class DeviceCCLReducer:
    """Device-resident gradient all-reducer compatible with ``DeviceOptimizer.step(reduce_fn=...)``.

    Packs all tile-aligned 2D+ parameter gradients by column width ``N`` into a handful of flat
    ``[1, 1, R_pad, N]`` buffers (4 calls covering 100% of RFD4's 558 parameters when
    ``fold_small_into_width=True``) and reduces them via :func:`exact_all_reduce_2d`. Any remaining
    small/non-32-aligned groups fall back to 4D ``all_gather(dim=0) + exact fp32 tree add``.
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
        fold_small_into_width: bool = True,
        max_fold_elems: int = 4_194_304,
    ) -> None:
        self.mesh_shape = tuple(int(v) for v in mesh_shape)
        self.full_shape = tuple(int(v) for v in full_shape)
        self.num_links = int(num_links)
        self.ag_ring = bool(ag_ring)
        self.force_linear = bool(force_linear)
        self.scale = float(scale)
        self.replicated_names = frozenset(replicated_names)
        self.min_2d_group_elems = int(min_2d_group_elems)
        self.fold_small_into_width = bool(fold_small_into_width)
        self.max_fold_elems = int(max_fold_elems)
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
                fold_small_into_width=self.fold_small_into_width,
                max_fold_elems=self.max_fold_elems,
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

        # 1. Tile-aligned width groups (+ folded small groups): one exact_all_reduce_2d per column width.
        for wg in plan.width_groups:
            views = []
            owned_temps = []
            pieces = []
            dev = None
            for name, _shp, _off, rows in wg.entries:
                g = grads[name]
                if dev is None:
                    dev = g.device()
                v = ttnn.reshape(g, [1, 1, rows, wg.cols])
                views.append((v, g))
                pieces.append(v)

            for slot in wg.folded_smalls:
                s_views = []
                s_pieces = []
                for name, _shp, _boff, b in slot.entries:
                    g = grads[name]
                    if dev is None:
                        dev = g.device()
                    v4 = ttnn.reshape(g, [1, b, slot.m, slot.n])
                    s_views.append((v4, g))
                    s_pieces.append(v4)
                p4 = _concat_chunked(s_pieces, dim=1)
                for v4, orig in s_views:
                    _safe_dealloc(v4, keep=orig)
                padded = ttnn.pad(
                    p4,
                    [(0, 0), (0, 0), (0, slot.m_pad - slot.m), (0, wg.cols - slot.n)],
                    0.0,
                )
                _safe_dealloc(p4, keep=padded)
                flat_slot = ttnn.reshape(padded, [1, 1, slot.padded_rows, wg.cols])
                owned_temps.append((flat_slot, padded))
                pieces.append(flat_slot)

            pad_r = wg.padded_rows - wg.total_rows
            if pad_r > 0:
                pieces.append(self._zero_pad(dev, pad_r, wg.cols))
            packed = _concat_chunked(pieces, dim=2)
            for v, orig in views:
                _safe_dealloc(v, keep=orig)
            for flat_slot, padded in owned_temps:
                _safe_dealloc(flat_slot, keep=padded)
                _safe_dealloc(padded)

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

            single_direct = (len(wg.entries) == 1 and not wg.folded_smalls and pad_r == 0)
            for name, orig_shp, off, rows in wg.entries:
                if single_direct:
                    sl = reduced
                    reduced = None
                else:
                    sl = ttnn.slice(reduced, [0, 0, off, 0], [1, 1, off + rows, wg.cols])
                res = ttnn.reshape(sl, list(orig_shp))
                _safe_dealloc(sl, keep=res)
                out[name] = res

            for slot in wg.folded_smalls:
                sl_blk = ttnn.slice(
                    reduced,
                    [0, 0, slot.row_off, 0],
                    [1, 1, slot.row_off + slot.padded_rows, wg.cols],
                )
                v4 = ttnn.reshape(sl_blk, [1, slot.total_batch, slot.m_pad, wg.cols])
                for name, orig_shp, b_off, b in slot.entries:
                    sl = ttnn.slice(v4, [0, b_off, 0, 0], [1, b_off + b, slot.m, slot.n])
                    res = ttnn.reshape(sl, list(orig_shp))
                    _safe_dealloc(sl, keep=res)
                    out[name] = res
                _safe_dealloc(v4, keep=sl_blk)
                _safe_dealloc(sl_blk)

            if reduced is not None and reduced.is_allocated():
                ttnn.deallocate(reduced)

        # 2. Fallback small groups (only when fold_small_into_width=False or no matching width group).
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


# ---------------------------------------------------------------------------
# Axis-0 SPMD K-Sharded Muon ZeRO-1 Sandwich (RS_2D -> AG_ax1 -> 1/rm NS -> AG_ax0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SPMDMuonBucketSlot:
    """One exact-shape 2D Muon bucket sharded evenly along mesh axis 0 (``rm`` rows)."""

    shape: Tuple[int, int]   # (m, cols)
    names: Tuple[str, ...]   # K real parameter names in deterministic order
    k_shard: int             # ceil(K / rm) matrices owned per row r in 0..rm-1
    k_pad: int               # rm * k_shard
    slot_off: int            # row offset inside one row's [1, 1, row_block_h, cols] block

    @property
    def m(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def k(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class SPMDMuonWidthGroup:
    """All axis-0 sharded Muon buckets sharing column width ``cols``."""

    cols: int
    row_block_raw: int       # sum(b.k_shard * b.m)
    row_block_h: int         # rounded up to multiple of TILE * cm (exact 0 tail pad on RFD4 8x4)
    full_rows: int           # rm * row_block_h
    buckets: Tuple[SPMDMuonBucketSlot, ...]


@dataclass(frozen=True)
class SPMDMuonPlan:
    """Zero-extra-CCL Axis-0 SPMD K-sharded Muon plan for a 2D mesh ``(rm, cm)``."""

    mesh_shape: Tuple[int, int]
    width_groups: Tuple[SPMDMuonWidthGroup, ...]
    sharded_names: FrozenSet[str]


def plan_spmd_muon_shards(
    shapes: Mapping[str, Tuple[int, ...]],
    muon_names: Collection[str],
    mesh_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
    *,
    min_k_ratio: float = 0.5,
) -> SPMDMuonPlan:
    """Plan Axis-0 K-sharded Muon buckets for zero-extra-CCL SPMD ZeRO-1.

    On an ``8x4`` Galaxy (``rm=8, cm=4``), all 10 large tile-aligned RFD4 Muon buckets (265 of 282
    Muon matrices, 176.2M params, 99.66% of Muon FLOPs) shard across axis 0 with only 11.1% total
    padding and exact 0 tail padding on every width group. Splitting ``exact_all_reduce_2d`` into
    ``RS_2D -> AG_axis1`` (before 1/rm Muon NS) and ``AG_axis0`` (after 1/rm Muon NS) cuts Muon
    compute by ~7.5x with zero extra fabric collectives.
    """
    rm, cm = int(mesh_shape[0]), int(mesh_shape[1])
    if rm <= 1:
        return SPMDMuonPlan(mesh_shape=(rm, cm), width_groups=(), sharded_names=frozenset())

    muon_set = set(muon_names)
    by_shape: Dict[Tuple[int, int], List[str]] = {}
    for nm in sorted(shapes):
        if nm not in muon_set:
            continue
        shp = tuple(int(d) for d in shapes[nm])
        if len(shp) == 2 and shp[0] % TILE == 0 and shp[1] % TILE == 0:
            by_shape.setdefault((shp[0], shp[1]), []).append(nm)

    min_k = max(2, math.ceil(rm * float(min_k_ratio)))
    by_cols: Dict[int, List[Tuple[Tuple[int, int], Tuple[str, ...]]]] = {}
    sharded: List[str] = []
    for shp in sorted(by_shape):
        nms = tuple(by_shape[shp])
        if len(nms) < min_k:
            continue
        by_cols.setdefault(shp[1], []).append((shp, nms))
        sharded.extend(nms)

    block_quantum = TILE * cm
    width_groups: List[SPMDMuonWidthGroup] = []
    for cols in sorted(by_cols, key=lambda c: -sum(len(n) * s[0] * c for s, n in by_cols[c])):
        items = sorted(by_cols[cols], key=lambda x: (-x[0][0], x[0]))
        slots: List[SPMDMuonBucketSlot] = []
        off = 0
        for (m, c), nms in items:
            k_shard = math.ceil(len(nms) / rm)
            slots.append(
                SPMDMuonBucketSlot(
                    shape=(m, c),
                    names=nms,
                    k_shard=k_shard,
                    k_pad=rm * k_shard,
                    slot_off=off,
                )
            )
            off += k_shard * m
        row_block_h = math.ceil(off / block_quantum) * block_quantum
        width_groups.append(
            SPMDMuonWidthGroup(
                cols=cols,
                row_block_raw=off,
                row_block_h=row_block_h,
                full_rows=rm * row_block_h,
                buckets=tuple(slots),
            )
        )

    return SPMDMuonPlan(
        mesh_shape=(rm, cm),
        width_groups=tuple(width_groups),
        sharded_names=frozenset(sharded),
    )


class SPMDMuonSharder:
    """Zero-extra-CCL Axis-0 SPMD Muon ZeRO-1 collective engine.

    Lifecycle per step on a ``MeshDevice(rm, cm)``:
      1. :meth:`reduce_scatter_grads`: packs sharded Muon grads interleaved by axis-0 row, runs
         ``RS_2D -> AG_axis1`` (one call per width group), and slices each bucket's ``[k_shard, m, cols]``
         exact 32-rank summed gradient shard on chip ``(r, c)``.
      2. Optimizer runs ``1/rm`` Muon momentum + Newton-Schulz + master/EMA update on each bucket's
         ``[k_shard, m, cols]`` shard (global clip norm sums ``rm`` scalars read from row reps).
      3. :meth:`all_gather_masters`: packs each chip's updated ``[k_shard, m, cols]`` master shards,
         runs ``AG_axis0`` (one call per width group), and unpacks bit-identical full masters for
         all ``K`` matrices on every chip.
    """

    def __init__(
        self,
        plan: SPMDMuonPlan,
        *,
        full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
        num_links: int = 2,
        ag_ring: bool = True,
        force_linear: bool = False,
        scale: float = 1.0,
    ) -> None:
        self.plan = plan
        self.mesh_shape = plan.mesh_shape
        self.full_shape = tuple(int(v) for v in full_shape)
        self.num_links = int(num_links)
        self.ag_ring = bool(ag_ring)
        self.force_linear = bool(force_linear)
        self.scale = float(scale)
        self._pad_zeros: Dict[Tuple[int, int, int], object] = {}

    def _zero_2d(self, device, rows: int, cols: int):
        import ttnn

        key = (id(device), rows, cols)
        z = self._pad_zeros.get(key)
        if z is None or not z.is_allocated():
            z = ttnn.zeros([1, 1, rows, cols], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            self._pad_zeros[key] = z
        return z

    def deallocate_caches(self) -> None:
        import ttnn

        for z in self._pad_zeros.values():
            if z is not None and z.is_allocated():
                ttnn.deallocate(z)
        self._pad_zeros.clear()

    def shard_initial_masters(
        self,
        initial_weights: Mapping[str, object],
        device,
    ) -> Dict[Tuple[int, int], object]:
        """Upload each sharded Muon bucket's ``[k_shard, m, cols]`` initial master per axis-0 row."""
        import numpy as np
        import torch
        import ttnn

        rm, cm = self.mesh_shape
        mapper = ttnn.ShardTensor2dMesh(device, (rm, cm), dims=(0, None))
        out: Dict[Tuple[int, int], object] = {}
        for wg in self.plan.width_groups:
            for slot in wg.buckets:
                buf = torch.zeros((slot.k_pad, slot.m, slot.cols), dtype=torch.float32)
                for i, nm in enumerate(slot.names):
                    v = initial_weights[nm]
                    if isinstance(v, np.ndarray):
                        t = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
                    elif torch.is_tensor(v):
                        t = v.detach().to(torch.float32).contiguous()
                    else:
                        t = ttnn.to_torch(ttnn.get_device_tensors(v)[0]).detach().to(torch.float32).contiguous()
                    buf[i] = t.reshape(slot.m, slot.cols)
                sharded_4d = ttnn.from_torch(
                    buf.reshape(rm, slot.k_shard, slot.m, slot.cols).contiguous(),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=mapper,
                )
                sharded_3d = ttnn.reshape(sharded_4d, [slot.k_shard, slot.m, slot.cols])
                _safe_dealloc(sharded_4d, keep=sharded_3d)
                out[slot.shape] = sharded_3d
        return out

    def active_masks_for_step(
        self,
        grads: Mapping[str, object],
        disabled: Collection[str],
        device,
    ) -> Dict[Tuple[int, int], Optional[Tuple[object, object]]]:
        """Return ``{shape: (m_act, m_inact) | None}`` per bucket (``None`` when all slots active & unpadded)."""
        import torch
        import ttnn

        rm, cm = self.mesh_shape
        dis = set(disabled)
        mapper = ttnn.ShardTensor2dMesh(device, (rm, cm), dims=(0, None))
        out: Dict[Tuple[int, int], Optional[Tuple[object, object]]] = {}
        for wg in self.plan.width_groups:
            for slot in wg.buckets:
                flags = [
                    (i < slot.k) and (slot.names[i] not in dis) and (grads.get(slot.names[i]) is not None)
                    for i in range(slot.k_pad)
                ]
                if all(flags):
                    out[slot.shape] = None
                    continue
                act_t = torch.tensor(
                    [1.0 if f else 0.0 for f in flags], dtype=torch.float32
                ).reshape(rm, slot.k_shard, 1, 1)
                inact_t = 1.0 - act_t
                ma4 = ttnn.from_torch(
                    act_t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    device=device, mesh_mapper=mapper,
                )
                mi4 = ttnn.from_torch(
                    inact_t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    device=device, mesh_mapper=mapper,
                )
                ma = ttnn.reshape(ma4, [slot.k_shard, 1, 1])
                mi = ttnn.reshape(mi4, [slot.k_shard, 1, 1])
                _safe_dealloc(ma4, keep=ma)
                _safe_dealloc(mi4, keep=mi)
                out[slot.shape] = (ma, mi)
        return out

    def reduce_scatter_grads(
        self,
        grads: Mapping[str, object],
        device,
        *,
        disabled: Collection[str] = (),
    ) -> Dict[Tuple[int, int], object]:
        """Run ``RS_2D -> AG_axis1`` per width group; return ``{shape: [k_shard, m, cols]}`` per row."""
        import ttnn

        rm, _cm = self.mesh_shape
        dis = set(disabled)
        out: Dict[Tuple[int, int], object] = {}

        for wg in self.plan.width_groups:
            views = []
            pieces = []
            tail_r = wg.row_block_h - wg.row_block_raw
            for r in range(rm):
                for slot in wg.buckets:
                    for j in range(slot.k_shard):
                        idx = r * slot.k_shard + j
                        g = None
                        if idx < slot.k:
                            nm = slot.names[idx]
                            if nm not in dis:
                                g = grads.get(nm)
                        if g is not None:
                            v = ttnn.reshape(g, [1, 1, slot.m, wg.cols])
                            views.append((v, g))
                            pieces.append(v)
                        else:
                            pieces.append(self._zero_2d(device, slot.m, wg.cols))
                if tail_r > 0:
                    pieces.append(self._zero_2d(device, tail_r, wg.cols))

            packed = _concat_chunked(pieces, dim=2)
            for v, orig in views:
                _safe_dealloc(v, keep=orig)

            rs = exact_reduce_scatter_2d(
                packed,
                self.mesh_shape,
                full_shape=self.full_shape,
                num_links=self.num_links,
                force_linear=self.force_linear,
            )
            ttnn.deallocate(packed)

            row_block = exact_all_gather_axis(
                rs,
                1,
                self.mesh_shape,
                full_shape=self.full_shape,
                num_links=self.num_links,
                ag_ring=self.ag_ring,
                force_linear=self.force_linear,
                deallocate_input=True,
            )
            if self.scale != 1.0:
                scaled = ttnn.multiply(row_block, self.scale)
                ttnn.deallocate(row_block)
                row_block = scaled

            for slot in wg.buckets:
                h = slot.k_shard * slot.m
                if len(wg.buckets) == 1 and tail_r == 0:
                    sl = row_block
                    row_block = None
                else:
                    sl = ttnn.slice(row_block, [0, 0, slot.slot_off, 0], [1, 1, slot.slot_off + h, wg.cols])
                sh = ttnn.reshape(sl, [slot.k_shard, slot.m, wg.cols])
                _safe_dealloc(sl, keep=sh)
                out[slot.shape] = sh

            if row_block is not None and row_block.is_allocated():
                ttnn.deallocate(row_block)

        return out

    def all_gather_width_groups(
        self,
        master_shards: Mapping[Tuple[int, int], object],
        device,
    ) -> list:
        """Run ``AG_axis0`` per width group and return packed ``full`` tensors (one per width group)."""
        import ttnn

        full_wgs = []
        for wg in self.plan.width_groups:
            views = []
            pieces = []
            tail_r = wg.row_block_h - wg.row_block_raw
            for slot in wg.buckets:
                sh = master_shards[slot.shape]
                v = ttnn.reshape(sh, [1, 1, slot.k_shard * slot.m, wg.cols])
                views.append((v, sh))
                pieces.append(v)
            if tail_r > 0:
                pieces.append(self._zero_2d(device, tail_r, wg.cols))

            local_blk = _concat_chunked(pieces, dim=2)
            for v, orig in views:
                _safe_dealloc(v, keep=orig)

            full = exact_all_gather_axis(
                local_blk,
                0,
                self.mesh_shape,
                full_shape=self.full_shape,
                num_links=self.num_links,
                ag_ring=self.ag_ring,
                force_linear=self.force_linear,
                deallocate_input=True,
            )
            full_wgs.append(full)
        return full_wgs

    def unpack_width_groups_in_place(
        self,
        full_wgs: Sequence[object],
        masters: Mapping[str, object],
    ) -> None:
        """Unpack gathered width-group tensors directly into existing ``masters[nm]`` slots in-place."""
        import ttnn

        for wg, full in zip(self.plan.width_groups, full_wgs):
            for slot in wg.buckets:
                for i, nm in enumerate(slot.names):
                    r = i // slot.k_shard
                    j = i % slot.k_shard
                    off = r * wg.row_block_h + slot.slot_off + j * slot.m
                    sl = ttnn.slice(full, [0, 0, off, 0], [1, 1, off + slot.m, wg.cols])
                    m2d = ttnn.reshape(sl, [slot.m, wg.cols])
                    _safe_dealloc(sl, keep=m2d)
                    old_m = masters[nm]
                    if m2d is not old_m:
                        ttnn.copy(m2d, old_m)
                        _safe_dealloc(m2d, keep=old_m)

    def all_gather_masters(
        self,
        master_shards: Mapping[Tuple[int, int], object],
        device,
    ) -> Dict[str, object]:
        """Run ``AG_axis0`` per width group and unpack bit-identical full ``{name: [m, cols]}`` masters."""
        import ttnn

        full_wgs = self.all_gather_width_groups(master_shards, device)
        out: Dict[str, object] = {}
        for wg, full in zip(self.plan.width_groups, full_wgs):
            for slot in wg.buckets:
                for i, nm in enumerate(slot.names):
                    r = i // slot.k_shard
                    j = i % slot.k_shard
                    off = r * wg.row_block_h + slot.slot_off + j * slot.m
                    sl = ttnn.slice(full, [0, 0, off, 0], [1, 1, off + slot.m, wg.cols])
                    m2d = ttnn.reshape(sl, [slot.m, wg.cols])
                    _safe_dealloc(sl, keep=m2d)
                    out[nm] = m2d
            ttnn.deallocate(full)
        return out

    def gather_host_shards(
        self,
        sharded_tensors: Mapping[Tuple[int, int], object],
    ) -> Dict[str, "np.ndarray"]:
        """Gather axis-0 sharded ``[k_shard, m, cols]`` tensors to host numpy reading only ``rm`` chips."""
        import numpy as np
        import ttnn

        rm, cm = self.mesh_shape
        out: Dict[str, np.ndarray] = {}
        for wg in self.plan.width_groups:
            for slot in wg.buckets:
                tt = sharded_tensors.get(slot.shape)
                if tt is None:
                    continue
                dev_tensors = ttnn.get_device_tensors(tt)
                row_arrays = [
                    ttnn.to_torch(dev_tensors[r * cm]).detach().float().cpu().numpy()
                    for r in range(rm)
                ]
                for i, nm in enumerate(slot.names):
                    r = i // slot.k_shard
                    j = i % slot.k_shard
                    out[nm] = np.ascontiguousarray(row_arrays[r][j, :slot.m, :slot.cols], dtype=np.float32)
        return out
