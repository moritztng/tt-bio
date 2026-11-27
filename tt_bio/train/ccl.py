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
from typing import Dict, Mapping, Sequence, Tuple

FABRIC_FP32_MAX_PAYLOAD_BYTES = 8192
TILE = 32
DEFAULT_FULL_MESH = (8, 4)

__all__ = [
    "FABRIC_FP32_MAX_PAYLOAD_BYTES",
    "DEFAULT_FULL_MESH",
    "configure_fp32_ring_fabric",
    "flat_2d_shape",
    "fp32_reduction_order_floor",
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


class DeviceCCLReducer:
    """Device-resident gradient all-reducer compatible with ``DeviceOptimizer.step(reduce_fn=...)``.

    Calls :func:`exact_all_reduce_2d` directly on flat ``[1, 1, R, C]`` fp32 buffers (zero host
    round-trip). For a dict of per-parameter device gradients, reduces each parameter on device
    when its height is divisible by ``32 * rm * cm``, or sums via ``all_gather(dim=0) + add`` for
    smaller tensors (1D norms/biases).
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
    ) -> None:
        self.mesh_shape = tuple(int(v) for v in mesh_shape)
        self.full_shape = tuple(int(v) for v in full_shape)
        self.num_links = int(num_links)
        self.ag_ring = bool(ag_ring)
        self.force_linear = bool(force_linear)
        self.scale = float(scale)

    def reduce_tensor(self, tensor):
        import ttnn

        rm, cm = self.mesh_shape
        if rm * cm <= 1:
            return tensor
        if len(tensor.shape) == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 1 and tensor.shape[2] % (TILE * rm * cm) == 0:
            out = exact_all_reduce_2d(
                tensor,
                self.mesh_shape,
                full_shape=self.full_shape,
                num_links=self.num_links,
                ag_ring=self.ag_ring,
                force_linear=self.force_linear,
            )
        else:
            # Exact small-tensor fallback: all_gather along axis 0 then axis 1 + fp32 add
            cur = tensor
            if rm > 1:
                g0 = ttnn.all_gather(cur, dim=0, cluster_axis=0, topology=ttnn.Topology.Linear, num_links=1)
                s0 = ttnn.slice(g0, [0, 0, 0, 0], [1, *cur.shape[1:]])
                for i in range(1, rm):
                    sl = ttnn.slice(g0, [i, 0, 0, 0], [i + 1, *cur.shape[1:]])
                    nxt = ttnn.add(s0, sl)
                    s0.deallocate()
                    sl.deallocate()
                    s0 = nxt
                g0.deallocate()
                cur = s0
            if cm > 1:
                g1 = ttnn.all_gather(cur, dim=0, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=1)
                s1 = ttnn.slice(g1, [0, 0, 0, 0], [1, *cur.shape[1:]])
                for i in range(1, cm):
                    sl = ttnn.slice(g1, [i, 0, 0, 0], [i + 1, *cur.shape[1:]])
                    nxt = ttnn.add(s1, sl)
                    s1.deallocate()
                    sl.deallocate()
                    s1 = nxt
                g1.deallocate()
                if cur is not tensor:
                    cur.deallocate()
                cur = s1
            out = cur
        if self.scale != 1.0:
            scaled = ttnn.multiply(out, self.scale)
            out.deallocate()
            out = scaled
        return out

    def __call__(self, grads: Mapping[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for name, g in grads.items():
            out[name] = None if g is None else self.reduce_tensor(g)
        return out
