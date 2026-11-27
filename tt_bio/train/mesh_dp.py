"""Single-process SPMD MeshDevice data-parallel helpers for device-resident training (D89).

On a Blackhole Galaxy (8x4), Ethernet fabric CCL (``tt_bio.train.ccl``) runs inside a single
process holding a ``MeshDevice`` (or submesh). All chips execute identical SPMD programs:
weights, optimizer masters, and constants are replicated identically across the mesh (plain
``ttnn.from_torch(..., device=mesh)``), while per-rank batch inputs and tiny per-rank outputs
cross the host via ``ShardTensorToMesh(mesh, dim=0)`` and ``ConcatMeshToTensor(mesh, dim=0)``.

Zero ``ttnn`` import at module load (preserves ``tests/test_training_opt_in.py``).
"""
from __future__ import annotations

import contextlib
import hashlib
from typing import Any, List, Mapping, Sequence, Tuple

__all__ = [
    "mesh_world",
    "mesh_shape",
    "is_multi_chip",
    "use_mesh_device",
    "open_galaxy_submesh",
    "put_sharded",
    "put_sharded_2d",
    "read_sharded",
    "read_rep0",
    "read_axis0_sum",
    "read_axis0_vectors",
    "assert_mesh_consensus",
    "stack_batches",
    "unstack_leading",
    "MeshBridge",
]


def mesh_world(device) -> int:
    """Return number of physical chips in ``device`` (1 for single-chip or None)."""
    if device is None:
        return 1
    fn = getattr(device, "get_num_devices", None)
    return int(fn()) if callable(fn) else 1


def mesh_shape(device) -> Tuple[int, int]:
    """Return ``(rm, cm)`` 2D mesh shape for ``device`` (``(1, 1)`` for single chip)."""
    if device is None:
        return (1, 1)
    shp = getattr(device, "shape", None)
    if shp is not None:
        vals = tuple(int(d) for d in shp)
        if len(vals) == 2:
            return vals
        if len(vals) == 1:
            return (1, vals[0])
    w = mesh_world(device)
    return (1, w)


def is_multi_chip(device) -> bool:
    """True when ``device`` is a multi-chip MeshDevice."""
    return mesh_world(device) > 1


@contextlib.contextmanager
def use_mesh_device(mesh):
    """Temporarily bind ``tt_bio.tenstorrent._device`` to ``mesh`` without closing it on exit.

    Every internal ``tenstorrent.get_device()`` call inside ``Trunk``, ``DeviceFolds``,
    ``DeviceHeads``, and custom ``generic_op`` wrappers receives ``mesh`` automatically.
    Also configures ``tenstorrent.CORE_GRID_MAIN`` / ``COMPUTE_GRID_MAIN`` and ``_dest_guard``
    from ``mesh`` so Blackhole Galaxy runs on its real 12x10 (120c) or 11x10 grid instead of
    the import-time fallback.
    """
    from .. import tenstorrent as T

    prev = T._device
    T._device = mesh
    if hasattr(mesh, "compute_with_storage_grid_size"):
        T._configure_active_compute_grid(mesh)
        try:
            T._dest_guard.install(mesh)
        except Exception:
            pass
    try:
        yield mesh
    finally:
        T._device = prev


@contextlib.contextmanager
def open_galaxy_submesh(
    target_shape: Tuple[int, int],
    *,
    full_shape: Tuple[int, int] = (8, 4),
    max_payload_bytes: int = 8192,
    trace_region_size: int = 0,
    eth_dispatch: bool | None = None,
):
    """Open a clean single-submesh Blackhole Galaxy ``MeshDevice`` and bind ``tenstorrent._device``.

    Enforces ``tt-metal`` ``mesh_device.cpp:861``: always opens ``full_shape`` with
    ``FABRIC_1D_RING`` (``max_packet_payload_size_bytes >= 8192``), creates at most ONE submesh
    without ever running ops on ``full``, and closes ``sub`` then ``full`` on exit.
    When ``eth_dispatch`` is True (or ``None`` on a Blackhole host supporting ETH dispatch),
    opens with ``DispatchCoreType.ETH`` so all chips use the full 12x10 (120-core) compute grid
    while active Ethernet links serve the 2D Ring fabric.
    """
    import ttnn
    from .ccl import configure_fp32_ring_fabric

    rm, cm = int(target_shape[0]), int(target_shape[1])
    fr, fc = int(full_shape[0]), int(full_shape[1])
    if rm <= 0 or cm <= 0 or rm > fr or cm > fc:
        raise ValueError(f"invalid target_shape {target_shape} for full_shape {full_shape}")

    configure_fp32_ring_fabric(max_payload_bytes)
    open_kw: dict[str, Any] = {}
    if trace_region_size > 0:
        open_kw["trace_region_size"] = int(trace_region_size)
    use_eth = eth_dispatch
    if use_eth is None:
        from .. import tenstorrent as T
        use_eth = (not T.is_wormhole()) and T.bh_eth_dispatch()
    if use_eth:
        open_kw["dispatch_core_config"] = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)
    full = ttnn.open_mesh_device(ttnn.MeshShape(fr, fc), **open_kw)
    sub = None
    try:
        active = full if (rm == fr and cm == fc) else full.create_submesh(ttnn.MeshShape(rm, cm))
        if active is not full:
            sub = active
        with use_mesh_device(active):
            yield active
    finally:
        if sub is not None:
            try:
                ttnn.synchronize_device(sub)
            except Exception:
                pass
            ttnn.close_mesh_device(sub)
        else:
            try:
                ttnn.synchronize_device(full)
            except Exception:
                pass
        ttnn.close_mesh_device(full)


def put_sharded(tensor, device, *, dtype=None, layout=None, memory_config=None):
    """Upload a host torch tensor sharded along ``dim=0`` across a multi-chip MeshDevice.

    On a single-chip device (``world == 1``), falls back to plain ``ttnn.from_torch``.
    ``tensor.shape[0]`` must be divisible by ``mesh_world(device)``.
    """
    import ttnn

    w = mesh_world(device)
    if w > 1 and int(tensor.shape[0]) % w != 0:
        raise ValueError(f"put_sharded: dim 0 ({tensor.shape[0]}) not divisible by mesh world {w}")
    kw: dict[str, Any] = {
        "device": device,
        "layout": ttnn.TILE_LAYOUT if layout is None else layout,
    }
    if dtype is not None:
        kw["dtype"] = dtype
    if memory_config is not None:
        kw["memory_config"] = memory_config
    if w > 1:
        kw["mesh_mapper"] = ttnn.ShardTensorToMesh(device, dim=0)
    return ttnn.from_torch(tensor.contiguous(), **kw)


def put_sharded_2d(
    tensor,
    device,
    shape_2d: Tuple[int, int] | None = None,
    dims: Tuple[int | None, int | None] = (0, None),
    *,
    dtype=None,
    layout=None,
    memory_config=None,
):
    """Upload a host torch tensor with 2D mesh placement (e.g. ``dims=(0, None)`` for axis-0 sharding)."""
    import ttnn

    w = mesh_world(device)
    kw: dict[str, Any] = {
        "device": device,
        "layout": ttnn.TILE_LAYOUT if layout is None else layout,
    }
    if dtype is not None:
        kw["dtype"] = dtype
    if memory_config is not None:
        kw["memory_config"] = memory_config
    if w > 1:
        shp = mesh_shape(device) if shape_2d is None else tuple(int(v) for v in shape_2d)
        kw["mesh_mapper"] = ttnn.ShardTensor2dMesh(device, shp, dims=dims)
    return ttnn.from_torch(tensor.contiguous(), **kw)


def read_sharded(tt_tensor, device=None):
    """Read a per-rank sharded device tensor to host torch concatenated along ``dim=0``."""
    import ttnn

    dev = tt_tensor.device() if device is None else device
    if mesh_world(dev) > 1:
        return ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))
    return ttnn.to_torch(tt_tensor)


def read_rep0(tt_tensor, device=None):
    """Read chip 0's copy of a replicated MeshDevice tensor (4 B for scalars, zero peer PCIe)."""
    import ttnn

    dev = tt_tensor.device() if device is None else device
    if mesh_world(dev) > 1:
        shards = ttnn.get_device_tensors(tt_tensor)
        return ttnn.to_torch(shards[0])
    return ttnn.to_torch(tt_tensor)


def read_axis0_sum(tt_tensor, shape_2d: Tuple[int, int] | None = None, device=None) -> float:
    """Sum scalar[0] across the ``rm`` row representatives ``r * cm`` in exact float64.

    Used for Axis-0 SPMD ZeRO-1 global clip norm: each row ``r`` owns a disjoint shard replicated
    across its ``cm`` columns, so reading only chips ``0, cm, 2*cm, ...`` computes the exact
    global sum of squares in ``4 * rm`` bytes (32 bytes on an 8x4 Galaxy).
    """
    import ttnn

    dev = tt_tensor.device() if device is None else device
    if mesh_world(dev) <= 1:
        return float(ttnn.to_torch(tt_tensor).float().reshape(-1)[0].item())
    rm, cm = mesh_shape(dev) if shape_2d is None else (int(shape_2d[0]), int(shape_2d[1]))
    shards = ttnn.get_device_tensors(tt_tensor)
    total = 0.0
    for r in range(rm):
        total += float(ttnn.to_torch(shards[r * cm]).float().reshape(-1)[0].item())
    return total


def read_axis0_vectors(tt_tensor, shape_2d: Tuple[int, int] | None = None, device=None) -> list:
    """Read flat float32 numpy vectors from the ``rm`` row representatives ``r * cm``.

    Used by Axis-0 SPMD ZeRO-1 to collect exact per-parameter squared gradient norms (and
    ``dead_grads`` flags) for all sharded Muon parameters in one tiny readback (~1.3 KB on 8x4)
    without running a device binary-tree scalar reduction.
    """
    import ttnn

    dev = tt_tensor.device() if device is None else device
    if mesh_world(dev) <= 1:
        return [ttnn.to_torch(tt_tensor).float().numpy().reshape(-1)]
    rm, cm = mesh_shape(dev) if shape_2d is None else (int(shape_2d[0]), int(shape_2d[1]))
    shards = ttnn.get_device_tensors(tt_tensor)
    return [ttnn.to_torch(shards[r * cm]).float().numpy().reshape(-1) for r in range(rm)]


def assert_mesh_consensus(tt_tensor, device=None, *, name: str = "") -> str:
    """Assert all ranks of a replicated MeshDevice tensor are bit-identical; return SHA-256[:16]."""
    import torch
    import ttnn

    dev = tt_tensor.device() if device is None else device
    if mesh_world(dev) <= 1:
        arr = ttnn.to_torch(tt_tensor).detach().cpu().contiguous()
        return hashlib.sha256(arr.numpy().tobytes()).hexdigest()[:16]

    shards = ttnn.get_device_tensors(tt_tensor)
    ref = ttnn.to_torch(shards[0]).detach().cpu().contiguous()
    for r in range(1, len(shards)):
        cur = ttnn.to_torch(shards[r]).detach().cpu().contiguous()
        if not torch.equal(ref, cur):
            diff = float((ref.float() - cur.float()).abs().max().item())
            label = f" ({name})" if name else ""
            raise AssertionError(
                f"MeshDevice consensus failure{label}: rank {r} differs from rank 0 (max_abs={diff:.3e})"
            )
    return hashlib.sha256(ref.numpy().tobytes()).hexdigest()[:16]


def stack_batches(batches: Sequence[Mapping[str, Any]]) -> dict:
    """Concatenate ``world`` per-rank batch dicts along ``dim=0`` (recursively for nested dicts).

    Each rank's batch has shape ``[B, ...]`` (e.g. ``B=K`` for candidates or ``B=1`` for winners);
    the stacked batch has shape ``[world * B, ...]`` so batched PyTorch glue runs in one call and
    ``ShardTensorToMesh(mesh, dim=0)`` delivers rank ``r``'s exact ``[B, ...]`` slice to chip ``r``.
    """
    import torch

    if not batches:
        raise ValueError("stack_batches requires at least one batch")
    keys = list(batches[0].keys())
    out: dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batches]
        v0 = vals[0]
        if isinstance(v0, Mapping):
            out[k] = stack_batches(vals)
        elif torch.is_tensor(v0):
            if v0.dim() == 0:
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = torch.cat(vals, dim=0)
        else:
            out[k] = v0
    return out


def unstack_leading(tensor_or_array, world: int) -> List[Any]:
    """Split a ``[world * B, ...]`` tensor/ndarray into ``world`` equal ``[B, ...]`` slices."""
    if world <= 0:
        raise ValueError(f"world must be > 0, got {world}")
    n0 = int(tensor_or_array.shape[0])
    if n0 % world != 0:
        raise ValueError(f"leading dim {n0} not divisible by world {world}")
    step = n0 // world
    return [tensor_or_array[r * step:(r + 1) * step] for r in range(world)]


class MeshBridge:
    """MeshDevice-aware host torch <-> device tape bridge for SPMD data-parallel steps.

    * ``up(h)`` uploads a ``[world * B, ...]`` host activation sharded along ``dim=0`` so chip
      ``r`` holds ``[B, ...]``. On backward, ``backward_ups()`` gathers ``[world * B, ...]`` grads
      via ``ConcatMeshToTensor(mesh, dim=0)`` so one batched ``torch.autograd.backward`` accumulates
      the exact rank-summed gradient into every host-glue parameter.
    * ``up_replicated(p)`` uploads a replicated model parameter ``[*p.shape]`` (such as
      ``pair_repr_builder``'s ``linear_out.weight`` / ``ln_out.weight``) to every chip. On backward,
      each chip's per-rank device gradient is gathered and summed across all ``world`` ranks before
      passing into ``torch.autograd.backward``.
    """

    def __init__(self, device, dtype=None):
        import ttnn

        self.device = device
        self.dtype = ttnn.bfloat16 if dtype is None else dtype
        self.world = mesh_world(device)
        self.ups: List[Tuple[Any, Any, bool]] = []  # (host_tensor, device_leaf, is_replicated)
        self.downs: List[Tuple[Any, Any]] = []

    def up(self, h, dtype=None):
        import torch
        import ttnn
        from .. import autograd as ag

        v = put_sharded(
            h.detach().to(torch.float32),
            self.device,
            dtype=dtype or self.dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        leaf = ag.Tensor(v, requires_grad=h.requires_grad)
        if h.requires_grad:
            self.ups.append((h, leaf, False))
        return leaf

    def up_replicated(self, p, dtype=None):
        import torch
        import ttnn
        from .. import autograd as ag

        v = ttnn.from_torch(
            p.detach().to(torch.float32).contiguous(),
            dtype=dtype or self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        leaf = ag.Tensor(v, requires_grad=p.requires_grad)
        if p.requires_grad:
            self.ups.append((p, leaf, True))
        return leaf

    def backward_ups(self) -> None:
        """Consume all recorded ``up`` / ``up_replicated`` device gradients and backprop in one call."""
        import torch
        import ttnn

        host_leaves, host_grads = [], []
        for h_t, d_t, is_rep in self.ups:
            if d_t.grad is None:
                continue
            if is_rep and self.world > 1:
                shards = ttnn.get_device_tensors(d_t.grad)
                g_acc = ttnn.to_torch(shards[0]).float().reshape(h_t.shape)
                for r in range(1, self.world):
                    g_acc = g_acc + ttnn.to_torch(shards[r]).float().reshape(h_t.shape)
                g_h = g_acc.to(h_t.dtype)
            else:
                g_h = read_sharded(d_t.grad, self.device).float().reshape(h_t.shape).to(h_t.dtype)
            ttnn.deallocate(d_t.grad)
            d_t.grad = None
            host_leaves.append(h_t)
            host_grads.append(g_h)
        if host_leaves:
            torch.autograd.backward(host_leaves, host_grads)
        self.ups.clear()
        self.downs.clear()
