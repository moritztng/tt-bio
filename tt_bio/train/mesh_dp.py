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
    "is_multi_chip",
    "use_mesh_device",
    "put_sharded",
    "read_sharded",
    "read_rep0",
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


def is_multi_chip(device) -> bool:
    """True when ``device`` is a multi-chip MeshDevice."""
    return mesh_world(device) > 1


@contextlib.contextmanager
def use_mesh_device(mesh):
    """Temporarily bind ``tt_bio.tenstorrent._device`` to ``mesh`` without closing it on exit.

    Every internal ``tenstorrent.get_device()`` call inside ``Trunk``, ``DeviceFolds``,
    ``DeviceHeads``, and custom ``generic_op`` wrappers receives ``mesh`` automatically.
    """
    from .. import tenstorrent as T

    prev = T._device
    T._device = mesh
    try:
        yield mesh
    finally:
        T._device = prev


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

    ``up(h)`` uploads a ``[world * B, ...]`` host torch tensor sharded along ``dim=0`` so chip
    ``r`` holds ``[B, ...]``. On backward, ``backward_ups()`` gathers ``[world * B, ...]`` grads
    via ``ConcatMeshToTensor(mesh, dim=0)`` and runs one batched ``torch.autograd.backward``,
    accumulating the exact rank-summed gradient into every host-glue parameter.
    """

    def __init__(self, device, dtype=None):
        import ttnn

        self.device = device
        self.dtype = ttnn.bfloat16 if dtype is None else dtype
        self.world = mesh_world(device)
        self.ups: List[Tuple[Any, Any]] = []
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
            self.ups.append((h, leaf))
        return leaf

    def backward_ups(self) -> None:
        """Consume all recorded ``up`` device gradients and backprop into host glue in one call."""
        import torch
        import ttnn

        host_leaves, host_grads = [], []
        for h_t, d_t in self.ups:
            if d_t.grad is None:
                continue
            g_h = read_sharded(d_t.grad, self.device).float().reshape(h_t.shape).to(h_t.dtype)
            ttnn.deallocate(d_t.grad)
            d_t.grad = None
            host_leaves.append(h_t)
            host_grads.append(g_h)
        if host_leaves:
            torch.autograd.backward(host_leaves, host_grads)
        self.ups.clear()
        self.downs.clear()
