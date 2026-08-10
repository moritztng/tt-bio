"""The trimul channel move ``permute(x, (0, 3, 1, 2))`` as a hand-written Tensix kernel.

Shape is ``[1, N, N, C]`` bf16 TILE with ``C`` a multiple of 32. **The channel count is not fixed
at 32.** ``_trimul_chunk_size`` doubles the trunk's chunk width while the chunk still fits an L1
budget scaled by the compute grid, so 298 aa folds with ``C = 64`` on a 13x10 grid and with
``C = 32`` on an 11x10 one. A kernel hardcoded to 32 channels serves zero calls on the wider grid.

The three kernels under ``tt_bio/kernels/reblock_permute/`` are run through ``ttnn.generic_op``,
which JIT-compiles a kernel named by a ``KernelDescriptor`` against the shipped ttnn wheel. No
tt-metal source build, no nanobind registration, no dependency bump.

The move is a pure index reordering, so it is bit-exact against ``ttnn.permute`` by construction and
is measured with ``torch.equal`` rather than argued.

**Why the descriptor is cached.** ``generic_op`` takes the whole program description per call.
Building it in Python costs ~155 us at N=320 (100 cores x 3 kernels of per-core runtime args), which
is more than the 91 us of device time the op needs, so a rebuilt-per-call descriptor is a net loss.
Everything in the descriptor except the two buffer addresses is a pure function of the shape, the
dtype/layout, the buffer types and the core grid; the addresses now live in ``common_runtime_args``
(see the kernels), so a cached descriptor needs two scalars rewritten per call.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "reblock_permute"

TILE_H = TILE_W = 32
FACE_H = FACE_W = 16
GROUP_TILES = 32
IN_CB, OUT_CB, STAGE_CB = 0, 16, 24

# How the cached descriptor gets its two per-call addresses. Set on first use; kept as module state
# only so a probe can report which path the wheel took.
ADDR_WRITE_MODE = None

_CACHE: dict = {}
# Counters for the A/B harness: (eligible calls served, calls that fell through to ttnn.permute).
STATS = [0, 0]
# Why calls were refused, keyed by (reason, shape). A gate that never fires has to say why: the
# first wiring of this op served zero calls in a whole fold because the production shape is
# [1, 298, 298, 32] and the kernel required N % 32 == 0.
REJECTS: dict = {}


def _reject(reason, shape):
    k = (reason, tuple(shape))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return False


def _cache_key(x, out, device, reader_ct, writer_ct):
    """Everything the descriptor depends on except the two buffer addresses.

    The compile-time args of both TensorAccessors are in the key **verbatim**, so anything the
    accessor bakes into the kernel (buffer type, page size, shape, shard spec) is covered whether or
    not this function knows what it means. What is left is a pure function of ``(N, C, grid)``: the
    CB sizes, the core ranges, the work split and the per-core ``start`` / ``per_core`` / ``Nt`` /
    ``Ct``. The two addresses are the only per-call values and they are written on every call.
    """
    g = device.compute_with_storage_grid_size()
    return (
        device.id(),
        int(x.shape[1]), int(x.shape[3]),
        str(x.dtype), str(x.layout),
        str(x.memory_config()), str(out.memory_config()),
        g.x, g.y,
        tuple(reader_ct), tuple(writer_ct),
    )


def _build(x, out, device, reader_ct, writer_ct):
    N = int(x.shape[1])
    Ct = int(x.shape[3]) // TILE_W
    # The fold runs this at N=298, not at a multiple of 32, so the tile grid is ceil(N/32) in both
    # directions and the last row-group is ragged. The kernels take N and handle it.
    Nt = (N + TILE_H - 1) // TILE_H
    num_groups = Nt * Nt

    g = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
    )
    (_, core_grid, cg1, cg2, work1, work2) = ttnn.split_work_to_cores(all_cores, num_groups)

    tile_bytes = TILE_H * TILE_W * 2  # bf16

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes
        )
        return ttnn.CBDescriptor(
            total_size=depth * tile_bytes, core_ranges=core_grid, format_descriptors=[fmt]
        )

    # c_16 depth MUST be a multiple of the 32-tile group or the writer's L1 window wraps mid-group.
    cbs = [cb(IN_CB, 2), cb(OUT_CB, GROUP_TILES * 2), cb(STAGE_CB, 2)]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    reader_rt[cx][cy] = [start, per_core, Nt, N, Ct]
                    compute_rt[cx][cy] = [per_core * GROUP_TILES * Ct]
                    writer_rt[cx][cy] = [start, per_core, Nt, N, Ct]
                    start += per_core
    assert start == num_groups, (start, num_groups)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_reblock_permute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=reader_ct, runtime_args=reader_rt,
        common_runtime_args=[0], config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_reblock_permute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=writer_ct, runtime_args=writer_rt,
        common_runtime_args=[0], config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_reblock_permute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=[IN_CB, OUT_CB], runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    # Decide once how the addresses reach the descriptor. Mutating the kernels held inside the
    # cached ProgramDescriptor is the cheap path; if the binding hands back copies, fall back to
    # rebuilding the ProgramDescriptor from the cached kernel objects (still ~3 orders cheaper than
    # rebuilding the per-core runtime args).
    global ADDR_WRITE_MODE
    if ADDR_WRITE_MODE is None:
        probe = 0xABCD1234
        pd.kernels[0].common_runtime_args = [probe]
        got = list(pd.kernels[0].common_runtime_args)
        ADDR_WRITE_MODE = "in_place" if got == [probe] else "rebuild_pd"
        pd.kernels[0].common_runtime_args = [0]

    return {"pd": pd, "kernels": [reader, writer, compute], "cbs": cbs, "core_grid": core_grid}


def _prepare(x, out, device):
    reader_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [2, OUT_CB, TILE_H, TILE_W, FACE_H, FACE_W, STAGE_CB]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    key = _cache_key(x, out, device, reader_ct, writer_ct)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x, out, device, reader_ct, writer_ct)
    return entry


def reblock_permute(x, memory_config=None, device=None):
    """``ttnn.permute(x, (0, 3, 1, 2))`` for ``x`` of shape ``[1, N, N, C]`` bf16 TILE, C % 32 == 0."""
    device = device or x.device()
    mc = memory_config or x.memory_config()
    N, C = int(x.shape[1]), int(x.shape[3])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, C, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc
    )
    entry = _prepare(x, out, device)
    src, dst = x.buffer_address(), out.buffer_address()
    if ADDR_WRITE_MODE == "in_place":
        pd = entry["pd"]
        pd.kernels[0].common_runtime_args = [src]
        pd.kernels[1].common_runtime_args = [dst]
    else:
        reader, writer, compute = entry["kernels"]
        reader.common_runtime_args = [src]
        writer.common_runtime_args = [dst]
        pd = entry["pd"] = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute], semaphores=[], cbs=entry["cbs"]
        )
    STATS[0] += 1
    return ttnn.generic_op([x, out], pd)


def eligible(x, memory_config) -> bool:
    """The gate, measured against the wheel's own ``ttnn.permute`` on the card that runs it.

    Two things decide it: the destination buffer type and ``N``. On DRAM the custom move wins from
    N=256 upward on both wheels measured (qb1 / 0.67.4: 1.90x at 298 and 320; qb2 / 0.68.0: 1.5x).
    On L1 the margin is small and wheel-dependent, which is why the L1 window is narrow and why the
    whole gate ships default-OFF.

    The channel count is deliberately not part of the window: the kernel handles any ``C`` that is a
    multiple of 32, because the trunk's own chunk width depends on the compute grid.
    """
    if not _ENABLED:
        return False
    shape = [int(d) for d in x.shape]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != shape[2] or shape[3] % TILE_W:
        return _reject("shape", shape)
    N = shape[1]
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        return _reject("dtype_layout", shape)
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_out", shape)
    if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_in", shape)
    bt = memory_config.buffer_type
    if (bt == ttnn.BufferType.DRAM and N >= 256) or (bt == ttnn.BufferType.L1 and 288 <= N <= 352):
        return True
    return _reject(f"window_{bt}", shape)


# Whether `_channel_move` reaches for this kernel at all. Bit-exact: a permute is a pure index
# reordering, `torch.equal` against `ttnn.permute` at every shape in `eligible`'s window including
# the ragged group's output tile padding, and a live 298 aa protenix-v2 fold returns plDDT to the
# same six decimals with it on and off. Worth PLACEHOLDER_MS ms/fold on that fold, measured on the
# trimul block wall on qb1 card 3 at ttnn 0.67.4. Release-gated.
# PLACEHOLDER_STATUS
REBLOCK_PERMUTE = False
# `TT_BIO_REBLOCK_PERMUTE` stays as an out-of-process override for A/B harnesses; the default is
# the constant above, so the release gate and any in-process import can see and set it.
_ENABLED = os.environ.get("TT_BIO_REBLOCK_PERMUTE", "1" if REBLOCK_PERMUTE else "0") == "1"


def set_enabled(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global _ENABLED
    prev, _ENABLED = _ENABLED, bool(on)
    return prev
