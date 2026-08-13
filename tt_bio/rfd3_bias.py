"""RFD3's sparse attention bias built in one pass, as a hand-written Tensix kernel.

Replaces three ops on the hottest path in the model -- the cached ``-1e4`` template, the
``ttnn.scatter`` that writes the pair bias into it, and the ``bf16 -> fp32`` typecast --
with one kernel that materialises the fp32 bias directly.

**Why.** The bias is ``[1, 4, 3359, 3360]``, 45.14 M elements, and only ``L*K = 430 k``
of them per head carry anything: the other 96.2 % are the mask constant. ``ttnn.scatter``
is out-of-place, so it copies all 45.14 M and then writes the 430 k, and its copy is
per-element-rate limited at 9.6 G elem/s where a clone of the same tensor runs at 92
(``state/rfd3-host-half.md`` §3, ``ttnn-scatter-gather-per-element-limited``). Measured on
qb1 at ttnn 0.67.4 the three ops cost **5.475 ms/call** -- scatter 4.683 + typecast 0.792 --
against the 0.47 ms that writing 180.6 MB of fp32 at this card's measured 385 GB/s clone
roof costs. Nothing about dtype or layout moves the scatter; only not doing the copy does.

**Bit-exact by construction, and checked rather than argued.** Widening bf16 to fp32 is a
16-bit left shift. The fill is passed in as the fp32 bit pattern of ``bfloat16(-1e4)``,
which is ``-9984.0`` and is exactly what the shipped template held after its own widen, so
the kernel is not allowed to re-derive the constant and get ``-10000.0``.
``scripts/rfd3_port/p36_bias_kernel_probe.py`` gates it with ``torch.equal`` against the
shipped three-op chain at the production shape.

**Why the descriptor is cached.** ``generic_op`` takes the whole program description per
call; building the per-core runtime args in Python costs more than the op's device time
(``ttnn-generic-op-no-build-deploy-route``, and ``reblock_permute`` measured 155 us against
91 us). Everything except the three buffer addresses is a pure function of
``(H, L, K, dtype, grid)``, and the addresses live in ``common_runtime_args``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import torch
import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "rfd3_bias"

TILE_H = TILE_W = 32
IDX_CB, BIAS_CB, OUT_CB, SLOT_CB = 0, 1, 2, 3
# The second data-movement RISC's four CBs sit CB_STRIDE above the first's.
CB_STRIDE = 4
# Output tiles in flight per core. Must be a power of two (the kernel masks with it). This is
# the op's only real tuning knob: the write is 180.6 MB at the production shape and the card's
# measured clone roof is 385 GB/s, so 0.47 ms is the floor, and how close the kernel gets to
# it is decided by how many 4 KB DRAM writes a core can have outstanding. Set from a sweep,
# see perf/p36/slot_sweep.json.
OUT_SLOTS = int(os.environ.get("RFD3_SPARSE_BIAS_SLOTS", "16"))

ADDR_WRITE_MODE = None
_CACHE: dict = {}
# (calls served, calls declined) for the A/B harness.
STATS = [0, 0]
REJECTS: dict = {}


def _fill_bits(value: float) -> int:
    """fp32 bit pattern of ``float32(bfloat16(value))``.

    The shipped path fills a **bf16** template and then widens it, so the mask constant the
    softmax actually sees is ``bfloat16(-1e4) = -9984.0``, not ``-10000.0``. Writing the
    latter would be a different tensor -- harmless after ``exp`` (both underflow to exactly
    0.0 in fp32) but not ``torch.equal``, and a parity gate that has to reason about which
    differences are harmless is not a parity gate.
    """
    bf = torch.tensor([value], dtype=torch.bfloat16).to(torch.float32).item()
    return struct.unpack("<I", struct.pack("<f", bf))[0]


def _even_split(n_units: int, cores: list[tuple[int, int]]) -> list[int]:
    """``n_units`` split as evenly as possible over ``cores``, remainder to the front.

    ``ttnn.split_work_to_cores`` is not used: it raises ``TT_FATAL`` when ``units % cores``
    is a non-zero multiple of the grid height (``ttnn-split-work-to-cores-grid-height-holes``),
    and ``H * It`` lands in a hole at the production shape -- 420 groups on qb1's 130 cores
    leaves 30, a multiple of the grid height 10. Nothing here needs the utility's core-range
    grouping, because every core gets its own runtime args anyway.
    """
    per, rem = divmod(n_units, len(cores))
    return [per + 1 if i < rem else per for i in range(len(cores))]


def _cache_key(bias, idx, out, device, ct_args):
    return (
        device.id(),
        tuple(int(d) for d in bias.shape), tuple(int(d) for d in idx.shape),
        tuple(int(d) for d in out.shape),
        str(bias.dtype), str(idx.dtype), str(out.dtype),
        str(bias.layout), str(idx.layout), str(out.layout),
        str(bias.memory_config()), str(idx.memory_config()), str(out.memory_config()),
        tuple(ct_args),
    )


def _build(bias, idx, out, device, ct_head, acc_args):
    H, L, K = int(bias.shape[1]), int(bias.shape[2]), int(bias.shape[3])
    N = int(out.shape[3])
    It = (L + TILE_H - 1) // TILE_H
    Jt = N // TILE_W
    Kt = (K + TILE_W - 1) // TILE_W

    g = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
    )
    cores = [(cx, cy) for cx in range(g.x) for cy in range(g.y)]
    counts = _even_split(H * It, cores)

    def cb(idx_, fmt, page_size, depth):
        return ttnn.CBDescriptor(
            total_size=depth * page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx_, data_format=fmt, page_size=page_size)
            ],
        )

    # One set of four per data-movement RISC: the two run disjoint bands, so they need
    # disjoint index/bias/output/cursor windows and never synchronise.
    def cb_set(base):
        return [
            cb(base + IDX_CB, ttnn.uint32, K * 4, TILE_H),
            cb(base + BIAS_CB, ttnn.bfloat16, TILE_H * TILE_W * 2, Kt),
            cb(base + OUT_CB, ttnn.float32, TILE_H * TILE_W * 4, OUT_SLOTS),
            cb(base + SLOT_CB, ttnn.uint32, TILE_H * 4, OUT_SLOTS),
        ]

    cbs = cb_set(0) + cb_set(CB_STRIDE)

    # Both data-movement RISCs run the same kernel over DISJOINT bands. The op is not DRAM
    # bound -- at 4 slots one RISC reaches 91 GB/s of a measured 385 GB/s roof and adding
    # slots stops helping at 8, because the cost is the kernel's own volatile L1 traffic
    # (the index scan, the pokes and the repair) and not the write. A second RISC is the
    # cheapest way to buy that back: the bands are independent, so the two kernels never
    # interact, need no semaphore, and even issue on different NOCs.
    rt_a, rt_b = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for (cx, cy), n in zip(cores, counts):
        half = n // 2
        rt_a[cx][cy] = [start, n - half]
        rt_b[cx][cy] = [start + n - half, half]
        start += n
    assert start == H * It, (start, H * It)

    tail = [It, Jt, Kt, K, L, _fill_bits(-1e4), OUT_SLOTS] + list(acc_args)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer_sparse_bias.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=list(head) + tail, runtime_args=rt,
            common_runtime_args=[0, 0, 0], config=cfg,
        )
        for head, rt, cfg in (
            (ct_head, rt_a, ttnn.WriterConfigDescriptor()),
            ([c + CB_STRIDE for c in ct_head], rt_b, ttnn.ReaderConfigDescriptor()),
        )
    ]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    global ADDR_WRITE_MODE
    if ADDR_WRITE_MODE is None:
        probe = [0xABCD1234, 0x1234ABCD, 0xFEED0001]
        pd.kernels[0].common_runtime_args = probe
        ADDR_WRITE_MODE = (
            "in_place" if list(pd.kernels[0].common_runtime_args) == probe else "rebuild_pd"
        )
        pd.kernels[0].common_runtime_args = [0, 0, 0]
    return {"pd": pd, "kernels": kernels, "cbs": cbs}


def sparse_bias_fp32(pair_bias, idx_rm, out=None, memory_config=None, device=None):
    """``typecast(scatter(full(-1e4), 3, idx, pair_bias), fp32)`` in one pass.

    ``pair_bias`` is ``[1, H, L, K]`` bf16 TILE, ``idx_rm`` is ``[1, 1, L, K]`` uint32
    ROW_MAJOR (the index is the same for every head, so only one copy is uploaded and the
    shipped path's 4-way ``ttnn.concat`` goes away with it), and the output is
    ``[1, H, L, align_tile(L)]`` fp32 TILE.

    ``idx_rm`` must be sorted ascending along the last axis, which
    ``_create_attention_indices`` guarantees -- it ends in ``torch.sort``. The kernel walks
    each row once with a cursor as the tile-column window advances, so an unsorted row
    would silently drop entries.
    """
    device = device or pair_bias.device()
    H, L, K = int(pair_bias.shape[1]), int(pair_bias.shape[2]), int(pair_bias.shape[3])
    N = ((L + TILE_W - 1) // TILE_W) * TILE_W
    if out is None:
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, L, N]), ttnn.float32, ttnn.TILE_LAYOUT, device, mc
        )

    ct_head = [IDX_CB, BIAS_CB, OUT_CB, SLOT_CB]
    acc_args = (
        list(ttnn.TensorAccessorArgs(pair_bias).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(idx_rm).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    )
    key = _cache_key(pair_bias, idx_rm, out, device, tuple(ct_head) + tuple(acc_args))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(pair_bias, idx_rm, out, device, ct_head, acc_args)

    addrs = [pair_bias.buffer_address(), idx_rm.buffer_address(), out.buffer_address()]
    if ADDR_WRITE_MODE == "in_place":
        pd = entry["pd"]
        for k in pd.kernels:
            k.common_runtime_args = addrs
    else:
        for k in entry["kernels"]:
            k.common_runtime_args = addrs
        pd = entry["pd"] = ttnn.ProgramDescriptor(
            kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"]
        )
    STATS[0] += 1
    return ttnn.generic_op([pair_bias, idx_rm, out], pd)


def _reject(reason, shape):
    k = (reason, tuple(shape))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return False


def eligible(pair_bias, idx_rm) -> bool:
    """Whether the kernel may serve this call. See the module docstring for the measurement."""
    if not _ENABLED:
        return False
    if idx_rm is None:
        return _reject("no_row_major_index", [0])
    shape = [int(d) for d in pair_bias.shape]
    if len(shape) != 4 or shape[0] != 1 or shape[3] % TILE_W:
        return _reject("shape", shape)
    if pair_bias.dtype != ttnn.bfloat16 or pair_bias.layout != ttnn.TILE_LAYOUT:
        return _reject("dtype_layout", shape)
    if idx_rm.dtype != ttnn.uint32 or idx_rm.layout != ttnn.ROW_MAJOR_LAYOUT:
        return _reject("idx_dtype_layout", shape)
    if pair_bias.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_in", shape)
    return True


RFD3_SPARSE_BIAS = False   # release-gated: opt-in until the fold A/B and the trajectory gate land
_ENABLED = os.environ.get(
    "RFD3_SPARSE_BIAS", "1" if RFD3_SPARSE_BIAS else "0"
) == "1"


def set_enabled(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global _ENABLED
    prev, _ENABLED = _ENABLED, bool(on)
    return prev
