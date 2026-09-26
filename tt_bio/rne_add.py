"""``round_rne_bf16(a + b)`` for two bfloat16 tensors, as one Tensix kernel.

AF2's Evoformer residual needs the bfloat16 sum rounded the way torch and JAX round it, and
`ttnn.add` does not compute that function: against a float64 reference it misses on 11.0 % of
random pairs, 6.1 % at a 256x operand ratio and 50.1 % on real ties, because its datapath follows
the narrowest operand. `af2.py::AF2PairBlock._residual` buys the right answer with four ttnn calls
-- widen both operands to float32, add, narrow the result -- which is **30 B/element** against the
6 B an elementwise bfloat16 add moves, and 1.05 s of the composed round's device column
(`state/perf10/bcx-CALLS.md`).

A compute kernel does it in one pass: unpack both operands into a 32-bit DEST, add there, round
once. Per element 6 B, **5.0x**. The rounding is the whole claim, so it is graded against a float64
host reference on four cases including real ties (`perf/bcx_p10_rneker/grade.py`), never against
`ttnn.add` -- **zero differing elements over 4 x 82,944, real ties included**, at 356.1 GB/s on
qb2 card 0 against a measured 442.3 GB/s DRAM roof.

Everything the kernel declines falls through to the caller's existing wide path, which is correct
at every shape.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from . import core_split
from . import ops as _ops

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "rne_add"

TILE_H = TILE_W = 32
A_CB, B_CB, OUT_CB = 0, 1, 16

#: Tiles per DEST acquire. A 32-bit DEST holds four tiles on Blackhole, and the SFPU add arm keeps
#: both operands in DEST, so its cap is two (`compute_rne_add.cpp`, and `bcx-TABWD.md` for where
#: the four came from).
_DST_TILES = 4

#: Which unit does the add and which one rounds. Both are compile-time arms of the one kernel,
#: both are graded in `perf/bcx_p10_rneker/grade.py`, and (1, 1) is the ONLY combination that is
#: bit-exact. Over 4 cases x 82,944 elements against `round_rne_bf16(exact_sum)` in float64:
#:
#:     ADD_MODE  ROUND_MODE   random   256x ratio   real ties   total wrong
#:     0 FPU     0 packer     11.001 %    6.116 %    50.055 %        97,234
#:     0 FPU     1 SFPU RNE    0.481 %    5.601 %     0.000 %         5,045
#:     1 SFPU    0 packer     10.526 %    0.700 %    50.055 %        92,348
#:     1 SFPU    1 SFPU RNE    0.000 %    0.000 %     0.000 %             0
#:
#: The two columns are independent defects and the table separates them: the packer breaks ties
#: away from zero (both ROUND_MODE=0 rows are ~50 % wrong on ties and agree with `ttnn.add`), and
#: the FPU's `add_tiles` is not the exact sum even into a 32-bit DEST (exact on 92.2 % of random
#: pairs where the SFPU add is exact on 100 %). Moving either is a silent precision change.
ADD_MODE = int(os.environ.get("TT_BIO_RNE_ADD_MODE", "1"))      # 0 FPU add_tiles, 1 SFPU
ROUND_MODE = int(os.environ.get("TT_BIO_RNE_ROUND_MODE", "1"))  # 0 packer, 1 SFPU round-to-even
GRAN = int(os.environ.get("TT_BIO_RNE_GRAN", "0")) or None

_DTYPE = ttnn.bfloat16
#: Probe-only: pack the DEST straight into a float32 result so the INTERMEDIATE can be read on the
#: host instead of inferred from what survives the narrowing. Production is always bfloat16.
OUT_DTYPE = ttnn.bfloat16
_ELEM = {ttnn.bfloat16: 2, ttnn.float32: 4}

# How the cached descriptor gets its three per-call addresses; see `reblock_permute` for the cost
# that makes this worth caching at all.
ADDR_WRITE_MODE = None

_CACHE: dict = {}
_SPLIT_CACHE: dict = {}

#: (calls served, calls declined). A process total cannot tell a round that served 108 from two
#: rounds that served 54 and 162, so every reader of this samples it at a round boundary.
STATS = [0, 0]
#: Why calls were refused, keyed by (reason, shape). A gate that never fires has to say why.
REJECTS: dict = {}

_ENABLED = os.environ.get("TT_BIO_RNE_ADD_KERNEL", "0") == "1"


def set_enabled(on: bool) -> bool:
    """A/B switch. Returns the previous state."""
    global _ENABLED
    prev, _ENABLED = _ENABLED, bool(on)
    return prev


def enabled() -> bool:
    return _ENABLED


def _gran() -> int:
    cap = _DST_TILES // (2 if ADD_MODE == 1 else 1)
    return min(GRAN or cap, cap)


def _reject(reason, shape):
    k = (reason, tuple(shape))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return False


def _tile_count(t) -> int:
    """The PADDED tile count, which is what the buffer holds and what the kernels index.

    Tile padding is added and written like any other element: the wide path pads the same way and
    the padding of a residual's operands is the padding of its result.
    """
    shape = [int(d) for d in t.shape]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    ht = (rows + TILE_H - 1) // TILE_H
    wt = (shape[-1] + TILE_W - 1) // TILE_W
    return ht * wt


def _split_plan(device, units):
    g = device.compute_with_storage_grid_size()
    key = (device.id(), g.x, g.y, units)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = (core_split.split_work_to_cores(g, units, 0) if units > 0 else None)
    return _SPLIT_CACHE[key]


def _cache_key(a, out, device, reader_ct, writer_ct):
    g = device.compute_with_storage_grid_size()
    return (
        device.id(), _tile_count(a), str(a.dtype), str(a.layout),
        str(a.memory_config()), str(out.memory_config()),
        g.x, g.y, tuple(reader_ct), tuple(writer_ct),
        ADD_MODE, ROUND_MODE, _gran(), str(OUT_DTYPE),
    )


def _build(a, out, device, reader_ct, writer_ct):
    num_tiles = _tile_count(a)
    plan = _split_plan(device, num_tiles)
    assert plan is not None, "no work to split"
    num_cores, core_grid, cg1, cg2, work1, work2 = plan

    gran = _gran()

    def cb(idx, depth, dtype):
        tile_bytes = TILE_H * TILE_W * _ELEM[dtype]
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=dtype, page_size=tile_bytes
        )
        return ttnn.CBDescriptor(
            total_size=depth * tile_bytes, core_ranges=core_grid, format_descriptors=[fmt]
        )

    # Depth 2*gran on every CB: the dataflow kernels reserve `gran` tiles and walk them from one
    # `get_write_ptr`, so a depth that is not a multiple of `gran` would wrap a block mid-walk.
    cbs = [cb(A_CB, 2 * gran, _DTYPE), cb(B_CB, 2 * gran, _DTYPE),
           cb(OUT_CB, 2 * gran, OUT_DTYPE)]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # EVERY placed core gets runtime args, and the placement set is built from the same loop that
    # writes them. A core placed without runtime args reads them as zero, so its writer takes
    # address 0 as its output base and writes into the bottom of DRAM -- silently, into whatever
    # tensor the allocator put there (`state/perf10/bcx-TABWD.md`, which cost that row four
    # bisects). `perf/bcx_p10_rneker/grade.py` reads the INPUT tensors back after every program to
    # prove this one does not.
    first, placed = 0, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    reader_rt[cx][cy] = [placed, per_core]
                    compute_rt[cx][cy] = [per_core]
                    writer_rt[cx][cy] = [placed, per_core]
                    first += 1
                    placed += per_core
    assert (first, placed) == (num_cores, num_tiles), (first, placed, num_cores, num_tiles)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_rne_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=reader_ct, runtime_args=reader_rt,
        common_runtime_args=[0, 0], config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_rne_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=writer_ct, runtime_args=writer_rt,
        common_runtime_args=[0], config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_rne_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[A_CB, B_CB, OUT_CB, gran, ADD_MODE, ROUND_MODE],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            # HiFi4 keeps every mantissa bit of a bfloat16 operand on the FPU path, and the 32-bit
            # DEST is what makes the intermediate wider than either operand. Both are the kernel's
            # accuracy claim, not a tuning choice.
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    global ADDR_WRITE_MODE
    if ADDR_WRITE_MODE is None:
        probe = 0xABCD1234
        pd.kernels[0].common_runtime_args = [probe, 0]
        ADDR_WRITE_MODE = ("in_place" if list(pd.kernels[0].common_runtime_args)[0] == probe
                           else "rebuild_pd")
        pd.kernels[0].common_runtime_args = [0, 0]

    return {"pd": pd, "kernels": [reader, writer, compute], "cbs": cbs,
            "core_grid": core_grid, "num_cores": num_cores, "num_tiles": num_tiles}


def _prepare(a, b, out, device):
    reader_ct = [_gran()]
    reader_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
    writer_ct = [_gran()]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    key = _cache_key(a, out, device, reader_ct, writer_ct)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(a, out, device, reader_ct, writer_ct)
    return entry


@_ops.fused_kernel("rne_add")
def rne_add(a, b, memory_config=None, out=None, device=None):
    """``round_rne_bf16(a + b)`` with both operands bfloat16 TILE and the same padded tile count.

    Neither operand is deallocated: the caller owns them, exactly as it owns the operands of the
    four-call path this replaces.
    """
    device = device or a.device()
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            a.shape, OUT_DTYPE, ttnn.TILE_LAYOUT, device, memory_config or a.memory_config()
        )
    entry = _prepare(a, b, out, device)
    common_r = [a.buffer_address(), b.buffer_address()]
    common_w = [out.buffer_address()]
    if ADDR_WRITE_MODE == "in_place":
        pd = entry["pd"]
        pd.kernels[0].common_runtime_args = common_r
        pd.kernels[1].common_runtime_args = common_w
    else:
        reader, writer, compute = entry["kernels"]
        reader.common_runtime_args = common_r
        writer.common_runtime_args = common_w
        pd = entry["pd"] = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute], semaphores=[], cbs=entry["cbs"]
        )
    STATS[0] += 1
    return ttnn.generic_op([a, b, out], pd)


def eligible(a, b, memory_config) -> bool:
    """What the kernel serves. Everything else falls through to the caller's wide path."""
    if not _ENABLED:
        return False
    shape_a = [int(d) for d in a.shape]
    # A `generic_op` has no tape entry of its own, so without one registered this kernel would
    # silently cut the graph. `taped_ttnn._kernel("rne_add")` is the entry; until it is installed
    # (`TT_BIO_TAPED_KERNELS`) every taped call declines here and the four-call wide path runs.
    if _ops.declines_under_tape("rne_add"):
        return _reject("taped_no_entry", shape_a)
    if a.dtype != _DTYPE or b.dtype != _DTYPE:
        return _reject("dtype", shape_a)
    if a.layout != ttnn.TILE_LAYOUT or b.layout != ttnn.TILE_LAYOUT:
        return _reject("layout", shape_a)
    if [int(d) for d in b.shape] != shape_a:
        return _reject("shape_mismatch", shape_a)
    # The kernels index both operands and the result with ONE page index, so a broadcast, a
    # different padded tile count or a sharded operand is not this op.
    for t in (a, b):
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return _reject("sharded_in", shape_a)
    if memory_config is not None and \
            memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_out", shape_a)
    if _tile_count(a) != _tile_count(b):
        return _reject("tile_count", shape_a)
    return True
