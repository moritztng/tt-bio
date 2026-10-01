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

import collections
import os
from pathlib import Path

import ttnn

from . import core_split
from .envflags import env_flag
from . import genq
from . import ops as _ops
from .envflags import env_flag

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

# Whether the kernel exists at all, not whether a model uses it. The release gate is on the
# model side -- `AF2PairBlock.rne_kernel` is False and the tape entry is out of
# TT_BIO_TAPED_KERNELS -- and this is the switch a probe or an A/B flips to take the kernel out
# of a process that has already asked for it. Two switches for one lever is how the first wiring
# of this row measured a clean zero: `--rne-kernel 1` armed the model and the module was still
# off, so `eligible` returned before it could even count a decline.
_ENABLED = env_flag("TT_BIO_RNE_ADD_KERNEL", True)


def set_enabled(on: bool) -> bool:
    """A/B switch. Returns the previous state."""
    global _ENABLED
    prev, _ENABLED = _ENABLED, bool(on)
    return prev


def enabled() -> bool:
    return _ENABLED


def _gran(add_mode=None) -> int:
    cap = _DST_TILES // (2 if (ADD_MODE if add_mode is None else add_mode) == 1 else 1)
    return min(GRAN or cap, cap)


def _reject(reason, shape, other=None):
    """Count a decline and say what shape it was on. `other` is the second operand where the
    two disagreeing shapes are the reason -- a table of one of them cannot be read."""
    k = (reason, tuple(shape)) if other is None else (reason, tuple(shape), tuple(other))
    REJECTS[k] = REJECTS.get(k, 0) + 1
    STATS[1] += 1
    return False


def _tile_grid(t) -> tuple:
    """The PADDED tile grid, which is what the buffer holds and what the kernels index.

    TILE layout pads the last two dims each to 32 and every leading index gets its own padded
    `[H, W]` block, so the page count is `prod(leading) * ceil(H/32) * ceil(W/32)`. Flattening
    the leading dims into H first is a different count whenever H is not a multiple of 32:
    `[3, 50, 70]` holds 3 x 2 x 3 pages, not 5 x 3, and an op that walks the second number reads
    the wrong tiles (`widen_add`'s grade measured 0.29-0.67 rel L2 on exactly those shapes before
    this was fixed; every shape the round runs is aligned, so no shipped number moved).

    Tile padding is added and written like any other element: the wide path pads the same way and
    the padding of an add's operands is the padding of its result.
    """
    shape = [int(d) for d in t.shape]
    if len(shape) == 1:
        shape = [1] + shape
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return (lead * ((shape[-2] + TILE_H - 1) // TILE_H), (shape[-1] + TILE_W - 1) // TILE_W)


def _tile_count(t) -> int:
    ht, wt = _tile_grid(t)
    return ht * wt


def _core_shape(t) -> list:
    """The shape with leading 1s dropped.

    Two operands of one elementwise add can disagree on rank and still be the same tensor
    physically: the MSA track hands `_residual` an `[1, S, N, C]` activation and an `[S, N, C]`
    update, which is the same tile grid, the same element count and the same page order. A
    LEADING 1 is the only difference this tolerates -- an interior one is a broadcast, which is
    a different function and is declined.
    """
    shape = [int(d) for d in t.shape]
    while len(shape) > 1 and shape[0] == 1:
        shape.pop(0)
    return shape


def _split_plan(device, units):
    g = device.compute_with_storage_grid_size()
    key = (device.id(), g.x, g.y, units)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = (core_split.split_work_to_cores(g, units, 0) if units > 0 else None)
    return _SPLIT_CACHE[key]


def _cache_key(a, b, out, device, mode):
    """What decides which cached program serves this call, built only from cheap reads.

    This runs on EVERY dispatch, so what it costs is what every call costs. The obvious spelling
    -- `str(memory_config())` on both operands and the tuple of compile-time accessor args --
    read 0.0079 ms a call at 130 cores, more than the dispatch it was guarding once the dispatch
    got cheap (`state/perf10/bcx-GENQ.md` leg 4). Formatting a `MemoryConfig` into a string is
    the bulk of it.

    Enum members are hashable and compare by identity, so the buffer type and memory layout carry
    the same distinction at the price of an attribute read. The accessor compile-time args are
    NOT in the key: they are a pure function of the dtype, layout, shape and memory config that
    are, so keying on them would be keying twice, and they are built only on a miss.
    """
    mca, mcb, mco = a.memory_config(), b.memory_config(), out.memory_config()
    g = device.compute_with_storage_grid_size()
    return (
        device.id(), _tile_count(a), a.dtype, a.layout, b.dtype, b.layout,
        mca.buffer_type, mca.memory_layout, mcb.buffer_type, mcb.memory_layout,
        mco.buffer_type, mco.memory_layout, g.x, g.y,
        mode, _gran(mode[0]), out.dtype, genq.compact(),
    )


def _build(a, b, out, device, reader_ct, writer_ct, mode):
    num_tiles = _tile_count(a)
    plan = _split_plan(device, num_tiles)
    assert plan is not None, "no work to split"
    num_cores, core_grid, cg1, cg2, work1, work2 = plan

    add_mode, round_mode = mode
    gran = _gran(add_mode)

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
    cbs = [cb(A_CB, 2 * gran, a.dtype), cb(B_CB, 2 * gran, b.dtype),
           cb(OUT_CB, 2 * gran, out.dtype)]

    # EVERY placed core gets a slice, and the placement set is built from the same loop that
    # assigns them. A core placed without runtime args reads them as zero, so its writer takes
    # address 0 as its output base and writes into the bottom of DRAM -- silently, into whatever
    # tensor the allocator put there (`state/perf10/bcx-TABWD.md`, which cost that row four
    # bisects). `perf/bcx_p10_rneker/grade.py` reads the INPUT tensors back after every program to
    # prove this one does not.
    assign, first, placed = {}, 0, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    assign[(cx, cy)] = (placed, per_core)
                    first += 1
                    placed += per_core
    assert (first, placed) == (num_cores, num_tiles), (first, placed, num_cores, num_tiles)

    # The cheap dispatch path: the same slices, recomputed on the core from seven constants, so
    # the descriptor carries no per-core runtime args and the dispatch costs a third of what it
    # costs with them. `compact_plan` returns None unless that arithmetic reproduces `assign`
    # exactly, and then the per-core args below are what runs -- byte for byte today's program.
    plan_ct = genq.compact_plan(assign, core_grid) if genq.compact() else None
    genq_ct = [1] + plan_ct if plan_ct else [0, 0, 0, 0, 0, 0, 0, 0]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    if plan_ct is None:
        for (cx, cy), (fst, per_core) in assign.items():
            reader_rt[cx][cy] = [fst, per_core]
            compute_rt[cx][cy] = [per_core]
            writer_rt[cx][cy] = [fst, per_core]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_rne_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=reader_ct + genq_ct, runtime_args=reader_rt,
        common_runtime_args=[0, 0], config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_rne_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=writer_ct + genq_ct, runtime_args=writer_rt,
        common_runtime_args=[0], config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_rne_add.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[A_CB, B_CB, OUT_CB, gran, add_mode, round_mode] + genq_ct,
        runtime_args=compute_rt,
        config=_compute_config(a, b),
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
            "core_grid": core_grid, "num_cores": num_cores, "num_tiles": num_tiles,
            "compact": plan_ct is not None}


#: Circular buffers per core on Blackhole; the length `unpack_to_dest_mode` is indexed by CB id.
_NUM_CBS = 64


def _compute_config(a, b):
    """HiFi4 keeps every mantissa bit of a bfloat16 operand on the FPU path, and the 32-bit DEST
    is what makes the intermediate wider than either operand. Both are the kernel's accuracy
    claim, not a tuning choice.

    A float32 operand is unpacked straight to DEST. Through SrcA it would arrive as a 19-bit
    TF32 and `widen_add` would round the accumulator on every contribution.
    """
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       fp32_dest_acc_en=True)
    wide = [cb for cb, t in ((A_CB, a), (B_CB, b)) if t.dtype == ttnn.float32]
    if wide:
        modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
        for cb in wide:
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    return cfg


def _prepare(a, b, out, device, mode):
    key = _cache_key(a, b, out, device, mode)
    entry = _CACHE.get(key)
    if entry is None:
        # The miss path. Everything here is built once per (shape, placement, mode), never per
        # call.
        gran = _gran(mode[0])
        reader_ct = [gran]
        reader_ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())
        reader_ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
        writer_ct = [gran]
        writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        entry = _CACHE[key] = _build(a, b, out, device, reader_ct, writer_ct, mode)
    return entry


@_ops.fused_kernel("rne_add")
def rne_add(a, b, memory_config=None, out=None, device=None):
    """``round_rne_bf16(a + b)`` with both operands bfloat16 TILE and the same padded tile count.

    Neither operand is deallocated: the caller owns them, exactly as it owns the operands of the
    four-call path this replaces.
    """
    STATS[0] += 1
    return _dispatch(a, b, OUT_DTYPE, (ADD_MODE, ROUND_MODE), memory_config, out, device)


def _dispatch(a, b, out_dtype, mode, memory_config, out, device):
    device = device or a.device()
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            a.shape, out_dtype, ttnn.TILE_LAYOUT, device, memory_config or a.memory_config()
        )
    entry = _prepare(a, b, out, device, mode)
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
    if _core_shape(a) != _core_shape(b):
        return _reject("shape_mismatch", shape_a, [int(d) for d in b.shape])
    # The kernels index both operands and the result with ONE page index, so a broadcast, a
    # different padded tile count or a sharded operand is not this op.
    for t in (a, b):
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return _reject("sharded_in", shape_a)
    if memory_config is not None and \
            memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_out", shape_a)
    if _tile_grid(a) != _tile_grid(b):
        return _reject("tile_grid", shape_a, [int(d) for d in b.shape])
    return True


# ---------------------------------------------------------------------------------------------
# widen_add: the backward's gradient fan-in, `f32(a) + f32(b)`, on the same program.
#
# `autograd.Tensor.add_grad` promotes on the second contribution: `typecast(acc, f32)`,
# `typecast(grad, f32)`, `add`. The casts write float32 tensors the add reads straight back, and
# those edges are the seven largest write-then-reread chains in the round's backward
# (`state/perf10/bcx-p10-l1fuse.md`: 24.46 GB a round at `1x288x288x128`, 18.35 GB each on the
# triangle-attention bias at `288x1x288x384`). The promotion itself is load-bearing -- a mixed
# `ttnn.add(..., dtype=float32)` is 5-6 orders of magnitude less accurate, because ttnn's binary
# datapath follows the narrowest operand -- so this keeps the arithmetic and removes the round
# trips: widen in the unpacker, add in a 32-bit DEST with the SFPU (exact, like rne_add's
# ADD_MODE=1), pack float32. No rounding step: the float32 DEST is the result. Bytes per element
# 8 against 24 on the first promotion and 10 against 18 on every later one.
#
# A float32 + float32 call is declined: `ttnn.add` already moves 12 B/element there and there is
# no cast to remove.

#: The lever. Default OFF and release-gated; `autograd.Tensor.add_grad` is the only caller.
WIDEN_ADD = env_flag("TT_BIO_WIDEN_ADD", False)

#: Served and declined calls by name, cumulative; sample at a round boundary.
WIDEN_REACH: collections.Counter = collections.Counter()

_WIDEN_MODE = (1, 0)  # SFPU add, no rounding: the float32 DEST is packed as it is
_WIDE_IN = (ttnn.bfloat16, ttnn.float32)


def _widen_decline(reason) -> bool:
    WIDEN_REACH["declined: " + reason] += 1
    return False


def widen_eligible(a, b) -> bool:
    """What `widen_add` serves. Everything else falls through to the caller's widened path."""
    if not WIDEN_ADD:
        return False
    if _ops.taping():
        # Nothing tapes a backward, so this never fires on the fan-in; a taped caller would need
        # a tape entry, and there is none.
        return _widen_decline("taped")
    if a.dtype not in _WIDE_IN or b.dtype not in _WIDE_IN:
        return _widen_decline("dtype")
    if a.dtype == ttnn.float32 and b.dtype == ttnn.float32:
        return _widen_decline("both float32")
    if a.layout != ttnn.TILE_LAYOUT or b.layout != ttnn.TILE_LAYOUT:
        return _widen_decline("not TILE")
    if [int(d) for d in a.shape] != [int(d) for d in b.shape]:
        return _widen_decline("shape")
    for t in (a, b):
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return _widen_decline("sharded")
    return True


def widen_add(a, b):
    """``f32(a) + f32(b)`` rounded once to float32, into a new DRAM float32 tensor.

    `a` and `b` are TILE, same shape, each bfloat16 or float32 and not both float32 (the gate).
    Neither is deallocated.
    """
    WIDEN_REACH["served: " + ("first" if a.dtype == b.dtype else "later")] += 1
    return _dispatch(a, b, ttnn.float32, _WIDEN_MODE, ttnn.DRAM_MEMORY_CONFIG, None, None)


def round_add(a, b):
    """``round_rne_bf16(f32(a) + f32(b))``: `widen_add`'s sum, rounded once to bfloat16 at pack.

    The last add of a fan-in whose value is bf16 (`autograd.FANIN_CAST_FUSED`), so the float32
    sum is never written and cast. Same gate as `widen_add`; neither operand is deallocated.
    """
    WIDEN_REACH["served: round"] += 1
    return _dispatch(a, b, ttnn.bfloat16, (1, 1), ttnn.DRAM_MEMORY_CONFIG, None, None)


def widen_reach() -> dict:
    """A snapshot of `WIDEN_REACH`, for a round boundary or a run stamp."""
    return dict(WIDEN_REACH)
