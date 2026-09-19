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

from . import core_split
from .envflags import env_flag

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "reblock_permute"
KERNEL_DIR_BACK = Path(__file__).resolve().parent / "kernels" / "reblock_permute_back"

TILE_H = TILE_W = 32
FACE_H = FACE_W = 16
GROUP_TILES = 32
IN_CB, OUT_CB, STAGE_CB = 0, 16, 24

# How the cached descriptor gets its two per-call addresses. Set on first use; kept as module state
# only so a probe can report which path the wheel took.
ADDR_WRITE_MODE = None

# The element width the kernels are built for. bf16 at every call site in the engine; a probe moves
# it so the BYTE SENSITIVITY of this path can be measured instead of assumed. The three gates below
# admit only `_DTYPE`, so moving this alone changes nothing a caller can reach -- a probe that wants
# a different width sets it and forces the call. `perf/ttx_reblock_bfp8/byte_sensitivity.py` is the
# reader; its header says why fp32 (2x the bytes at an unchanged transaction count) is the control
# that decides whether bfp8 (half the bytes) could ever pay here.
_DTYPE = ttnn.bfloat16
_ELEM_BYTES = {ttnn.bfloat16: 2, ttnn.float32: 4}


def set_dtype(dtype):
    """Probe-only: build the kernels for ``dtype``. Returns the previous value."""
    global _DTYPE
    prev, _DTYPE = _DTYPE, dtype
    return prev


def _elem():
    return _ELEM_BYTES[_DTYPE]


_CACHE: dict = {}
_CACHE_BACK: dict = {}
# Counters for the A/B harness: (eligible calls served, calls that fell through to ttnn.permute).
STATS = [0, 0]
# The same pair for the back direction, counted separately so one A/B can read both legs.
STATS_BACK = [0, 0]
# Why calls were refused, keyed by (reason, shape). A gate that never fires has to say why: the
# first wiring of this op served zero calls in a whole fold because the production shape is
# [1, 298, 298, 32] and the kernel required N % 32 == 0.
REJECTS: dict = {}


# --- the wheel's own work split has a hole, and `core_split` fills it ------------------------------
#
# `ttnn.split_work_to_cores` raises `TT_FATAL @ work_split.cpp:305: remaining == 0` whenever
# `units > cores` and `units % cores` is a non-zero multiple of the grid height, because it anchors
# core group 2 at the bottom of the next column instead of the top. `tt_bio.core_split` performs the
# same split in Python and without that bug -- its docstring carries the upstream one-word fix we are
# pinned below -- so every shape gets the whole grid.
#
# What this replaced was a memoised search for the largest rectangular sub-grid the wheel would
# accept, plus a `work_split` refusal in each of the three gates for the shapes where even that
# failed. The search always found something on an 11x10 grid, so no shape was actually losing the
# leg here, but it cost the busiest core an extra group on the affected bands: the back leg at
# N=640, C=128 ran its 1600 groups 16 deep on a 10x10 sub-grid where the full 11x10 grid runs them
# 15 deep. `scripts/verify_core_split.py` prints that census and checks the replacement against the
# wheel on every unit count the wheel can serve.

# How many cores a split may use, 0 meaning the whole grid. The whole grid is right at every shape
# measured and this exists only to reproduce a reduced grid for an A/B.
#
# It used to look like a real tuning knob: sweeping it moved this leg 1.10x-1.19x with optima at
# 80/100/88 cores rather than 110, and the best count differed per shape. That was an artifact of
# how groups are handed to cores, not a property of the core count -- see `WALK` below for the bank
# arithmetic. Once the aliasing is removed the cost is a function of depth alone: the 400-group
# 80..99-core band lands within 1.2% of itself where it spread 1.30x, and the whole grid is the
# fastest point at all four shapes. So there was never a core count to fit.
# `perf/ttx_splitwork/core_count_sweep.py` is the harness and the JSON sits beside it.
REBLOCK_CORES = int(os.environ.get("TT_BIO_REBLOCK_CORES", "0"))

# How a core walks the groups it owns. Every mode gives every core the SAME groups and the same
# count, so the output is bit-identical; what moves is which groups are in flight together, and
# therefore which DRAM banks are live at once.
#
#   "block"   (default, and what main ships) the contiguous run of `per_core` groups.
#   "stride"  round-robin: core i takes groups i, i+num_cores, ... Concurrent groups become
#             consecutive indices, so every bank is live, at the cost of page locality.
#   "rotate"  the contiguous run kept, started at the core's own phase `i % per_core`. Spreads the
#             banks without scattering the pages.
#
# All three are the same kernel loop: `num_groups` steps of `group_stride` from `first_group`, with
# a fold back to `group_wrap_lo` at `group_wrap_hi`.
#
# The default stays "block" because NEITHER alternative is safe, which was measured and is the
# whole finding. Aliasing is real and large: page p lives in bank p % banks, every page of group g
# is congruent to g, and a contiguous block puts core k at k*w, so the machine sits on
# banks/gcd(w, banks) of them -- two of eight at w = 4, which reads 1.71x slower per wave at
# identical traffic. Fixing it is worth 1.10x-1.44x. But the shapes that pay it are not the shapes
# a fold asks for. On the Boltz-2 ladder (`perf/ttx_splitwork/shape_census_ladder.json`) 298 aa
# runs the forward leg at 100 groups over 110 cores, one group a core, where all three walks are
# the same walk; every larger size runs only the gated and back legs, at 1024, 1600 and 4096
# groups. Measured there, against "block" on the whole grid
# (`prod_shapes_ab.json`, `prod_ladder_ab.json`):
#
#             512 aa   640 aa   1024 aa
#   gated     stride    0.964x   1.082x   0.910x        rotate  1.030x  0.753x  1.004x
#   back      stride    1.019x   1.023x   1.188x        rotate  1.018x  0.831x  1.077x
#
# Each column wants a different walk and each alternative regresses by up to 33% somewhere, so no
# fixed walk ships. Neither does a rule: `work1 >= banks`, fitted on six Blackhole shapes, mispicks
# twice on Wormhole's 12 banks (`walk_wh_j10glx02c0.json`) and picks the 0.753x at 640 aa. The bank
# arithmetic alone cannot choose either -- `perf/ttx_splitwork/bank_model.py` computes the exact
# per-wave bank histogram from the kernels' own index expressions and gets "block is never the
# cheapest walk" right at 12 of 12 while picking the device's winner at only 6, because the forward
# and back legs at N=960 have IDENTICAL bank costs and opposite winners. What separates them is the
# per-kernel issue pattern, which an index model cannot see.
#
# So: the knob is understood, it is bit-exact, and it stays off. The flag is here so a future pass
# can re-open it with wider coverage without rebuilding any of this.
WALK = os.environ.get("TT_BIO_REBLOCK_WALK", "block")
_NO_WRAP = 0xFFFFFFFF


def _walk(mode, i, block, per_core, num_cores):
    """``(first_group, group_stride, group_wrap_hi, group_wrap_lo)`` for linear core index ``i``.

    ``block`` is where this core's contiguous run of ``per_core`` groups starts.
    """
    if mode == "stride":
        return i, num_cores, _NO_WRAP, 0
    if mode == "rotate" and per_core > 1:
        return block + i % per_core, 1, block + per_core, block
    return block, 1, _NO_WRAP, 0

_SPLIT_CACHE: dict = {}


def _split_plan(device, units):
    """The work split for ``units`` groups over the whole compute grid.

    ``(grid_x, grid_y, <the six-tuple ttnn.split_work_to_cores returns>)``, or ``None`` when there
    is no work to split at all. Cached per ``(device, grid, units, REBLOCK_CORES)`` because
    `_channel_move` runs 4352 times in a 298 aa fold, and with the knob in the key so an in-process
    A/B can move it between two arms.
    """
    g = device.compute_with_storage_grid_size()
    key = (device.id(), g.x, g.y, units, REBLOCK_CORES)
    if key not in _SPLIT_CACHE:
        plan = ((g.x, g.y, core_split.split_work_to_cores(g, units, REBLOCK_CORES))
                if units > 0 else None)
        _SPLIT_CACHE[key] = plan
    return _SPLIT_CACHE[key]


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
        g.x, g.y, WALK,
        tuple(reader_ct), tuple(writer_ct),
    )


def _build(x, out, device, reader_ct, writer_ct):
    N = int(x.shape[1])
    Ct = int(x.shape[3]) // TILE_W
    # The fold runs this at N=298, not at a multiple of 32, so the tile grid is ceil(N/32) in both
    # directions and the last row-group is ragged. The kernels take N and handle it.
    Nt = (N + TILE_H - 1) // TILE_H
    num_groups = Nt * Nt

    plan = _split_plan(device, num_groups)
    # `eligible` has already refused any shape with no plan, so this cannot fire from the production
    # path. It stays as an assertion because a direct caller of `reblock_permute` bypasses the gate.
    assert plan is not None, f"no expressible work split for {num_groups} groups"
    _, _, (num_cores, core_grid, cg1, cg2, work1, work2) = plan

    tile_bytes = TILE_H * TILE_W * _elem()

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=_DTYPE, page_size=tile_bytes
        )
        return ttnn.CBDescriptor(
            total_size=depth * tile_bytes, core_ranges=core_grid, format_descriptors=[fmt]
        )

    # c_16 depth MUST be a multiple of the 32-tile group or the writer's L1 window wraps mid-group.
    cbs = [cb(IN_CB, 2), cb(OUT_CB, GROUP_TILES * 2), cb(STAGE_CB, 2)]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # `first` is this core's linear index, `block` the start of its contiguous run of groups. Both
    # counters are kept for every mode because `_walk` needs each of them; `placed` is the audit.
    first, placed, block = 0, 0, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    g0, gs, ghi, glo = _walk(WALK, first, block, per_core, num_cores)
                    reader_rt[cx][cy] = [g0, per_core, Nt, N, Ct, gs, ghi, glo]
                    compute_rt[cx][cy] = [per_core * GROUP_TILES * Ct]
                    writer_rt[cx][cy] = [g0, per_core, Nt, N, Ct, gs, ghi, glo]
                    first += 1
                    block += per_core
                    placed += per_core
    assert (first, placed) == (num_cores, num_groups), (first, placed, num_cores, num_groups)

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
    writer_ct = [_elem(), OUT_CB, TILE_H, TILE_W, FACE_H, FACE_W, STAGE_CB]
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
        ttnn.Shape([1, C, N, N]), _DTYPE, ttnn.TILE_LAYOUT, device, mc
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


# The L1 leg's window edges, named so a fold-level A/B can move one of them in-process without
# editing the gate. See eligible()'s docstring for what measured them.
#
# The env overrides exist because `predict` folds in spawned worker processes, so setting the
# module attribute in the launcher does not reach the code that reads it, and an in-process A/B
# is only possible for a caller that folds in its own process. They default to the measured
# constants, so with neither variable set this is byte-for-byte the shipped gate.
#
# Why they were added: ESMFold2 on Wormhole asks this gate 8672 times per fold at 256 aa and is
# refused every time on `window_BufferType.L1`, because its call sites want an L1 output and no
# rung of the 256/512/640/768/896/1024 size ladder falls inside 288..352. The 352 edge is one
# grid's collapse point (qb1 13x10) applied to every grid, while qb2 measured 1.02-1.62x wins on
# an L1 output out to N=544. Whether the window should be wider on a given card type is a
# measurement, and this is what makes it measurable per rung.
L1_N_MIN = int(os.environ.get("TT_BIO_REBLOCK_L1_N_MIN", "288"))
L1_N_MAX = int(os.environ.get("TT_BIO_REBLOCK_L1_N_MAX", "352"))


def eligible(x, memory_config) -> bool:
    """The gate, measured against the wheel's own ``ttnn.permute`` on the card that runs it.

    Two things decide it: the destination buffer type and ``N``. On DRAM the custom move wins from
    N=256 upward on both wheels measured (qb1 / 0.67.4: 1.90x at 298 and 320; qb2 / 0.68.0: 1.5x).
    On L1 the margin is smaller and grid-dependent. It opens at 288 because below that there are
    fewer work groups than cores and the per-call cost is not amortised: N=256 on an L1 output
    measures 0.952x on 110 cores, a real loss, and it is the shape boltzgen runs 2384 of its 3024
    channel moves on.

    The upper edge stays 352. It was widened to 544 on qb2 evidence and reverted: the widening is
    worth 0.000 s/fold at 512 aa (the fit test already routes the pair tensor to DRAM, where the leg
    is open, so 52224 of 52224 moves were already served and 0 declined), and re-measured on qb1's
    13x10 grid it does not reproduce -- two runs there read 0.68/0.72x at N=320, 0.65/0.86x at N=384,
    and a run-to-run spread up to 23 % at N=512. A no-op with ambiguous cross-grid evidence is not
    worth a shipped behaviour change. The qb2 band below is kept because it is the measurement, and
    the qb1 repeats are in perf/bigswing/reblock_window_band_qb1c0{,_r2}.json.

    The qb2 measurement was: 544, on the 11x10 grid at ttnn 0.68.0
    (``perf/bigswing/reblock_window_band_qb2c0.json``): every N in {320, 352, 384, 416, 448, 480,
    512, 544} wins on an L1 output and every one is ``torch.equal`` against ``ttnn.permute`` --
    1.3317 / 1.0150 / 1.1500 / 1.3549 / 1.5587 / 1.3401 / 1.3958 / 1.6163x. There is no cliff in
    that range, and the weakest point is N=352, which is where the window used to close. That 352
    came from qb1's 13x10 grid, where Nt=12 puts 144 groups on 130 cores and the win was measured
    to collapse to 1.002x. The band above 352 has NOT been re-measured on a 130-core grid, so this
    edge is qb2-evidenced only and a qb1 re-measure is owed before it ships.

    The channel count is deliberately not part of the window: the kernel handles any ``C`` that is a
    multiple of 32, because the trunk's own chunk width depends on the compute grid.
    """
    if not _ENABLED:
        return False
    shape = [int(d) for d in x.shape]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != shape[2] or shape[3] % TILE_W:
        return _reject("shape", shape)
    N = shape[1]
    if x.dtype != _DTYPE or x.layout != ttnn.TILE_LAYOUT:
        return _reject("dtype_layout", shape)
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_out", shape)
    if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_in", shape)
    bt = memory_config.buffer_type
    if not ((bt == ttnn.BufferType.DRAM and N >= 256)
            or (bt == ttnn.BufferType.L1 and L1_N_MIN <= N <= L1_N_MAX)):
        return _reject(f"window_{bt}", shape)
    return True


# Whether `_channel_move` reaches for this kernel at all. Bit-exact: a permute is a pure index
# reordering, `torch.equal` against `ttnn.permute` at all 24 shapes in `eligible`'s window including
# the ragged group's output tile padding, and twelve on/off fold pairs across five models on both
# grids write byte-identical structures. Worth 209-251 ms/fold on a 298 aa protenix-v2 fold: four
# sessions on byte-identical code (209.3 / 217.7 / 218.5 / 251.2), qb1 at ttnn 0.67.4, all four read
# on the trimul block wall because that host's fold-wall A/A floor runs to 1480 ms and cannot
# resolve the effect. 4352 of 4352 eligible calls served in a live fold; esmfold2 serves 4336 of its
# own at 298 aa and none at 117 aa, where the window declines them. openfold3, boltz2 and opendde
# all gain (+236 / +314 / +681 ms/fold as qb2 ratios). Every model here is N-dependent, boltzgen
# included: it serves none of the 3024 moves in an `examples/binder.yaml` design, whose pair track is
# the [1,256,256,64] shape the window excludes for losing, and 4768 of 5408 against a 214-residue
# target, which lands at N=320.
# Evidence: state/protenix-trunk--y-permute-flip.md, y-permute-crossmodel.md, z-permute-bands.md,
# z-permute-flip-land.md (the per-model release gate at this default).
REBLOCK_PERMUTE = True
# `TT_BIO_REBLOCK_PERMUTE` stays as an out-of-process override for A/B harnesses; the default is
# the constant above, so the release gate and any in-process import can see and set it.
_ENABLED = env_flag("TT_BIO_REBLOCK_PERMUTE", REBLOCK_PERMUTE)


def set_enabled(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global _ENABLED
    prev, _ENABLED = _ENABLED, bool(on)
    return prev


# --- the inverse move, permute(x, (0,2,3,1)) -------------------------------------------------------
#
# The trimul's channel loop moves the chunk to the batch axis for the per-channel contraction and
# then has to move it back. Forward is one kernel; back was two `ttnn.transpose` calls, and the pair
# reads and writes the tensor TWICE. Measured at 512 aa on qb2 card 0 (state/trimul-absolute-optimal
# §5): `transpose(1,2)` on [1,256,512,512] is 4.082 ms at 65.8 GB/s, 17.5 % of the measured combined
# roof and the single largest class in the module, because the destination tile it feeds spans 32
# channels while the source tile spans one, so a source tile scatters 32 rows into 32 destinations.
# Its partner `transpose(2,3)` is tile-local and already at 93 %. One pass over the tensor at the
# forward kernel's own measured rate is 1.51 ms against the pair's 4.849.
#
# Bit-exact by construction and verified with `torch.equal`: a permute is a pure index reordering,
# and the tile transpose the compute kernel applies is the same `transpose_wh` the stock op uses.


def _cache_key_back(x, out, device, reader_ct, writer_ct):
    g = device.compute_with_storage_grid_size()
    return (
        device.id(),
        int(x.shape[1]), int(x.shape[2]),
        str(x.dtype), str(x.layout),
        str(x.memory_config()), str(out.memory_config()),
        g.x, g.y, WALK,
        tuple(reader_ct), tuple(writer_ct),
    )


def _build_back(x, out, device, reader_ct, writer_ct):
    C, N = int(x.shape[1]), int(x.shape[2])
    Nt, Ct = N // TILE_H, C // TILE_W
    # A group is (it, jt, ct) and owns 32 output tiles. Keeping `ct` INSIDE the group index rather
    # than looping over it per group is what makes the work split even: at 512 aa with C=256 that is
    # 2048 groups over 110 cores (19 and 18 per core, 5 % imbalance) where Nt*Nt groups would be 256
    # over 110 (3 and 2, 33 %).
    num_groups = Nt * Nt * Ct

    plan = _split_plan(device, num_groups)
    assert plan is not None, f"no expressible work split for {num_groups} groups"
    _, _, (num_cores, core_grid, cg1, cg2, work1, work2) = plan

    tile_bytes = TILE_H * TILE_W * _elem()

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=_DTYPE, page_size=tile_bytes
        )
        return ttnn.CBDescriptor(
            total_size=depth * tile_bytes, core_ranges=core_grid, format_descriptors=[fmt]
        )

    # Both 32-tile CBs MUST have a depth that is a multiple of 32, or the ring wraps mid-group: the
    # reader would gather from a scratch window that is not contiguous and the writer would stream 32
    # tiles from an address range that runs off the end of the buffer. That failure is silent -- it
    # passes at N=128 and N=256, where a group is the whole buffer, and produces garbage at N=512.
    # 64 is the smallest multiple that also double-buffers.
    cbs = [cb(IN_CB, 2), cb(OUT_CB, GROUP_TILES * 2), cb(STAGE_CB, GROUP_TILES * 2)]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # `first` is this core's linear index, `block` the start of its contiguous run of groups. Both
    # counters are kept for every mode because `_walk` needs each of them; `placed` is the audit.
    first, placed, block = 0, 0, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    g0, gs, ghi, glo = _walk(WALK, first, block, per_core, num_cores)
                    reader_rt[cx][cy] = [g0, per_core, Nt, Ct, gs, ghi, glo]
                    compute_rt[cx][cy] = [per_core * GROUP_TILES]
                    writer_rt[cx][cy] = [g0, per_core, Nt, Ct, gs, ghi, glo]
                    first += 1
                    block += per_core
                    placed += per_core
    assert (first, placed) == (num_cores, num_groups), (first, placed, num_cores, num_groups)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR_BACK / "reader_reblock_permute_back.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=reader_ct, runtime_args=reader_rt,
        common_runtime_args=[0], config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR_BACK / "writer_reblock_permute_back.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=writer_ct, runtime_args=writer_rt,
        common_runtime_args=[0], config=ttnn.WriterConfigDescriptor(),
    )
    # The compute kernel is the forward direction's, unchanged: both moves end in one `transpose_wh`
    # per tile, and the CB indices are the same.
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_reblock_permute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=[IN_CB, OUT_CB], runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    # Same one-off probe as the forward direction: find out whether the binding hands back a
    # reference to the kernel held inside the cached descriptor or a copy of it.
    global ADDR_WRITE_MODE
    if ADDR_WRITE_MODE is None:
        probe = 0xABCD1234
        pd.kernels[0].common_runtime_args = [probe]
        ADDR_WRITE_MODE = "in_place" if list(pd.kernels[0].common_runtime_args) == [probe] \
            else "rebuild_pd"
        pd.kernels[0].common_runtime_args = [0]

    return {"pd": pd, "kernels": [reader, writer, compute], "cbs": cbs, "core_grid": core_grid}


def _prepare_back(x, out, device):
    reader_ct = [_elem(), STAGE_CB, IN_CB, TILE_H, TILE_W, FACE_H, FACE_W]
    reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [_elem(), OUT_CB, TILE_H, TILE_W]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    key = _cache_key_back(x, out, device, reader_ct, writer_ct)
    entry = _CACHE_BACK.get(key)
    if entry is None:
        entry = _CACHE_BACK[key] = _build_back(x, out, device, reader_ct, writer_ct)
    return entry


def reblock_permute_back(x, memory_config=None, device=None):
    """``ttnn.permute(x, (0, 2, 3, 1))`` for ``x`` of shape ``[1, C, N, N]`` bf16 TILE."""
    device = device or x.device()
    mc = memory_config or x.memory_config()
    C, N = int(x.shape[1]), int(x.shape[2])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, N, N, C]), _DTYPE, ttnn.TILE_LAYOUT, device, mc
    )
    entry = _prepare_back(x, out, device)
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
    STATS_BACK[0] += 1
    return ttnn.generic_op([x, out], pd)


# Whether `_channel_move_back` reaches for the kernel at all.
REBLOCK_PERMUTE_BACK = True
_ENABLED_BACK = os.environ.get(
    "TT_BIO_REBLOCK_PERMUTE_BACK", "1" if REBLOCK_PERMUTE_BACK else "0") == "1"


def set_enabled_back(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global _ENABLED_BACK
    prev, _ENABLED_BACK = _ENABLED_BACK, bool(on)
    return prev


def eligible_back(x, memory_config) -> bool:
    """The gate for the back direction.

    Deliberately narrower than the forward one. ``N`` must be a multiple of 32: the forward kernels
    carry an explicit ragged path because the trunk runs them at 298 aa, and the back direction can
    simply decline that shape and keep the two transposes, since the class it exists for is the DRAM
    path at 512 aa and above. The destination must be DRAM for the same reason -- that is the only
    place the two-transpose pair is expensive, and it is where `_triangle_mul_memory_config` puts the
    chunk from 352 aa up.
    """
    if not _ENABLED_BACK:
        return False
    shape = [int(d) for d in x.shape]
    if len(shape) != 4 or shape[0] != 1 or shape[2] != shape[3] or shape[1] % TILE_W:
        return _reject("back_shape", shape)
    C, N = shape[1], shape[2]
    if N % TILE_H:
        return _reject("back_ragged", shape)
    if x.dtype != _DTYPE or x.layout != ttnn.TILE_LAYOUT:
        return _reject("back_dtype_layout", shape)
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("back_sharded_out", shape)
    if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("back_sharded_in", shape)
    if memory_config.buffer_type != ttnn.BufferType.DRAM or N < 256:
        return _reject(f"back_window_{memory_config.buffer_type}", shape)
    # An L1 SOURCE is the channel loop's L1 path, where this kernel replaces a permute into L1
    # plus a clone to DRAM. It carries the forward kernel's L1 floor rather than the DRAM one:
    # MEASURED on the 11x10 grid, qb1 card 0, 1350 MHz sampled during, 9 interleaved reps against
    # an A/A floor of 1.0002-1.0015x -- 256 LOSES at 0.9918x / 0.9903x, 288 wins 1.0086x /
    # 1.0092x, 320 wins 1.0135x / 1.0172x (perf/trix_layout/back_onepass_qb1c0.json). 256's
    # chunks are 4.19 MB, small enough that the kernel's own dispatch outweighs the deleted pass.
    if x.memory_config().buffer_type == ttnn.BufferType.L1 and N < L1_N_MIN:
        return _reject("back_l1_src_narrow", shape)
    return True


# --- E6: the gate folded into the forward move ------------------------------------------------------
#
# The trimul's channel loop produces one wide projection [1, N, N, 4*Cg] and then spends three full
# DRAM passes taking it apart: `ttnn.chunk(_, 4, dim=-1)` and two `multiply_(p, g, [SIGMOID])`. At
# 512 aa those three ops measure 2.832 + 2 x 1.022 = 4.876 ms per call on qb2 card 1, at 95-99 % of
# the 399.2 GB/s combined roof (perf/trimul_f2/e6_prize.py), so there is no tuning left in them: the
# only way to make them cheaper is to stop doing them.
#
# The forward move runs at 47.5 % of that roof and its critical path is the WRITER RISC's gather,
# 2048 local transactions per 32-tile group against 32 DRAM writes. So it has room on both of the
# RISCs this work lands on: the reader gains a second DRAM stream and the compute kernel gains an
# SFPU sigmoid and an FPU multiply, and neither is the binding resource.
#
# Bit-exactness is not by construction here -- unlike the plain permute, this kernel does
# arithmetic. It is achieved by mirroring binary_ng's own kernel structure (see
# compute_reblock_permute_gated.cpp) and pinned by `torch.equal` against the two-op sequence on
# device, per shape, in perf/trimul_f2/e6_parity.py.

KERNEL_DIR_GATED = Path(__file__).resolve().parent / "kernels" / "reblock_permute_gated"

# Tiles per DST acquire in the gated compute kernel; TT_BIO_GATE_GRANULARITY=1 restores the
# per-tile acquire. Same three stages in the same order through the same two bf16 circular
# buffers, so every rounding point is untouched and this kernel's standing `torch.equal` claim
# against the two-op ttnn sequence still holds -- checked, not assumed, at every value on both
# architectures.
#
# 2, not the 4 that is fastest on Wormhole. The two architectures do not agree: WH reads
# 1.0420x at 2 and 1.0759x at 4, Blackhole reads 1.0149x at 2 and 1.0011x at 4, so 4 is a wash on
# the architecture the published cell is measured on. 2 is the value that wins on one and keeps
# most of the other (perf/b2z_levers/gategran_512_qb2c0_n40.json,
# perf/b2z_sdpa_floor/gategran_512_whglx_c3.json). Worth 0.014 s on a 512 aa Blackhole fold,
# which is under the A/A floor -- it ships because it is free and bit-exact, not for the fold.
# Capped at 4 because the multiply stage holds two DST slots a tile against 8 slots of a 16-bit
# DST; above that the kernel would corrupt.
GATE_GRANULARITY = max(1, min(4, int(os.environ.get("TT_BIO_GATE_GRANULARITY", "2"))))

P_CB, G_CB, SIG_CB, MUL_CB = 0, 1, 2, 3

_CACHE_GATED: dict = {}
STATS_GATED = [0, 0]


def _cache_key_gated(x, out, device, reader_ct, writer_ct):
    """As `_cache_key`, plus the slice width. The two slice OFFSETS are deliberately absent: they
    are common runtime args, so one descriptor serves both the `a` and the `b` call."""
    g = device.compute_with_storage_grid_size()
    return (
        device.id(),
        int(x.shape[1]), int(x.shape[3]), int(out.shape[1]), int(out.shape[2]),
        str(x.dtype), str(x.layout),
        str(x.memory_config()), str(out.memory_config()),
        g.x, g.y, WALK,
        tuple(reader_ct), tuple(writer_ct),
        # `_build_gated` bakes this into the compute kernel's compile-time args AND into four CB
        # depths, so it has to be in the key. Without it an A/B that flips the granularity gets the
        # FIRST arm's compiled program back for both legs and reads a 1.000x that means nothing.
        GATE_GRANULARITY,
    )


def _build_gated(x, out, device, reader_ct, writer_ct, fidelity, fp32_acc):
    # `x` may be a ROW BLOCK of the wide projection, [1, R, N, Cw] with R < N, while `out` is
    # always the full [1, slice_c, N, N] destination. Nt is therefore read off the destination and
    # Nrt off the source: the block is its own tensor and is addressed locally, while every
    # destination index is absolute via the `row_off` common arg. R == N is the whole-tensor move
    # and is byte-for-byte what it was.
    N = int(out.shape[2])
    Ctw = int(x.shape[3]) // TILE_W       # channel tiles of the wide input
    Ct = int(out.shape[1]) // TILE_W      # channel tiles of one slice
    Nt = (N + TILE_H - 1) // TILE_H
    Nrt = (int(x.shape[1]) + TILE_H - 1) // TILE_H
    # A group is (row-tile, col-tile, channel-tile). Keeping the channel tile INSIDE the group
    # index is what makes the split even on a row block: a 64-row block at 512 aa is 2*16*8 = 256
    # groups over 110 cores where (row-tile, col-tile) alone would be 32, i.e. 8 waves against the
    # whole-tensor move's 3. `_build_back` does the same and records the same arithmetic.
    num_groups = Nrt * Nt * Ct

    plan = _split_plan(device, num_groups)
    assert plan is not None, f"no expressible work split for {num_groups} groups"
    _, _, (num_cores, core_grid, cg1, cg2, work1, work2) = plan

    tile_bytes = TILE_H * TILE_W * _elem()

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=_DTYPE, page_size=tile_bytes
        )
        return ttnn.CBDescriptor(
            total_size=depth * tile_bytes, core_ranges=core_grid, format_descriptors=[fmt]
        )

    # c_16 keeps the 32-tile group multiple the writer's L1 window needs. The four working CBs are
    # double-buffered singles at the default granularity: the compute kernel consumes and produces
    # one tile at a time, and a deeper ring would only hold more of a stream the writer is already
    # the slow end of. That last clause is exactly what GATE_GRANULARITY tests -- if the writer is
    # the slow end then removing compute barriers buys nothing and the A/B reads 1.00x.
    cbs = [cb(P_CB, 2 * GATE_GRANULARITY), cb(G_CB, 2 * GATE_GRANULARITY),
           cb(SIG_CB, 2 * GATE_GRANULARITY), cb(MUL_CB, 2 * GATE_GRANULARITY),
           cb(OUT_CB, GROUP_TILES * 2), cb(STAGE_CB, 2)]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    # `first` is this core's linear index, `block` the start of its contiguous run of groups. Both
    # counters are kept for every mode because `_walk` needs each of them; `placed` is the audit.
    first, placed, block = 0, 0, 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    g0, gs, ghi, glo = _walk(WALK, first, block, per_core, num_cores)
                    reader_rt[cx][cy] = [g0, per_core, Nt, N, Ct, Ctw, gs, ghi, glo]
                    compute_rt[cx][cy] = [per_core * GROUP_TILES]
                    writer_rt[cx][cy] = [g0, per_core, Nt, N, Ct, gs, ghi, glo]
                    first += 1
                    block += per_core
                    placed += per_core
    assert (first, placed) == (num_cores, num_groups), (first, placed, num_cores, num_groups)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR_GATED / "reader_reblock_permute_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=reader_ct, runtime_args=reader_rt,
        common_runtime_args=[0, 0, 0, 0], config=ttnn.ReaderConfigDescriptor(),
    )
    # The writer is a fork of the ungated one: same gather, same staging, same DRAM write, but the
    # work unit carries the channel tile and the destination index carries the block's row offset.
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR_GATED / "writer_reblock_permute_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=writer_ct, runtime_args=writer_rt,
        common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR_GATED / "compute_reblock_permute_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[P_CB, G_CB, SIG_CB, MUL_CB, OUT_CB, int(GATE_SKIP_SIGMOID),
                           GATE_GRANULARITY],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc
        ),
    )
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    global ADDR_WRITE_MODE
    if ADDR_WRITE_MODE is None:
        probe = 0xABCD1234
        pd.kernels[0].common_runtime_args = [probe, 0, 0, 0]
        ADDR_WRITE_MODE = "in_place" if list(pd.kernels[0].common_runtime_args)[0] == probe \
            else "rebuild_pd"
        pd.kernels[0].common_runtime_args = [0, 0, 0, 0]

    return {"pd": pd, "kernels": [reader, writer, compute], "cbs": cbs, "core_grid": core_grid}


def _prepare_gated(x, out, device, fidelity, fp32_acc):
    reader_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [_elem(), OUT_CB, TILE_H, TILE_W, FACE_H, FACE_W, STAGE_CB]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    key = _cache_key_gated(x, out, device, reader_ct, writer_ct)
    entry = _CACHE_GATED.get(key)
    if entry is None:
        entry = _CACHE_GATED[key] = _build_gated(
            x, out, device, reader_ct, writer_ct, fidelity, fp32_acc)
    return entry


# The compute config the kernel has to run under to stay bit-exact against . Both
# values are load-bearing and neither is a default: see the kernel for the two mechanisms and
# perf/trimul_f2/e6_parity.py for the shapes they were pinned at. A change here is a silent
# precision change, not an error.
GATE_FIDELITY = ttnn.MathFidelity.HiFi4
GATE_FP32_ACC = False
# Diagnostic, never on in production: drops the activation so the multiply can be measured alone.
GATE_SKIP_SIGMOID = False


def reblock_permute_gated(xw, p_slice, g_slice, slice_c, memory_config=None, device=None,
                          out=None, row_off=0):
    """``permute(chunk(xw, 4, -1)[p_slice] * sigmoid(chunk(xw, 4, -1)[g_slice]), (0, 3, 1, 2))``.

    ``xw`` is ``[1, N, N, Cw]`` bf16 TILE and the result is ``[1, slice_c, N, N]``. The slice
    arguments are in CHANNELS, not tiles; the kernel takes tile offsets.

    Pass ``out`` and ``row_off`` to move ONE ROW BLOCK: ``xw`` is then ``[1, R, N, Cw]`` holding
    the rows starting at ``row_off``, and the block is written into the full ``out`` in place.
    That is what lets the projection feeding this move stay L1-resident -- the whole
    ``[1, N, N, Cw]`` projection is 512 MB at 512 aa against 160 MB of L1, and a 64-row block is
    67 MB. The result is identical to moving the whole tensor: the destination index is absolute
    and the blocks partition it.
    """
    device = device or xw.device()
    if out is None:
        mc = memory_config or xw.memory_config()
        N = int(xw.shape[1])
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, slice_c, N, N]), _DTYPE, ttnn.TILE_LAYOUT, device, mc
        )
    assert row_off % TILE_H == 0, f"row_off {row_off} is not a tile boundary"
    entry = _prepare_gated(xw, out, device, GATE_FIDELITY, GATE_FP32_ACC)
    src, dst = xw.buffer_address(), out.buffer_address()
    common_r = [src, p_slice // TILE_W, g_slice // TILE_W, row_off // TILE_H]
    common_w = [dst, row_off // TILE_H]
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
    STATS_GATED[0] += 1
    return ttnn.generic_op([xw, out], pd)


# Master switch for folding the trimul's chunk and its two sigmoid gates into the forward move.
# ON, but every TriangleMultiplication still has to opt in (`gated_move=`). The recorded reason was
# that the fused pair is a measured LOSS on boltz2 (+0.373 s/fold at 512 aa, only 64 of 560 of its
# moves eligible) against 1.5046x on opendde's, `torch.equal` at both of its slice widths
# (perf/odde512/screen3.json). The loss was COVERAGE, not the kernel: `eligible_gated`'s caller
# required `mask_u is None` and boltz2 always passes a pair mask, so none of those 64 moves can
# have been in the pairformer. With the mask moved past the channel move the same kernel is
# 1.2981x / 1.3329x per trimul at 512 aa and 1.0555x on the fold, bit-exact
# (perf/b2x_trimul/). The model decides; this only says whether the kernel exists at all.
REBLOCK_PERMUTE_GATED = True
_ENABLED_GATED = os.environ.get(
    "TT_BIO_REBLOCK_PERMUTE_GATED", "1" if REBLOCK_PERMUTE_GATED else "0") == "1"


def set_enabled_gated(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global _ENABLED_GATED
    prev, _ENABLED_GATED = _ENABLED_GATED, bool(on)
    return prev


def eligible_gated(xw, slice_c, memory_config) -> bool:
    """The gate for the fused path, deliberately the forward gate's window plus what fusion adds.

    Everything `eligible` checks applies unchanged: the same reader, the same writer, the same work
    split. On top of it the wide input must actually be the four-way fused projection, and the
    slice width must be a whole number of tiles, because the reader addresses slices in tile units.
    """
    if not _ENABLED_GATED:
        return False
    shape = [int(d) for d in xw.shape]
    # `xw` is either the whole [1, N, N, 4*slice_c] projection or ONE ROW BLOCK of it,
    # [1, R, N, 4*slice_c] with R a whole number of tiles, which is the mode `_build_gated`'s
    # `out`/`row_off` arguments exist for. N is the DESTINATION width in both cases, so it comes
    # off axis 2 and never off axis 1. The whole-tensor case keeps its exact old window: any N,
    # including the 298 that production runs and that is not a tile multiple.
    if len(shape) != 4 or shape[0] != 1:
        return _reject("gated_shape", shape)
    if shape[1] != shape[2] and not (shape[1] < shape[2] and shape[1] % TILE_H == 0):
        return _reject("gated_rowblock", shape)
    if shape[3] != 4 * slice_c or slice_c % TILE_W:
        return _reject("gated_slice", shape)
    N = shape[2]
    if xw.dtype != _DTYPE or xw.layout != ttnn.TILE_LAYOUT:
        return _reject("gated_dtype_layout", shape)
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("gated_sharded_out", shape)
    if xw.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("gated_sharded_in", shape)
    bt = memory_config.buffer_type
    if not ((bt == ttnn.BufferType.DRAM and N >= 256)
            or (bt == ttnn.BufferType.L1 and L1_N_MIN <= N <= L1_N_MAX)):
        return _reject(f"gated_window_{bt}", shape)
    return True
