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
KERNEL_DIR_BACK = Path(__file__).resolve().parent / "kernels" / "reblock_permute_back"

TILE_H = TILE_W = 32
FACE_H = FACE_W = 16
GROUP_TILES = 32
IN_CB, OUT_CB, STAGE_CB = 0, 16, 24

# How the cached descriptor gets its two per-call addresses. Set on first use; kept as module state
# only so a probe can report which path the wheel took.
ADDR_WRITE_MODE = None

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


# --- the wheel's own work split has holes, and they move with the part shape -----------------------
#
# `ttnn.split_work_to_cores` raises `TT_FATAL @ work_split.cpp:305: remaining == 0` for some unit
# counts. Measured on qb1 at ttnn 0.67.4 over 4000 unit counts on ten grids (13x10, 11x10, 9x13,
# 13x13, 13x9, 12x10, 12x7, 8x10, 7x10, 6x6, 5x11), zero mismatches against this rule: it throws
# exactly when `units > cores` and `units % cores` is a NON-ZERO MULTIPLE OF THE GRID HEIGHT. The
# split's two core groups then have sizes that are both multiples of the height, and the utility
# cannot express the second one as core ranges from where the first one ends.
#
# On a 13x10 grid that is Nt = 20, 30, 40, 50, 60, i.e. N in [609,640], [929,960], [1249,1280],
# [1569,1600], [1889,1920]. On a 7x10 part it also catches Nt = 10, which is the Protenix trunk's
# own tile count. So a hardcoded Nt exclusion list is right on this card and wrong on the next one,
# silently.
#
# The rule is therefore used only to ORDER the search. The utility is always the authority: every
# candidate rectangle is handed to it, and a shape it cannot split at all is refused by `eligible`
# and reaches `ttnn.permute` instead of a TT_FATAL.

_SPLIT_CACHE: dict = {}


def _split_hole(cores, height, units):
    """The measured rule above. A search heuristic, never the final word."""
    r = units % cores
    return units > cores and r != 0 and r % height == 0


def _split_plan(device, units):
    """The work split for ``units`` groups, or ``None`` if no rectangle of cores can carry it.

    The full grid is tried first, so every shape that works today is split exactly as it is today
    and production at N=298 is untouched. Only when the wheel throws does this look for the largest
    rectangular sub-grid it will accept -- 117 of 130 cores at Nt=20 on qb1's 13x10, 90.0 % -- which
    keeps the kernel on those bands instead of handing them back to an op it beats by 1.9x on DRAM.

    Cached per ``(device, grid, units)``. The cost matters: a split that works costs 0.61 us on qb1,
    a throwing one costs 357 us, and ``_channel_move`` runs 4352 times in a 298 aa fold. Probing per
    call would replace a crash with a slowdown.
    """
    g = device.compute_with_storage_grid_size()
    key = (device.id(), g.x, g.y, units)
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]
    candidates = [(g.x, g.y)] + sorted(
        ((sx, sy) for sy in range(1, g.y + 1) for sx in range(1, g.x + 1)
         if (sx, sy) != (g.x, g.y) and not _split_hole(sx * sy, sy, units)),
        key=lambda s: -s[0] * s[1],
    )
    plan = None
    for sx, sy in candidates:
        cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(sx - 1, sy - 1))]
        )
        try:
            plan = (sx, sy, ttnn.split_work_to_cores(cores, units))
        except Exception:                                              # noqa: BLE001 -- wheel TT_FATAL
            continue
        break
    _SPLIT_CACHE[key] = plan
    return plan


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

    plan = _split_plan(device, num_groups)
    # `eligible` has already refused any shape with no plan, so this cannot fire from the production
    # path. It stays as an assertion because a direct caller of `reblock_permute` bypasses the gate.
    assert plan is not None, f"no expressible work split for {num_groups} groups"
    _, _, (_, core_grid, cg1, cg2, work1, work2) = plan

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


# The L1 leg's window edges, named so a fold-level A/B can move one of them in-process without
# editing the gate. See eligible()'s docstring for what measured them.
L1_N_MIN = 288
L1_N_MAX = 352


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
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        return _reject("dtype_layout", shape)
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_out", shape)
    if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("sharded_in", shape)
    bt = memory_config.buffer_type
    if not ((bt == ttnn.BufferType.DRAM and N >= 256)
            or (bt == ttnn.BufferType.L1 and L1_N_MIN <= N <= L1_N_MAX)):
        return _reject(f"window_{bt}", shape)
    # Last, because it is the only clause that touches the device, and cached, so a fold pays it once
    # per shape. `_build` requests a work split for Nt*Nt groups and the wheel's utility throws on
    # some of them (see `_split_plan`); a shape it cannot split has to reach `ttnn.permute`, not a
    # TT_FATAL.
    if _split_plan(x.device(), ((N + TILE_H - 1) // TILE_H) ** 2) is None:
        return _reject("work_split", shape)
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
_ENABLED = os.environ.get("TT_BIO_REBLOCK_PERMUTE", "1" if REBLOCK_PERMUTE else "0") == "1"


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
        g.x, g.y,
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
    _, _, (_, core_grid, cg1, cg2, work1, work2) = plan

    tile_bytes = TILE_H * TILE_W * 2  # bf16

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes
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
    start = 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    reader_rt[cx][cy] = [start, per_core, Nt, Ct]
                    compute_rt[cx][cy] = [per_core * GROUP_TILES]
                    writer_rt[cx][cy] = [start, per_core, Nt, Ct]
                    start += per_core
    assert start == num_groups, (start, num_groups)

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
    reader_ct = [2, STAGE_CB, IN_CB, TILE_H, TILE_W, FACE_H, FACE_W]
    reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [2, OUT_CB, TILE_H, TILE_W]
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
        ttnn.Shape([1, N, N, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc
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
REBLOCK_PERMUTE_BACK = False  # WIP: the kernel is NOT correct yet, see state doc E2
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
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        return _reject("back_dtype_layout", shape)
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("back_sharded_out", shape)
    if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return _reject("back_sharded_in", shape)
    if memory_config.buffer_type != ttnn.BufferType.DRAM or N < 256:
        return _reject(f"back_window_{memory_config.buffer_type}", shape)
    if _split_plan(x.device(), (N // TILE_H) ** 2 * (C // TILE_W)) is None:
        return _reject("back_work_split", shape)
    return True
