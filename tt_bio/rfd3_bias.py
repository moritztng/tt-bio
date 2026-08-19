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
# measured clone roof is 385 GB/s, so 0.47 ms is the floor. Swept at the production shape on qb1
# card 1 (perf/p36/slot_sweep.json, slot_sweep_2risc.json): on one RISC 4/8/16/32 slots read
# 1.987 / 1.652 / 1.655 / 1.693 ms, on two 1.081 / 0.932 / 0.972. It saturates at 8 on both, which
# is what says the op is bound by the kernel's own L1 traffic rather than by the write -- and why
# the second RISC, not more slots, is what took it from 1.65 to 0.93 ms.
OUT_SLOTS = int(os.environ.get("RFD3_SPARSE_BIAS_SLOTS", "8"))

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
    B, H, L, K = (int(bias.shape[0]), int(bias.shape[1]), int(bias.shape[2]),
                  int(bias.shape[3]))
    N = int(out.shape[3])
    It = (L + TILE_H - 1) // TILE_H
    Jt = N // TILE_W
    Kt = (K + TILE_W - 1) // TILE_W

    g = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
    )
    cores = [(cx, cy) for cx in range(g.x) for cy in range(g.y)]
    counts = _even_split(B * H * It, cores)

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
    assert start == B * H * It, (start, B * H * It)

    tail = [It, Jt, Kt, K, L, _fill_bits(-1e4), OUT_SLOTS, H * It] + list(acc_args)
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
    B, H, L, K = (int(pair_bias.shape[0]), int(pair_bias.shape[1]), int(pair_bias.shape[2]),
                  int(pair_bias.shape[3]))
    N = ((L + TILE_W - 1) // TILE_W) * TILE_W
    if out is None:
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, H, L, N]), ttnn.float32, ttnn.TILE_LAYOUT, device, mc
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


def eligible_shape(batch, n_heads, length, n_keys, dtype) -> bool:
    """Whether the kernel may serve a step, decided from the shape alone.

    Called once per step from ``_sparse_qk_inputs``, before either tensor exists, because the
    answer decides how the index is uploaded and whether the dense template is built at all.

    ``batch`` is not a constraint: a group is a (design, head, tile-row) band and the group index is
    that triple flattened, so multiplicity batching costs the kernels one divide. It was refused
    until 2026-08-13 and that refusal cost every multi-design run the whole win, since
    ``--batch_size`` defaults to 8.
    """
    shape = [batch, n_heads, length, n_keys]
    if not _ENABLED:
        return False
    if dtype != ttnn.bfloat16:
        return _reject("dtype", shape)
    if n_keys % TILE_W or n_keys < TILE_W:
        return _reject("n_keys", shape)
    return True


RFD3_SPARSE_BIAS = True    # default-on: torch.equal at every shape, byte-identical designs
_ENABLED = os.environ.get(
    "RFD3_SPARSE_BIAS", "1" if RFD3_SPARSE_BIAS else "0"
) == "1"


def set_enabled(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global _ENABLED
    prev, _ENABLED = _ENABLED, bool(on)
    return prev


# --- L6b: the same bias, fused into the score path -------------------------------------------
# cb ids for the fused op. The three scratch buffers are reader-local (it only ever takes their
# write pointer), so their depth is a size and not a pipeline knob.
# The pristine all-fill page is how a used slot is restored: one 4 KB local L1->L1 copy, measured
# 1.686 -> 1.393 ms/call against replaying the tile's poke walk backwards (state §28), bit-exact.
F_SCORES_CB, F_BIAS_CB, F_IDX_CB, F_PB_CB, F_TPL_CB, F_OUT_CB = 0, 1, 2, 3, 4, 16
# cb_bias / cb_out depth. Both must be powers of two: the reader and the writer index the ring
# with a mask, because they track their own slot rather than trusting a CB pointer across a wrap.
F_BIAS_SLOTS = int(os.environ.get("RFD3_FUSED_BIAS_SLOTS", "8"))
F_OUT_SLOTS = int(os.environ.get("RFD3_FUSED_OUT_SLOTS", "8"))
F_SCORES_SLOTS = int(os.environ.get("RFD3_FUSED_SCORES_SLOTS", "4"))
# Writes coalesced per barrier on the writer RISC.
F_WINDOW = int(os.environ.get("RFD3_FUSED_WINDOW", "4"))

_FCACHE: dict = {}
FSTATS = [0, 0]


def _scale_bits(value: float) -> int:
    """fp32 bit pattern of ``value``, which is how MUL_UNARY_SFPU carries its scalar.

    ttnn's own emitter writes ``mul_unary_tile({}, {:#x}u)``, so the scalar never reaches the
    device as a float and the fp64 -> fp32 rounding happens here, exactly where ttnn does it.
    """
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _fbuild(scores, pair_bias, idx_rm, out, device, scale, acc):
    B, H, L, K = (int(pair_bias.shape[0]), int(pair_bias.shape[1]), int(pair_bias.shape[2]),
                  int(pair_bias.shape[3]))
    N = int(out.shape[3])
    It = (L + TILE_H - 1) // TILE_H
    Jt = N // TILE_W
    Kt = (K + TILE_W - 1) // TILE_W

    g = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
    )
    cores = [(cx, cy) for cx in range(g.x) for cy in range(g.y)]
    counts = _even_split(B * H * It, cores)

    def cb(idx_, fmt, page_size, depth):
        return ttnn.CBDescriptor(
            total_size=depth * page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx_, data_format=fmt, page_size=page_size)
            ],
        )

    cbs = [
        cb(F_SCORES_CB, ttnn.bfloat16, TILE_H * TILE_W * 2, F_SCORES_SLOTS),
        cb(F_BIAS_CB, ttnn.float32, TILE_H * TILE_W * 4, F_BIAS_SLOTS),
        cb(F_OUT_CB, ttnn.float32, TILE_H * TILE_W * 4, F_OUT_SLOTS),
        cb(F_IDX_CB, ttnn.uint32, K * 4, TILE_H),
        cb(F_PB_CB, ttnn.bfloat16, TILE_H * TILE_W * 2, Kt),
        cb(F_TPL_CB, ttnn.float32, TILE_H * TILE_W * 4, 1),
    ]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for (cx, cy), n in zip(cores, counts):
        reader_rt[cx][cy] = [start, n]
        writer_rt[cx][cy] = [start, n]
        compute_rt[cx][cy] = [n * Jt]
        start += n
    assert start == B * H * It, (start, B * H * It)

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_fused_scores.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[F_SCORES_CB, F_BIAS_CB, F_IDX_CB, F_PB_CB, F_TPL_CB,
                           It, Jt, Kt, K, L, _fill_bits(-1e4), F_BIAS_SLOTS,
                           F_NOPOKE, H * It] + list(acc),
        runtime_args=reader_rt, common_runtime_args=[0, 0, 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_fused_scores.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[F_OUT_CB, It, Jt, F_WINDOW, F_OUT_SLOTS]
        + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        runtime_args=writer_rt, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_fused_scores.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[F_SCORES_CB, F_BIAS_CB, F_OUT_CB, _scale_bits(scale), F_DIAG_COPY],
        runtime_args=compute_rt,
        # fp32_dest_acc_en is not a tuning knob here, it is the parity condition: ttnn packs the
        # scaled tile to an fp32 intermediate and this kernel keeps it in DST, and those two agree
        # only while DST is fp32. See compute_fused_scores.cpp.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, dst_full_sync_en=False,
        ),
    )
    kernels = [reader, writer, compute]
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


# Diagnostics, never set in production: NOPOKE drops the poke walk and the repair replay,
# DIAG_COPY drops both SFPU passes. Together they decompose the op's 1.68 ms/call into the
# pipeline (0.864) and the per-element placement (0.819), which is what refuted L6c -- the
# two-RISC split of the placement work, 1.682 against 1.683 (state/rfd3-host-half.md §25).
F_NOPOKE = int(os.environ.get("RFD3_FUSED_NOPOKE", "0"))
F_DIAG_COPY = int(os.environ.get("RFD3_FUSED_DIAG_COPY", "0"))


def fused_scores_bias_fp32(scores, pair_bias, idx_rm, scale, out=None, memory_config=None):
    """``add(typecast(scores, fp32), sparse_bias, a_activations=[MUL_UNARY_SFPU(scale)])``, fused.

    Five ops in one pass: the ``-1e4`` template, the ``scatter`` of the neighbour pair bias, its
    widen, the scores' widen, and the scaled add. The dense fp32 bias is never materialised in
    DRAM -- it is built one tile at a time in L1 and consumed there -- so the traffic is the
    90.3 MB of scores read plus the 180.6 MB of output written and nothing else.

    ``scores`` is ``[1, H, L, N]`` bf16 TILE with ``N = align_tile(L)``, ``pair_bias`` is
    ``[1, H, L, K]`` bf16 TILE, ``idx_rm`` is ``[1, 1, L, K]`` uint32 ROW_MAJOR sorted ascending
    along the last axis. The output is ``[1, H, L, N]`` fp32 TILE.
    """
    device = scores.device()
    B, H, L, K = (int(pair_bias.shape[0]), int(pair_bias.shape[1]), int(pair_bias.shape[2]),
                  int(pair_bias.shape[3]))
    N = int(scores.shape[3])
    if out is None:
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, H, L, N]), ttnn.float32, ttnn.TILE_LAYOUT, device, mc
        )

    acc = (
        list(ttnn.TensorAccessorArgs(scores).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(pair_bias).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(idx_rm).get_compile_time_args())
    )
    key = _cache_key(pair_bias, idx_rm, out, device, tuple(acc)) + (
        tuple(int(d) for d in scores.shape), str(scores.dtype), _scale_bits(scale),
        F_NOPOKE, F_DIAG_COPY,
    )
    entry = _FCACHE.get(key)
    if entry is None:
        entry = _FCACHE[key] = _fbuild(scores, pair_bias, idx_rm, out, device, scale, acc)

    per_kernel = [[scores.buffer_address(), pair_bias.buffer_address(),
                   idx_rm.buffer_address()], [out.buffer_address()]]
    if ADDR_WRITE_MODE == "in_place":
        pd = entry["pd"]
        for k, a in zip(pd.kernels, per_kernel):
            k.common_runtime_args = a
    else:
        for k, a in zip(entry["kernels"], per_kernel):
            k.common_runtime_args = a
        pd = entry["pd"] = ttnn.ProgramDescriptor(
            kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"]
        )
    FSTATS[0] += 1
    return ttnn.generic_op([scores, pair_bias, idx_rm, out], pd)


def fused_enabled() -> bool:
    return _FUSED_ENABLED


def stats_line() -> str:
    """What the two kernels actually did in this process, for the shape sweep.

    A shape sweep cannot be read off wall times or even off the output: both kernels gate
    themselves and fall back silently, which is correct behaviour and indistinguishable from
    "served" in the CIF. This is the only way to tell the two apart, so it is printed on request
    rather than reconstructed.
    """
    return (f"[rfd3_bias] sparse_bias served={STATS[0]} declined={STATS[1]} "
            f"fused served={FSTATS[0]} rejects={ {k: v for k, v in REJECTS.items()} }")


if os.environ.get("RFD3_BIAS_STATS") == "1":
    import atexit

    atexit.register(lambda: print(stats_line(), flush=True))


def set_fused_enabled(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global _FUSED_ENABLED
    prev, _FUSED_ENABLED = _FUSED_ENABLED, bool(on)
    return prev


RFD3_FUSED_SCORES = True    # default-on: torch.equal on the result and on the softmax
_FUSED_ENABLED = os.environ.get(
    "RFD3_FUSED_SCORES", "1" if RFD3_FUSED_SCORES else "0"
) == "1"


# --- L2: the same fusion for the DENSE score path --------------------------------------------
# The DiT (`LocalTokenTransformer`) takes `RFD3AtomBlock`'s dense branch, where the bias is a
# full [1, H, I, n_key] tensor rather than a gathered [1, H, L, K] one, so the sparse kernel above
# does not apply. What is left to fuse there is the tail: two typecasts and the scaled add.
#
# Measured at the page fixture's shape, [1, 16, 685, 704], card 1 under benchlock, warm, n=5
# (perf/p71/bias_prep_arms.json):
#
#   typecast + typecast + scaled add            0.4986 ms/call   17.948 ms/step over 36 calls
#   ttnn.add(bf16, bf16, dtype=fp32, act=scale) 0.1857 ms/call    6.685 ms/step
#
# The second is the traffic this kernel moves -- r 30.9, w 30.9 MB -- and it is what says the
# fusion is worth ~11 ms/step. ttnn's own folded form is NOT usable, though: it is 2.55 maxabs
# away from the three-op chain, because the activation pass packs the scaled operand back into an
# intermediate CB at the INPUT dtype and rounds it to bf16. See the note at model.py's dense
# branch. This kernel keeps the scaled tile in an fp32 DST instead, which is exactly what the
# three-op chain's fp32 intermediate does, so it is bit-exact rather than close.
#
# Two ops that the DiT chain also runs were screened and left alone. The (0, 3, 1, 2) permute is
# 13.969 ms/step and cannot be absorbed: in the pre-permute [1, I, J, H] layout a tile is
# [32 j x 32 h], so one output tile needs column h of 32 separate pages and assembling it is a
# per-element strided gather of 7.7 M elements per call, which the sparse kernel's own measured
# poke rate prices at ~2.1 ms/call -- worse than the whole shipped chain. Batching the permute
# across all 18 blocks' slots is worse still: 576-wide measured 80.0 GB/s against 148.2 for
# 16-wide, so one permute of 1059.6 MB costs 12.94 ms where 18 of 58.9 MB cost 6.98.
D_SCORES_CB, D_BIAS_CB, D_OUT_CB = 0, 1, 16
# Ring depth and pages per barrier. SLOTS must be a power of two (both kernels mask with it) and
# WINDOW must not exceed it. The whole footprint is 8 x (2 + 2 + 4) KB = 64 KB of L1.
D_SLOTS = int(os.environ.get("RFD3_DENSE_SLOTS", "8"))
D_WINDOW = int(os.environ.get("RFD3_DENSE_WINDOW", "4"))

_DCACHE: dict = {}
DSTATS = [0, 0]


def _dbuild(scores, bias, out, device, scale, acc):
    """Descriptor for the dense fusion. Cached: everything here but the three addresses is a pure
    function of (padded shape, dtypes, grid, scale)."""
    pages = 1
    for d in out.padded_shape:
        pages *= int(d)
    pages //= TILE_H * TILE_W

    g = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
    )
    cores = [(cx, cy) for cx in range(g.x) for cy in range(g.y)]
    counts = _even_split(pages, cores)

    def cb(idx_, fmt, page_size, depth):
        return ttnn.CBDescriptor(
            total_size=depth * page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx_, data_format=fmt, page_size=page_size)
            ],
        )

    cbs = [
        cb(D_SCORES_CB, ttnn.bfloat16, TILE_H * TILE_W * 2, D_SLOTS),
        cb(D_BIAS_CB, ttnn.bfloat16, TILE_H * TILE_W * 2, D_SLOTS),
        cb(D_OUT_CB, ttnn.float32, TILE_H * TILE_W * 4, D_SLOTS),
    ]

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for (cx, cy), n in zip(cores, counts):
        reader_rt[cx][cy] = [start, n]
        writer_rt[cx][cy] = [start, n]
        compute_rt[cx][cy] = [n]
        start += n
    assert start == pages, (start, pages)

    n_scores_acc = len(ttnn.TensorAccessorArgs(scores).get_compile_time_args())
    n_bias_acc = len(ttnn.TensorAccessorArgs(bias).get_compile_time_args())

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_dense_scores.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[D_SCORES_CB, D_BIAS_CB, D_SLOTS, D_WINDOW]
        + list(acc[: n_scores_acc + n_bias_acc]),
        runtime_args=reader_rt, common_runtime_args=[0, 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_dense_scores.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[D_OUT_CB, D_SLOTS, D_WINDOW]
        + list(acc[n_scores_acc + n_bias_acc:]),
        runtime_args=writer_rt, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_fused_scores.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[D_SCORES_CB, D_BIAS_CB, D_OUT_CB, _scale_bits(scale), 0],
        runtime_args=compute_rt,
        # fp32_dest_acc_en is the parity condition, not a tuning knob: with a 16-bit DST the
        # scaled scores would be rounded to bf16 here where the three-op chain keeps them fp32.
        # See compute_fused_scores.cpp.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, dst_full_sync_en=False,
        ),
    )
    kernels = [reader, writer, compute]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    # Same probe as _build/_fbuild: whether a descriptor accepts an address rewrite in place, or
    # has to be rebuilt per call. It is a property of the wheel, so the first builder to run wins.
    global ADDR_WRITE_MODE
    if ADDR_WRITE_MODE is None:
        probe = [0xABCD1234, 0x1234ABCD]
        pd.kernels[0].common_runtime_args = probe
        ADDR_WRITE_MODE = (
            "in_place" if list(pd.kernels[0].common_runtime_args) == probe else "rebuild_pd"
        )
        pd.kernels[0].common_runtime_args = [0, 0]
    return {"pd": pd, "kernels": kernels, "cbs": cbs}


def dense_fused_scores_bias_fp32(scores, bias, scale, out=None, memory_config=None):
    """``add(typecast(scores, fp32), typecast(bias, fp32), a_activations=[MUL_UNARY_SFPU(scale)])``.

    ``scores`` and ``bias`` are the same bf16 TILE shape, ``[1, H, I, n_key]`` with ``n_key`` a tile
    multiple, and the output is that shape in fp32. Both inputs and the output share a tile-page
    layout, which is the whole reason the kernel needs no band geometry.
    """
    device = scores.device()
    if out is None:
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([int(d) for d in scores.shape]), ttnn.float32, ttnn.TILE_LAYOUT, device, mc
        )

    acc = (
        list(ttnn.TensorAccessorArgs(scores).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(bias).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    )
    key = _cache_key(scores, bias, out, device, tuple(acc)) + (
        tuple(int(d) for d in out.padded_shape), _scale_bits(scale), D_SLOTS, D_WINDOW,
    )
    entry = _DCACHE.get(key)
    if entry is None:
        entry = _DCACHE[key] = _dbuild(scores, bias, out, device, scale, acc)

    per_kernel = [[scores.buffer_address(), bias.buffer_address()], [out.buffer_address()]]
    if ADDR_WRITE_MODE == "in_place":
        pd = entry["pd"]
        for k, a in zip(pd.kernels, per_kernel):
            k.common_runtime_args = a
    else:
        for k, a in zip(entry["kernels"], per_kernel):
            k.common_runtime_args = a
        pd = entry["pd"] = ttnn.ProgramDescriptor(
            kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"]
        )
    DSTATS[0] += 1
    return ttnn.generic_op([scores, bias, out], pd)


def dense_eligible(scores, bias) -> bool:
    """Whether the dense fusion may serve this call, from the tensors alone."""
    if not _DENSE_ENABLED:
        return False
    if scores.dtype != ttnn.bfloat16 or bias.dtype != ttnn.bfloat16:
        DSTATS[1] += 1
        return False
    if tuple(int(d) for d in scores.padded_shape) != tuple(int(d) for d in bias.padded_shape):
        DSTATS[1] += 1
        return False
    if int(scores.shape[-1]) % TILE_W:
        DSTATS[1] += 1
        return False
    return True


RFD3_DENSE_BIAS_FUSED = True   # default-on: torch.equal at the production shape, digest unchanged
_DENSE_ENABLED = os.environ.get(
    "RFD3_DENSE_BIAS_FUSED", "1" if RFD3_DENSE_BIAS_FUSED else "0"
) == "1"


def set_dense_enabled(on: bool) -> bool:
    """A/B switch for the fold harness. Returns the previous state."""
    global _DENSE_ENABLED
    prev, _DENSE_ENABLED = _DENSE_ENABLED, bool(on)
    return prev
