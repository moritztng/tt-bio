#!/usr/bin/env python3
"""S0, the kill gate for the whole fused-kernel route: re-drive ``ttnn.experimental.minimal_matmul``
through ``ttnn.generic_op``, using the wheel's own three kernel sources unmodified.

This is a transcription of ``minimal_matmul_program_factory.cpp`` at the ``v0.68.0`` tag (the wheel's
kernel sources are byte-identical to that tag, checked in the planning pass) into a Python
``ttnn.ProgramDescriptor``. No kernel edit, no design change, no tt-metal build. If the transcription
reproduces the native op's time and output, every later step is a dataflow edit on top of it; if it
does not, the route is dead and the task is a NO-GO.

PREDICTION, WRITTEN BEFORE THE RUN (state/triatt-fused-kernel-final.md 5):

    2.13-2.35 ms against the 2.127 ms MEASURED native qkv arm, and ``torch.equal`` on the output.
    > 2.40 ms, or not torch.equal: the whole route is dead.

Only the fixed case the fold issues is covered: bf16 in / bf16 out, interleaved DRAM, no bias, no
fused activation, no ternary, no all-gather fusion, N_chunks = 1.
"""

from __future__ import annotations

import ttnn

# tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp
INVALID, VALID = 0, 1
TILE_HW = 32

# Blackhole takes the default branch of detail::preferred_noc_for_dram_{read,write}
# (tt_metal/api/tt-metalium/kernel_types.hpp:126-138).
NOC_FOR_DRAM_READ = ttnn.NOC.NOC_0
NOC_FOR_DRAM_WRITE = ttnn.NOC.NOC_1

#: Bytes one TILE_HW x TILE_HW tile occupies, per dtype. sdpa_generic and softmax_generic
#: size their CBs from this same table -- see `tile_bytes`.
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}

#: Pages per block in the three pipeline circular buffers, as a multiple of one block. 2 is the
#: C++ factory's value and the shipped default: the reader can stage one block ahead of the block
#: the math thread is consuming. Raising it lets the reader run further ahead, at L1 cost.
#: `perf/b2z2_cb_depth/` sweeps it -- a program descriptor is cached by shape and not by this, so
#: clear `_CACHE` after changing it.
CB_DEPTH = 2

#: Bytes the three pipeline CBs plus the accumulator may occupy on one core. tt-metal's own ceiling
#: is 1499136 B on both Wormhole and Blackhole, and it is a *hard* throw at program validation, not
#: a fallback: at the 512 aa trimul in-projection the shipped depth-2 CBs are already 1030336 B, so
#: a blanket depth 3 does not fit and takes the whole fold down. The margin below the ceiling is for
#: the L1 tensors a call may have live at the same time.
CB_L1_BUDGET = 1_300_000

#: Divisor applied to the caller's `K_block` before the program is built, so one K block becomes
#: `K_SPLIT` of them. 1 is the shipped value. Every block config the fold uses today contracts in a
#: SINGLE K block, which means the math thread's `cb_wait_front` on `in0`/`in1` has nothing to
#: overlap with: there is no next block for the reader to be fetching. Splitting K is the only way
#: to give the ring something to prefetch, and it is NOT free -- the compute kernel accumulates
#: across K blocks through the packer with L1 accumulation, so each extra block is an extra pack
#: pass over the output block, and the contraction folds in a different order (not bit-exact).
K_SPLIT = 1

#: Overrides for the caller's block geometry, as `{"M": t, "N": t, "sh": t, "sw": t}`. Empty ships.
#: `M_block`/`N_block` set how many tiles the math thread waits on per `cb_wait_front`, and the
#: subblock pair sets how many of them go into DST before a pack. They are the other half of
#: "how far ahead can anything run" and they are not the same knob as ring depth: depth decides
#: whether a *next* block can be staged, geometry decides how big the block being waited on is.
BLOCK_OVERRIDE: dict = {}

#: What depth each built program actually got, keyed by its block config. A clamped call and a
#: deepened one time identically to a reader that only looks at `CB_DEPTH`, so a sweep has to be
#: able to say which of its calls the ceiling refused.
CB_DEPTH_STATS: dict = {}

#: Per built program, the tile arithmetic the block geometry decides: how many tiles each core
#: pulls in over the whole call. `in1` is re-read once per M block, so the geometry is a data-reuse
#: knob and not only a scheduling one -- this is what lets a sweep price it in tiles rather than in
#: milliseconds. Keyed the same way as `CB_DEPTH_STATS`.
TILE_TRAFFIC_STATS: dict = {}


def _fit_depth(want, per_depth_bytes, fixed_bytes, key):
    """The deepest ring <= `want` that fits `CB_L1_BUDGET`, never below the shipped 2."""
    d = want
    while d > 2 and d * per_depth_bytes + fixed_bytes > CB_L1_BUDGET:
        d -= 1
    CB_DEPTH_STATS[key] = d
    return d


def tile_bytes(dtype):
    """Bytes one tile occupies in `dtype`, for a CB page size or a runtime arg."""
    try:
        return _TILE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"no tile size for {dtype}: these transcriptions cover the call the "
                         "fold issues, which is bf16 and fp32 only") from None

_CACHE: dict = {}


def ttnn_cpp_root():
    """The directory holding ttnn's C++ ``cpp/`` tree, whichever way ttnn was installed.

    Kernel sources are pointed at in place so their sibling includes resolve, which means
    this path has to be right. The pip wheel vendors them at ``<pkg>/ttnn/cpp``; a source
    build imports ttnn from ``<checkout>/ttnn/ttnn`` and keeps them at ``<checkout>/ttnn/cpp``.
    Assuming the wheel layout produced ``<checkout>/ttnn/ttnn/ttnn/cpp/...`` on a source
    build, so every generic_op kernel failed to JIT-build with "No such file or directory"
    and four of the release gate's models could not fold at all.
    """
    import os
    import ttnn as _t
    from pathlib import Path
    root = Path(_t.__file__).resolve().parent
    candidates = [root / "ttnn", root.parent]
    metal_home = os.environ.get("TT_METAL_HOME")
    if metal_home:
        candidates.append(Path(metal_home) / "ttnn")
    for cand in candidates:
        if (cand / "cpp" / "ttnn" / "operations").is_dir():
            return cand
    return candidates[0]


def _kernel_dir(kind="minimal_matmul"):
    return ttnn_cpp_root() / "cpp/ttnn/operations/experimental" / kind / "device/kernels"


def ckc_args(ckc):
    """The four fields ``ComputeConfigDescriptor`` takes, read off a ``DeviceComputeKernelConfig``.

    ``get_compute_kernel_config_args`` also returns ``packer_l1_acc``, which the C++ factory
    destructures and never passes to ``ComputeConfig``; there is no field for it here either.
    """
    return (getattr(ckc, "math_fidelity", ttnn.MathFidelity.HiFi4),
            bool(getattr(ckc, "math_approx_mode", False)),
            bool(getattr(ckc, "fp32_dest_acc_en", False)),
            bool(getattr(ckc, "dst_full_sync_en", False)))


def _div_up(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _div_up(a, b) * b


def _build_core_order_for_axis(core, transpose, axis_length, noc, axis_is_x, initial_endpoint):
    """Transcription of ``build_core_order_for_axis``. ``core`` is (x, y)."""
    order = [initial_endpoint]
    cx, cy = core
    current = (cy if axis_is_x else cx) if transpose else (cx if axis_is_x else cy)
    increasing = noc == ttnn.NOC.NOC_0
    index_of_current = 0
    for w in range(1, axis_length):
        val = w if increasing else (axis_length - w)
        if transpose:
            wc = (cx, val) if axis_is_x else (val, cy)
        else:
            wc = (val, cy) if axis_is_x else (cx, val)
        if val == current:
            index_of_current = w
        order.append(wc)
    return order, index_of_current


def _cb(idx, core_grid, page_size, num_tiles, data_format):
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=data_format, page_size=page_size)
    return ttnn.CBDescriptor(
        total_size=num_tiles * page_size, core_ranges=core_grid, format_descriptors=[fmt])


def build(device, in0, in1, outs, cfg, ckc, defines=(), kernel_dir=None, m_k=None,
          noc_mode=None):
    """The ProgramDescriptor for ``minimal_matmul(in0, in1) -> outs`` with block config ``cfg``.

    ``cfg`` is a 5-tuple ``(M_block, K_block, N_block, subblock_h, subblock_w)`` and a
    ``(grid_x, grid_y)``; ``ckc`` is ``(math_fidelity, math_approx_mode, fp32_dest_acc_en,
    dst_full_sync_en)``. ``outs`` is a list of output tensors, one per N chunk, so a single-output
    matmul and the three-way qkv split are the same code path. Everything else is read off the
    tensors, exactly as the C++ factory does.

    ``defines`` and ``kernel_dir`` are the only additions to the transcription: they let the two DM
    kernels come from ``tt_bio/kernels/triatt/`` with the head-major guards set, which is K1.
    ``m_k`` overrides the ``(M, K)`` the factory would infer from ``in0``'s shape, which a
    head-major activation reports wrongly: ``[S, 8, S, 32]`` is the same tile grid as
    ``[S*S, 256]`` but its last dim is 32.

    ``noc_mode`` is for a DM kernel that issues transactions on the NOC it was NOT configured
    with. Under the default ``DM_DEDICATED_NOC`` the firmware only runs ``noc_local_state_init``
    for the kernel's own NOC, so a write on the other one never issues and the barrier spins
    forever -- MEASURED as a device hang on the first call, card 0, 2026-08-15.
    ``DM_DYNAMIC_NOC`` initialises both and arbitrates the command buffers atomically.
    """
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    out = outs[0]
    (M_block_tiles, K_block_tiles, N_block_tiles, subblock_h, subblock_w), (gx, gy) = cfg
    math_fidelity, math_approx_mode, fp32_dest_acc_en, dst_full_sync_en = ckc
    defines = [(str(k), str(v)) for k, v in dict(defines).items()]

    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    in0_shape = [int(d) for d in in0.padded_shape]
    in1_shape = [int(d) for d in in1.padded_shape]
    if m_k is None:
        K = in0_shape[-1]
        M = 1
        for d in in0_shape[:-1]:
            M *= d
    else:
        M, K = m_k
        vol = 1
        for d in in0_shape:
            vol *= d
        assert M * K == vol, (m_k, in0_shape)
    N = in1_shape[-1]

    M_tiles, K_tiles, N_tiles = M // TILE_HW, K // TILE_HW, N // TILE_HW
    if K_SPLIT > 1 and K_block_tiles % K_SPLIT == 0 and K_block_tiles // K_SPLIT >= 1:
        K_block_tiles //= K_SPLIT
    if BLOCK_OVERRIDE:
        M_block_tiles = BLOCK_OVERRIDE.get("M", M_block_tiles)
        N_block_tiles = BLOCK_OVERRIDE.get("N", N_block_tiles)
        subblock_h = BLOCK_OVERRIDE.get("sh", subblock_h)
        subblock_w = BLOCK_OVERRIDE.get("sw", subblock_w)
        # The compute kernel clamps its own subblock to the block it is given, but the descriptor
        # has to be legal before it gets there: a subblock wider than its block indexes past the
        # CB, and DST holds 8 bf16 tiles.
        subblock_h = max(1, min(subblock_h, M_block_tiles))
        subblock_w = max(1, min(subblock_w, N_block_tiles))
        while subblock_h * subblock_w > 8:
            subblock_h = max(1, subblock_h // 2) if subblock_h > 1 else subblock_h
            if subblock_h * subblock_w > 8:
                subblock_w = max(1, subblock_w // 2)
    N_chunks = len(outs)
    N_tiles_per_chunk = N_tiles // N_chunks

    in0_tile_size = tile_bytes(in0.dtype)
    in1_tile_size = tile_bytes(in1.dtype)
    out_tile_size = tile_bytes(out.dtype)
    in2_tile_size = in1_tile_size          # no bias: in2_data_format = in1_data_format
    in3_tile_size = in1_tile_size          # no all-gather fusion, same fallback
    interm_fmt = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    interm_tile_size = tile_bytes(interm_fmt)

    transpose = M > N
    in0_noc = NOC_FOR_DRAM_READ if transpose else NOC_FOR_DRAM_WRITE
    in0_risc = (ttnn.DataMovementProcessor.RISCV_0 if transpose
                else ttnn.DataMovementProcessor.RISCV_1)
    in1_noc = NOC_FOR_DRAM_WRITE if transpose else NOC_FOR_DRAM_READ
    in1_risc = (ttnn.DataMovementProcessor.RISCV_1 if transpose
                else ttnn.DataMovementProcessor.RISCV_0)
    in0_axis_cores = gx if transpose else gy
    in1_axis_cores = gy if transpose else gx

    padded_M_tiles = _round_up(M_tiles, in0_axis_cores)
    padded_N_tiles = _round_up(N_tiles, in1_axis_cores)
    padded_K_tiles = _round_up(K_tiles, K_block_tiles)
    M_tiles_per_core = padded_M_tiles // in0_axis_cores
    N_tiles_per_core = padded_N_tiles // in1_axis_cores
    K_blocks = padded_K_tiles // K_block_tiles
    M_blocks_per_core = _div_up(M_tiles_per_core, M_block_tiles)
    N_blocks_per_core = _div_up(N_tiles_per_core, N_block_tiles)

    in0_block = M_block_tiles * K_block_tiles
    in1_block = K_block_tiles * N_block_tiles
    out_block = M_block_tiles * N_block_tiles

    TILE_TRAFFIC_STATS[(M_tiles, K_tiles, N_tiles, M_block_tiles, K_block_tiles, N_block_tiles)] = {
        "cores": gx * gy,
        "in0_tiles_per_core": M_tiles_per_core * padded_K_tiles * N_blocks_per_core,
        "in1_tiles_per_core": padded_K_tiles * N_tiles_per_core * M_blocks_per_core,
        "out_tiles_per_core": M_tiles_per_core * N_tiles_per_core,
        "m_blocks_per_core": M_blocks_per_core, "n_blocks_per_core": N_blocks_per_core,
    }
    depth = _fit_depth(
        CB_DEPTH,
        in0_block * in0_tile_size + in1_block * in1_tile_size + out_block * out_tile_size,
        out_block * interm_tile_size,
        (M_block_tiles, K_block_tiles, N_block_tiles, in0_tile_size, out_tile_size))
    cbs = [
        _cb(0, core_grid, in0_tile_size, in0_block * depth, in0.dtype),
        _cb(1, core_grid, in1_tile_size, in1_block * depth, in1.dtype),
        _cb(2, core_grid, out_tile_size, out_block * depth, out.dtype),
        # The accumulation buffer, not a pipeline stage: the compute kernel reserves exactly one
        # out block into it and packs K_num_blocks times with L1 accumulation on top.
        _cb(3, core_grid, interm_tile_size, out_block, interm_fmt),
    ]

    # CreateSemaphore is called six times on the whole grid, so the ids are 0..5 in that order.
    sem_vals = [INVALID, INVALID, VALID, INVALID, INVALID, VALID]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=core_grid, initial_value=v)
        for i, v in enumerate(sem_vals)]
    in0_sender_sem, in0_recv_sem, in0_valid_sem, in1_sender_sem, in1_recv_sem, in1_valid_sem = range(6)

    acc_in0 = list(ttnn.TensorAccessorArgs(in0).get_compile_time_args())
    acc_in1 = list(ttnn.TensorAccessorArgs(in1).get_compile_time_args())
    acc_out = [a for o in outs for a in ttnn.TensorAccessorArgs(o).get_compile_time_args()]

    in0_is_writer = not transpose
    in1_is_writer = transpose

    def dm_ct(tile_size, sems, is_writer, is_injector, acc_main, tail):
        return ([M_tiles, padded_M_tiles, K_tiles, padded_K_tiles, N_tiles, padded_N_tiles,
                 M_block_tiles, K_block_tiles, N_block_tiles, M_blocks_per_core, N_blocks_per_core,
                 tile_size, out_tile_size, in2_tile_size, *sems,
                 int(is_writer), int(is_injector), N_chunks, N_tiles_per_chunk] + tail
                + acc_main + acc_out)

    in0_sems = [in0_sender_sem, in0_recv_sem, in0_valid_sem]
    in1_sems = [in1_sender_sem, in1_recv_sem, in1_valid_sem]

    def cr(a, b):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b))])

    in0_sender_cores = cr((0, 0), (gx - 1, 0) if transpose else (0, gy - 1))
    in0_recv_cores = cr((0, 1) if transpose else (1, 0), (gx - 1, gy - 1))
    in1_sender_cores = cr((0, 0), (0, gy - 1) if transpose else (gx - 1, 0))
    in1_recv_cores = cr((1, 0) if transpose else (0, 1), (gx - 1, gy - 1))

    kd = _kernel_dir()
    dmd = kernel_dir or kd
    in0_src, in1_src = str(dmd / "dm_in0_sender.cpp"), str(dmd / "dm_in1_sender_out.cpp")
    compute_src = str(kd / "compute.cpp")          # never patched, always the wheel's own

    k_blocks_per_core = _div_up(K_blocks, in1_axis_cores if transpose else in0_axis_cores)

    in0_addr, in1_addr = in0.buffer_address(), in1.buffer_address()
    out_addrs = [o.buffer_address() for o in outs]

    rt = {"in0_sender": [], "in0_recv": [], "in1_sender": [], "in1_recv": [], "compute": []}
    for cx in range(gx):
        for cy in range(gy):
            core = (cx, cy)
            in0_idx = cx if transpose else cy
            in1_idx = cy if transpose else cx
            left_core, top_core = (0, cy), (cx, 0)

            in0_order, in0_i = _build_core_order_for_axis(
                core, transpose, in1_axis_cores, in0_noc, True,
                top_core if transpose else left_core)
            in1_order, in1_i = _build_core_order_for_axis(
                core, transpose, in0_axis_cores, in1_noc, False,
                left_core if transpose else top_core)

            def phys(c):
                p = device.worker_core_from_logical_core(ttnn.CoreCoord(c[0], c[1]))
                return int(p.x), int(p.y)

            in0_prev = phys(in0_order[max(in0_i - 1, 0)])
            in0_next = phys(in0_order[min(in0_i + 1, len(in0_order) - 1)])
            in1_prev = phys(in1_order[max(in1_i - 1, 0)])
            in1_next = phys(in1_order[min(in1_i + 1, len(in1_order) - 1)])

            M_start, M_end = M_tiles_per_core * in0_idx, M_tiles_per_core * (in0_idx + 1)
            N_start, N_end = N_tiles_per_core * in1_idx, N_tiles_per_core * (in1_idx + 1)
            defer_k = min(cy * k_blocks_per_core, K_blocks - 1)

            cc = ttnn.CoreCoord(cx, cy)
            a0 = [in0_addr, 0, 0, int(core == in0_order[-1]),
                  in0_next[0], in0_next[1], in0_prev[0], in0_prev[1],
                  M_start, M_end, N_start, N_end, defer_k, *out_addrs]
            a1 = [in1_addr, 0, int(core == in1_order[-1]),
                  in1_next[0], in1_next[1], in1_prev[0], in1_prev[1],
                  M_start, M_end, N_start, N_end, defer_k, *out_addrs]
            rt["in0_sender" if in1_idx == 0 else "in0_recv"].append((cc, a0))
            rt["in1_sender" if in0_idx == 0 else "in1_recv"].append((cc, a1))
            rt["compute"].append((cc, [M_start, M_end, N_start, N_end]))

    def dm_kernel(src, cores, ct, args, risc, noc):
        dmc = (ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
               if noc_mode is None else
               ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc, noc_mode=noc_mode))
        return ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=args, defines=defines,
            config=dmc)

    kernels = [
        dm_kernel(in0_src, in0_sender_cores,
                  dm_ct(in0_tile_size, in0_sems, in0_is_writer, True, acc_in0, [in3_tile_size]),
                  rt["in0_sender"], in0_risc, in0_noc),
        dm_kernel(in0_src, in0_recv_cores,
                  dm_ct(in0_tile_size, in0_sems, in0_is_writer, False, acc_in0, [in3_tile_size]),
                  rt["in0_recv"], in0_risc, in0_noc),
        dm_kernel(in1_src, in1_sender_cores,
                  dm_ct(in1_tile_size, in1_sems, in1_is_writer, True, acc_in1, []),
                  rt["in1_sender"], in1_risc, in1_noc),
        dm_kernel(in1_src, in1_recv_cores,
                  dm_ct(in1_tile_size, in1_sems, in1_is_writer, False, acc_in1, []),
                  rt["in1_recv"], in1_risc, in1_noc),
        ttnn.KernelDescriptor(
            kernel_source=compute_src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=[K_blocks, M_block_tiles, K_block_tiles, N_block_tiles,
                               M_blocks_per_core, N_blocks_per_core, subblock_h, subblock_w],
            runtime_args=rt["compute"],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity, math_approx_mode=math_approx_mode,
                fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en)),
    ]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    return {"pd": pd, "kernels": kernels, "cbs": cbs, "semaphores": semaphores, "rt": rt,
            "addrs": (in0_addr, in1_addr, tuple(out_addrs)), "n_chunks": N_chunks,
            "dims": {"M_tiles": M_tiles, "K_tiles": K_tiles, "N_tiles": N_tiles,
                     "padded_M_tiles": padded_M_tiles, "padded_N_tiles": padded_N_tiles,
                     "M_blocks_per_core": M_blocks_per_core,
                     "N_blocks_per_core": N_blocks_per_core, "K_blocks": K_blocks,
                     "N_tiles_per_chunk": N_tiles_per_chunk,
                     "transpose_core_grid": transpose, "defines": defines}}


def _key(in0, in1, outs, cfg, ckc, defines, kernel_dir, m_k=None, noc_mode=None):
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    return (spec(in0), spec(in1), tuple(spec(o) for o in outs),
            cfg, tuple(str(c) for c in ckc),
            tuple(sorted(dict(defines).items())), str(kernel_dir), m_k, str(noc_mode))


def generic_minimal_matmul(device, in0, in1, outs, cfg, ckc, defines=(), kernel_dir=None,
                           m_k=None, noc_mode=None):
    """``minimal_matmul`` through ``generic_op``, descriptor cached per shape/config."""
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    key = _key(in0, in1, outs, cfg, ckc, defines, kernel_dir, m_k, noc_mode)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = build(device, in0, in1, outs, cfg, ckc, defines, kernel_dir, m_k,
                                    noc_mode)
    addrs = (in0.buffer_address(), in1.buffer_address(),
             tuple(o.buffer_address() for o in outs))
    if addrs != entry["addrs"]:
        rebind(entry, *addrs)
    ttnn.generic_op([in0, in1, *outs], entry["pd"])
    return outs[0] if len(outs) == 1 else outs


def rebind(entry, in0_addr, in1_addr, out_addrs):
    """Rewrite the buffer addresses in the cached per-core runtime args, in place.

    They sit at fixed indices: args[0] is the kernel's own input address and the last ``N_chunks``
    entries are the output addresses. 110 cores x 2 DM kernels, so a few hundred scalar writes plus
    the binding round-trip.
    """
    n = entry["n_chunks"]
    rt = entry["rt"]
    for name, addr in (("in0_sender", in0_addr), ("in0_recv", in0_addr),
                       ("in1_sender", in1_addr), ("in1_recv", in1_addr)):
        for _, a in rt[name]:
            a[0] = addr
            a[len(a) - n:] = list(out_addrs)
    for k, name in zip(entry["kernels"][:4],
                       ("in0_sender", "in0_recv", "in1_sender", "in1_recv")):
        k.runtime_args = rt[name]
    entry["pd"] = ttnn.ProgramDescriptor(
        kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])
    entry["addrs"] = (in0_addr, in1_addr, tuple(out_addrs))
