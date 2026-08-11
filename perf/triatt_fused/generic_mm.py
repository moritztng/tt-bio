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

_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}

_CACHE: dict = {}


def _kernel_dir(kind="minimal_matmul"):
    """The wheel's own kernel sources. Pointed at in place so their sibling includes resolve."""
    import ttnn as _t
    from pathlib import Path
    root = Path(_t.__file__).resolve().parent
    return root / "ttnn/cpp/ttnn/operations/experimental" / kind / "device/kernels"


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


def build(device, in0, in1, out, cfg, ckc):
    """The ProgramDescriptor for ``minimal_matmul(in0, in1) -> out`` with block config ``cfg``.

    ``cfg`` is a 5-tuple ``(M_block, K_block, N_block, subblock_h, subblock_w)`` and a
    ``(grid_x, grid_y)``; ``ckc`` is ``(math_fidelity, math_approx_mode, fp32_dest_acc_en,
    dst_full_sync_en)``. Everything else is read off the tensors, exactly as the C++ factory does.
    """
    (M_block_tiles, K_block_tiles, N_block_tiles, subblock_h, subblock_w), (gx, gy) = cfg
    math_fidelity, math_approx_mode, fp32_dest_acc_en, dst_full_sync_en = ckc

    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    in0_shape = [int(d) for d in in0.padded_shape]
    in1_shape = [int(d) for d in in1.padded_shape]
    K = in0_shape[-1]
    M = 1
    for d in in0_shape[:-1]:
        M *= d
    N = in1_shape[-1]

    M_tiles, K_tiles, N_tiles = M // TILE_HW, K // TILE_HW, N // TILE_HW
    N_chunks = 1
    N_tiles_per_chunk = N_tiles // N_chunks

    in0_tile_size = _TILE_BYTES[in0.dtype]
    in1_tile_size = _TILE_BYTES[in1.dtype]
    out_tile_size = _TILE_BYTES[out.dtype]
    in2_tile_size = in1_tile_size          # no bias: in2_data_format = in1_data_format
    in3_tile_size = in1_tile_size          # no all-gather fusion, same fallback
    interm_fmt = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    interm_tile_size = _TILE_BYTES[interm_fmt]

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

    cbs = [
        _cb(0, core_grid, in0_tile_size, in0_block * 2, in0.dtype),
        _cb(1, core_grid, in1_tile_size, in1_block * 2, in1.dtype),
        _cb(2, core_grid, out_tile_size, out_block * 2, out.dtype),
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
    acc_out = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

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
    in0_src, in1_src = str(kd / "dm_in0_sender.cpp"), str(kd / "dm_in1_sender_out.cpp")
    compute_src = str(kd / "compute.cpp")

    k_blocks_per_core = _div_up(K_blocks, in1_axis_cores if transpose else in0_axis_cores)

    in0_addr, in1_addr, out_addr = in0.buffer_address(), in1.buffer_address(), out.buffer_address()

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
                  M_start, M_end, N_start, N_end, defer_k, out_addr]
            a1 = [in1_addr, 0, int(core == in1_order[-1]),
                  in1_next[0], in1_next[1], in1_prev[0], in1_prev[1],
                  M_start, M_end, N_start, N_end, defer_k, out_addr]
            rt["in0_sender" if in1_idx == 0 else "in0_recv"].append((cc, a0))
            rt["in1_sender" if in0_idx == 0 else "in1_recv"].append((cc, a1))
            rt["compute"].append((cc, [M_start, M_end, N_start, N_end]))

    def dm_kernel(src, cores, ct, args, risc, noc):
        return ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=args,
            config=ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc))

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
            "addrs": (in0_addr, in1_addr, out_addr),
            "dims": {"M_tiles": M_tiles, "K_tiles": K_tiles, "N_tiles": N_tiles,
                     "padded_M_tiles": padded_M_tiles, "padded_N_tiles": padded_N_tiles,
                     "M_blocks_per_core": M_blocks_per_core,
                     "N_blocks_per_core": N_blocks_per_core, "K_blocks": K_blocks,
                     "transpose_core_grid": transpose}}


def _key(in0, in1, out, cfg, ckc):
    return (str(in0.padded_shape), str(in1.padded_shape), str(out.padded_shape),
            str(in0.dtype), str(in1.dtype), str(out.dtype),
            str(in0.memory_config()), str(in1.memory_config()), str(out.memory_config()),
            cfg, tuple(str(c) for c in ckc))


def generic_minimal_matmul(device, in0, in1, out, cfg, ckc):
    """``minimal_matmul`` through ``generic_op``, descriptor cached per shape/config."""
    key = _key(in0, in1, out, cfg, ckc)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = build(device, in0, in1, out, cfg, ckc)
    addrs = (in0.buffer_address(), in1.buffer_address(), out.buffer_address())
    if addrs != entry["addrs"]:
        rebind(entry, *addrs)
    ttnn.generic_op([in0, in1, out], entry["pd"])
    return out


def rebind(entry, in0_addr, in1_addr, out_addr):
    """Rewrite the three buffer addresses in the cached per-core runtime args, in place.

    The addresses sit at fixed indices: in0 args [0] = in0_addr and [-1] = out_addr, in1 args
    [0] = in1_addr and [-1] = out_addr. 110 cores x 2 DM kernels, so 440 scalar writes plus the
    binding round-trip -- measured by the harness, and the reason K1 moves them into
    ``common_runtime_args`` when it patches the writer anyway.
    """
    rt = entry["rt"]
    for name, addr in (("in0_sender", in0_addr), ("in0_recv", in0_addr),
                       ("in1_sender", in1_addr), ("in1_recv", in1_addr)):
        for _, a in rt[name]:
            a[0] = addr
            a[-1] = out_addr
    for k, name in zip(entry["kernels"][:4],
                       ("in0_sender", "in0_recv", "in1_sender", "in1_recv")):
        k.runtime_args = rt[name]
    entry["pd"] = ttnn.ProgramDescriptor(
        kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])
    entry["addrs"] = (in0_addr, in1_addr, out_addr)
