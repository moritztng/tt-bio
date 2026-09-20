#!/usr/bin/env python3
"""``ttnn.matmul``'s 1D ``mcast_in1`` program, transcribed into a ``ttnn.ProgramDescriptor``.

``tt_bio/mm_generic.py`` did this for ``minimal_matmul``; this is the same exercise against
``process_mcast_in1_program_and_create_override_variables`` in
``ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp``
at the ``v0.67.4`` tag, which is the tag the installed ``ttnn 0.67.4`` wheel is cut from. The
wheel ships those kernel sources in place, so they are pointed at through ``ttnn_cpp_root()`` and
nothing here needs a tt-metal build.

Why: the writer split (`perf/writersplit`) lives in that factory, the factory is compiled into the
wheel's ``.so``, and a source patch therefore cannot reach a user. Driving the same program through
``generic_op`` puts the whole thing in Python, where the split is an edit to a copy of two dataflow
kernels plus a handful of extra args -- the shape ``tt_bio/kernels/mm_split`` and
``tt_bio/kernels/triatt`` already ship in.

Only the case the fold issues is covered, and every other case is asserted out rather than
silently mis-built: interleaved in0/in1/out, no bias, no fused activation, no sharding, no
untilize, no transpose of either operand or of a tile, no op fusion, no sparsity, 32x32 tiles.
The block config is passed in rather than re-derived: ``create_matmul_1d_systolic_array_program_config``
picks it from an L1 fit estimate, and a second copy of that arithmetic is a second thing to keep
in step. Read it off the config a caller already has, or pass the same one to ``ttnn.matmul``.
"""

from __future__ import annotations

import os

import ttnn

from tt_bio.mm_generic import TILE_HW, ttnn_cpp_root, tile_bytes

INVALID, VALID = 0, 1

# detail::preferred_noc_for_dram_{read,write} take the default branch on Blackhole
# (tt_metal/api/tt-metalium/kernel_types.hpp:126-138).
NOC_FOR_DRAM_READ = ttnn.NOC.NOC_0
NOC_FOR_DRAM_WRITE = ttnn.NOC.NOC_1

#: The factory's CB indices, named the way the kernels' ``named_compile_args`` name them.
CB_IN0, CB_IN1, CB_IN0_SHARDED, CB_BIAS = 0, 1, 2, 3
CB_OUT, CB_INTERM0, CB_SPARSITY0, CB_SPARSITY1 = 4, 5, 6, 7
CB_IN0_INTERMEDIATE, CB_IN1_INTERMEDIATE, CB_IN0_TRANSPOSED = 8, 9, 10

_MM_KERNELS = "cpp/ttnn/operations/matmul/device/kernels"
IN0_SENDER = "dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp"
IN1_SENDER = "dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp"
IN1_RECEIVER = "dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp"
COMPUTE = "compute/bmm_large_block_zm_fused_bias_activation.cpp"

#: A default-constructed ``TensorAccessorArgs`` appends ``{args_config.raw(), aligned_page_size}``
#: with a null buffer, so ``{0, 0}`` (tt_metal/impl/buffers/tensor_accessor_args.cpp:155-168).
#: The factory pushes one of these wherever sparsity is not in play.
ACCESSOR_PLACEHOLDER = [0, 0]

#: Env vars that make the C++ factory add compute-kernel defines this transcription does not
#: reproduce (compute_throttle_utils.cpp). Unset they are no-ops, which is the only state covered.
_THROTTLE_ENV = ("TT_MM_STAGGER_TYPE", "TT_MM_THROTTLE_PERF", "TT_MM_SKIP_IN1_DRAM",
                 "TT_MM_SYNC_AFTER_IN1_DRAM")

_CACHE: dict = {}


def _num_cores_to_corerangeset(start, target, grid, row_wise=True):
    """Transcription of ``tt_metal/common/work_split.cpp:77`` -- the start-core overload.

    ttnn only binds the ``{0, 0}`` one, and the in1 mcast receivers start at the core after
    ``start_core``. Building the receivers as one CoreRange per core instead would be the same
    set of cores but not the same three merged ranges the factory hands CreateKernel.
    """
    gx, gy = grid.x, grid.y
    assert start.x < gx and start.y < gy, (start, grid)
    ranges = []
    left = target
    sx, sy = start.x, start.y
    if sx != 0 and left > gx - start.x:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(gx - 1, sy)))
        left -= gx - sx
        sx, sy = 0, sy + 1
    if left > gx:
        rows = left // gx
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(gx - 1, sy + rows - 1)))
        left -= rows * gx
        sx, sy = 0, sy + rows
    if left > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(sx + left - 1, sy)))
    return ttnn.CoreRangeSet(ranges)


def _dram_interleaved(t):
    mc = t.memory_config()
    return not mc.is_sharded()


def _cb(total_size, cores, fmts):
    return ttnn.CBDescriptor(total_size=total_size, core_ranges=cores, format_descriptors=fmts)


def _fmt(idx, dtype, page_size):
    return ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page_size)


def build(device, in0, in1, out, pc, ckc, kernel_dir=None, defines=(), bcast_batch=True):
    """The ProgramDescriptor for ``matmul(in0, in1) -> out`` on the 1D ``mcast_in1`` path.

    ``pc`` is ``(grid_xy, in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w,
    per_core_M, per_core_N)`` -- the fields of ``MatmulMultiCoreReuseMultiCast1DProgramConfig``
    with ``mcast_in0=False`` and ``fuse_batch=True``. ``ckc`` is ``(math_fidelity,
    math_approx_mode, fp32_dest_acc_en, packer_l1_acc)``.

    ``kernel_dir`` re-points the two dataflow kernels at a directory of our own copies, which is
    how the writer split ships; the compute kernel always comes from the wheel.
    """
    for v in _THROTTLE_ENV:
        assert os.environ.get(v) is None, f"{v} changes the shipped program; not transcribed"
    assert _dram_interleaved(in0) and _dram_interleaved(in1) and _dram_interleaved(out), \
        "sharded in0/out take branches this transcription does not cover"
    for t in (in0, in1, out):
        assert tuple(t.tile.tile_shape) == (TILE_HW, TILE_HW), "only 32x32 tiles"

    (gx, gy), in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w, \
        per_core_M, per_core_N = pc
    math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc = ckc
    defines = [(str(k), str(v)) for k, v in dict(defines).items()]

    in0_shape = [int(d) for d in in0.padded_shape]
    in1_shape = [int(d) for d in in1.padded_shape]
    # fuse_batch=true, so the batch folds into M and B is 1 (utilities::get_M_dim).
    B = 1
    Mt = 1
    for d in in0_shape[:-1]:
        Mt *= d
    Mt //= TILE_HW
    Kt = in0_shape[-1] // TILE_HW
    Nt = in1_shape[-1] // TILE_HW
    assert Kt % in0_block_w == 0, (Kt, in0_block_w)

    num_blocks = Kt // in0_block_w
    packer_l1_acc_en = bool(packer_l1_acc) and num_blocks > 2
    if packer_l1_acc_en:
        interm0_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    else:
        interm0_dtype = ttnn.float32 if fp32_dest_acc_en else out.dtype

    in0_tile_size = tile_bytes(in0.dtype)
    in1_tile_size = tile_bytes(in1.dtype)
    out_tile_size = tile_bytes(out.dtype)
    interm0_tile_size = tile_bytes(interm0_dtype)

    in0_block_h, in1_block_w = out_block_h, out_block_w
    out_num_blocks_y = per_core_M // out_block_h
    out_num_blocks_x = per_core_N // out_block_w

    in0_block_tiles = in0_block_h * in0_block_w
    in0_CB_size = in0_block_tiles * (2 if B * num_blocks > 1 else 1) * in0_tile_size
    in1_block_tiles = out_block_w * in0_block_w
    in1_CB_size = in1_block_tiles * (2 if B * num_blocks > 1 else 1) * in1_tile_size
    out_block_tiles = out_block_h * out_block_w
    out_CB_size = out_block_tiles * out_tile_size
    interm0_CB_size = out_block_tiles * interm0_tile_size

    # pad_last_ktile, transpose_a=false so the width side is the live one.
    in0_last_ktile_w = int(in0.shape[-1]) % TILE_HW

    start_core = ttnn.CoreCoord(0, 0)
    num_blocks_y = (Mt - 1) // per_core_M + 1
    num_blocks_x = (Nt - 1) // per_core_N + 1
    num_cores = num_blocks_y * num_blocks_x
    grid = ttnn.CoreCoord(gx, gy)
    assert num_cores <= gx * gy, (num_cores, gx, gy)

    all_cores = _num_cores_to_corerangeset(start_core, num_cores, grid, True)
    bbox = all_cores.bounding_box()
    in1_mcast_receiver_num_cores = ((bbox.end.x - bbox.start.x + 1)
                                    * (bbox.end.y - bbox.start.y + 1))
    in1_mcast_sender = ttnn.CoreRangeSet([ttnn.CoreRange(start_core, start_core)])
    in1_mcast_receivers = ttnn.CoreRangeSet([])
    if in1_mcast_receiver_num_cores > 1:
        recv_start = (ttnn.CoreCoord(start_core.x + 1, start_core.y) if start_core.x != gx - 1
                      else ttnn.CoreCoord(start_core.x, start_core.y + 1))
        in1_mcast_receivers = _num_cores_to_corerangeset(recv_start, num_cores - 1, grid, True)

    sem_sender, sem_receiver = 0, 1
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_cores, initial_value=INVALID)
                  for i in (sem_sender, sem_receiver)]

    def phys(c):
        p = device.worker_core_from_logical_core(c)
        return int(p.x), int(p.y)

    top_left = phys(bbox.start)
    bottom_right = phys(bbox.end)

    in0_tensor_stride_w, in0_tensor_stride_h = 1, Kt
    in1_tensor_stride_w, in1_tensor_stride_h = 1, Nt

    acc_in0 = list(ttnn.TensorAccessorArgs(in0).get_compile_time_args())
    acc_in1 = list(ttnn.TensorAccessorArgs(in1).get_compile_time_args())
    acc_out = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    in0_sender_ct = [
        in0_tensor_stride_w, in0_tensor_stride_h,
        in0_block_w * in0_tensor_stride_w,          # in0_tensor_next_block_stride
        in0_block_h * in0_tensor_stride_h,          # in0_tensor_next_h_dim_block_stride
        in0_block_w, in0_block_h, in0_block_w * in0_block_h,
        in0_last_ktile_w, 0,                        # transpose_a=false -> last_ktile_h = 0
        0, 0, 0,                                    # extract_shard_sub_blocks, shard w/h in tiles
        num_blocks, out_num_blocks_x, out_num_blocks_y,
        0, 0, 0, 0,                                 # in0 mcast args, unused on this path
        Mt * Kt, B,
        0, 0, 1, 0,                                 # batchB, sparsity_pagesize, bcast_A, get_batch
        0,                                          # fuse_op
    ] + acc_in0 + ACCESSOR_PLACEHOLDER

    writer_ct = [
        1, Nt,                                      # out_tensor_stride_w / _h
        out_subblock_w, out_subblock_h * Nt,        # next_subblock_stride_w / _h
        out_block_w, out_block_h * Nt,              # next_w_dim / _h_dim block stride
        out_subblock_w, out_subblock_h, out_subblock_w * out_subblock_h,
        Mt * Nt,
    ]
    in1_sender_ct = [
        in1_tensor_stride_w, in1_tensor_stride_h,
        in0_block_w * in1_tensor_stride_h,          # in1_tensor_next_block_stride
        in1_block_w * in1_tensor_stride_w,          # in1_tensor_next_w_dim_block_stride
        in1_block_w, in0_block_w, in1_block_w * in0_block_w,
        num_blocks, out_num_blocks_x, out_num_blocks_y,
        sem_sender, sem_receiver,
        num_cores - 1, in1_mcast_receiver_num_cores - 1,
        Kt * Nt, B, int(bcast_batch),
        0, 0,                                       # batchB, sparsity_pagesize
    ] + writer_ct + [
        0,                                          # in3_tensor_stride_w placeholder (no bias)
        0, 0,                                       # fuse_op twice
    ] + acc_in1 + ACCESSOR_PLACEHOLDER + acc_out

    in1_receiver_ct = [
        in1_block_w * in0_block_w,
        num_blocks, out_num_blocks_x, out_num_blocks_y,
        sem_sender, sem_receiver,
        B,
    ] + writer_ct + [
        0,                                          # in1_block_w placeholder (no bias)
        0,                                          # fuse_op
    ] + acc_out

    in0_num_subblocks = out_block_h // out_subblock_h
    in1_num_subblocks = out_block_w // out_subblock_w
    compute_ct = [
        in0_block_w, in0_num_subblocks,
        out_subblock_h * in0_block_w * in0_num_subblocks, out_subblock_h * in0_block_w,
        in1_num_subblocks, out_subblock_w * in0_block_w * in1_num_subblocks,
        out_subblock_w * in1_num_subblocks,
        num_blocks, out_num_blocks_x, out_num_blocks_y,
        out_subblock_h, out_subblock_w, out_subblock_h * out_subblock_w,
        B, out_block_tiles,
        0,                                          # untilize_out
        0,                                          # get_batch_from_reader
        0,                                          # in0_transpose_tile
    ]

    compute_defines = list(defines)
    if packer_l1_acc_en:
        compute_defines.append(("PACKER_L1_ACC", "1"))
    if fp32_dest_acc_en:
        compute_defines.append(("FP32_DEST_ACC_EN", "1"))

    in0_sender_defines = list(defines) + [("SKIP_MCAST", "1")]
    in1_sender_defines = list(defines)
    in1_receiver_defines = list(defines)
    if in1_mcast_receiver_num_cores == 1:
        in1_sender_defines.append(("SKIP_MCAST", "1"))
    # Blackhole's tiny-tile alignment workaround only fires for a tile size that is not a
    # multiple of 64; bf16 (2048) and bfp8_b (1088) both are, fp32 (4096) is.
    assert in0_tile_size % 64 == 0 and in1_tile_size % 64 == 0, "INTERMEDIATE_CB_READ not transcribed"

    # --- circular buffers -------------------------------------------------------------------
    cbs = [
        _cb(in0_CB_size, all_cores, [_fmt(CB_IN0, in0.dtype, in0_tile_size)]),
        _cb(in1_CB_size, all_cores, [_fmt(CB_IN1, in1.dtype, in1_tile_size)]),
    ]
    if interm0_dtype != out.dtype:
        cbs.append(_cb(out_CB_size, all_cores, [_fmt(CB_OUT, out.dtype, out_tile_size)]))
        cbs.append(_cb(interm0_CB_size, all_cores,
                       [_fmt(CB_INTERM0, interm0_dtype, interm0_tile_size)]))
    else:
        cbs.append(_cb(out_CB_size, all_cores,
                       [_fmt(CB_OUT, out.dtype, out_tile_size),
                        _fmt(CB_INTERM0, interm0_dtype, interm0_tile_size)]))

    # --- runtime args -----------------------------------------------------------------------
    last_per_core_M = per_core_M if Mt % per_core_M == 0 else Mt % per_core_M
    last_out_block_h = out_block_h if last_per_core_M % out_block_h == 0 else last_per_core_M % out_block_h
    last_out_num_blocks_h = (last_per_core_M - 1) // out_block_h + 1
    last_nonzero_subblocks_h = (last_out_block_h - 1) // out_subblock_h + 1
    last_subblock_of_last_block_h = (out_subblock_h if last_out_block_h % out_subblock_h == 0
                                     else last_out_block_h % out_subblock_h)
    last_padded_block_tiles_h_skip = ((out_block_h // out_subblock_h - last_nonzero_subblocks_h)
                                      * (out_block_w * out_subblock_h))

    in1_noc = NOC_FOR_DRAM_READ
    in0_noc = NOC_FOR_DRAM_WRITE
    start_core_noc, end_core_noc = bottom_right, top_left
    if in1_noc == ttnn.NOC.NOC_0:
        start_core_noc, end_core_noc = end_core_noc, start_core_noc

    in0_addr, in1_addr, out_addr = (in0.buffer_address(), in1.buffer_address(),
                                    out.buffer_address())
    cores = ttnn.corerange_to_cores(all_cores, None, True)
    rt = {"in0_sender": [], "in1_sender": [], "in1_receiver": [], "compute": []}
    for i, core in enumerate(cores):
        output_idx_x = i // num_blocks_y
        output_idx_y = i % num_blocks_y
        out_start_tile_id = output_idx_x * per_core_N + output_idx_y * per_core_M * Nt
        if core == start_core:
            rt["in1_sender"].append((core, [
                in1_addr,
                per_core_N * in1_tensor_stride_w * output_idx_x,
                start_core_noc[0], start_core_noc[1], end_core_noc[0], end_core_noc[1],
                0,                                  # sparsity_addr
                out_addr, out_start_tile_id,
                out_block_w,                        # last_block_w (READER padding)
                out_block_h // out_subblock_h, out_subblock_h, 0,
                out_block_w // out_subblock_w, out_block_w // out_subblock_w, out_subblock_w, 0, 0,
                0, 0,                               # bias addr / start tile id
                out_num_blocks_x,
            ]))
        else:
            last_h = output_idx_y == num_blocks_y - 1
            a = [top_left[0], top_left[1], out_addr, out_start_tile_id]
            if last_h:
                a += [out_block_h // out_subblock_h, last_nonzero_subblocks_h,
                      last_subblock_of_last_block_h, last_padded_block_tiles_h_skip]
            else:
                a += [out_block_h // out_subblock_h, out_block_h // out_subblock_h,
                      out_subblock_h, 0]
            a += [out_block_w // out_subblock_w, out_block_w // out_subblock_w, out_subblock_w, 0, 0]
            a += [last_out_num_blocks_h if last_h else out_num_blocks_y, out_num_blocks_x]
            rt["in1_receiver"].append((core, a))
        rt["in0_sender"].append((core, [
            in0_addr,
            per_core_M * in0_tensor_stride_h * output_idx_y,
            0, 0, 0, 0,                             # in0 mcast args, unused
            per_core_M,                             # last_block_h
            0,                                      # sparsity_addr
        ]))

    # --- kernels ----------------------------------------------------------------------------
    wheel = ttnn_cpp_root() / _MM_KERNELS
    dmd = kernel_dir or (wheel / "dataflow")

    def dm(src, cores_, ct, args, risc, noc, defs, named):
        return ttnn.KernelDescriptor(
            kernel_source=str(src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores_, compile_time_args=ct, named_compile_time_args=named,
            runtime_args=args, defines=defs,
            config=ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc))

    kernels = [
        dm(dmd / os.path.basename(IN0_SENDER), all_cores, in0_sender_ct, rt["in0_sender"],
           ttnn.DataMovementProcessor.RISCV_1, in0_noc, in0_sender_defines,
           [("cb_in0", CB_IN0), ("cb_in0_sharded", CB_IN0_SHARDED),
            ("cb_sparsity", CB_SPARSITY0), ("cb_in0_intermediate", CB_IN0_INTERMEDIATE)]),
        dm(dmd / os.path.basename(IN1_SENDER), in1_mcast_sender, in1_sender_ct, rt["in1_sender"],
           ttnn.DataMovementProcessor.RISCV_0, in1_noc, in1_sender_defines,
           [("cb_in1", CB_IN1), ("cb_bias", CB_BIAS), ("cb_out", CB_OUT),
            ("cb_sparsity", CB_SPARSITY1), ("cb_in1_intermediate", CB_IN1_INTERMEDIATE)]),
    ]
    if in1_mcast_receivers.num_cores() > 0:
        kernels.append(
            dm(dmd / os.path.basename(IN1_RECEIVER), in1_mcast_receivers, in1_receiver_ct,
               rt["in1_receiver"], ttnn.DataMovementProcessor.RISCV_0, in1_noc,
               in1_receiver_defines,
               [("cb_in1", CB_IN1), ("cb_bias", CB_BIAS), ("cb_out", CB_OUT)]))
    kernels.append(ttnn.KernelDescriptor(
        kernel_source=str(wheel / COMPUTE),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores, compile_time_args=compute_ct, runtime_args=[],
        defines=compute_defines,
        named_compile_time_args=[
            ("cb_in0", CB_IN0), ("cb_in1", CB_IN1), ("cb_bias", CB_BIAS), ("cb_out", CB_OUT),
            ("cb_intermed0", CB_INTERM0), ("cb_in0_intermediate", CB_IN0_INTERMEDIATE),
            ("cb_in1_intermediate", CB_IN1_INTERMEDIATE),
            ("cb_in0_transposed", CB_IN0_TRANSPOSED)],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity, math_approx_mode=bool(math_approx_mode),
            fp32_dest_acc_en=bool(fp32_dest_acc_en), dst_full_sync_en=False)))

    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    return {"pd": pd, "kernels": kernels, "cbs": cbs, "semaphores": semaphores, "rt": rt,
            "addrs": (in0_addr, in1_addr, out_addr), "num_cores": num_cores,
            "dims": {"Mt": Mt, "Kt": Kt, "Nt": Nt, "B": B, "num_blocks": num_blocks,
                     "num_blocks_x": num_blocks_x, "num_blocks_y": num_blocks_y,
                     "packer_l1_acc_en": packer_l1_acc_en,
                     "interm0_dtype": str(interm0_dtype),
                     "mcast_receiver_cores": in1_mcast_receiver_num_cores}}


#: Where the output address sits in each kernel's per-core runtime args, for a rebind.
_OUT_ADDR_IDX = {"in1_sender": 7, "in1_receiver": 2}


def rebind(entry, in0_addr, in1_addr, out_addr):
    """Rewrite the buffer addresses in the cached per-core runtime args, in place."""
    rt = entry["rt"]
    for _, a in rt["in0_sender"]:
        a[0] = in0_addr
    for _, a in rt["in1_sender"]:
        a[0] = in1_addr
        a[_OUT_ADDR_IDX["in1_sender"]] = out_addr
    for _, a in rt["in1_receiver"]:
        a[_OUT_ADDR_IDX["in1_receiver"]] = out_addr
    names = ["in0_sender", "in1_sender"] + (["in1_receiver"] if rt["in1_receiver"] else [])
    for k, name in zip(entry["kernels"], names):
        k.runtime_args = rt[name]
    entry["pd"] = ttnn.ProgramDescriptor(
        kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])
    entry["addrs"] = (in0_addr, in1_addr, out_addr)


def _key(in0, in1, out, pc, ckc, kernel_dir, defines, bcast_batch):
    spec = lambda t: (str(t.padded_shape), str(t.shape), str(t.dtype),
                      str(t.memory_config()))
    return (spec(in0), spec(in1), spec(out), tuple(map(str, pc)), tuple(str(c) for c in ckc),
            str(kernel_dir), tuple(sorted(dict(defines).items())), bool(bcast_batch))


def generic_mm1d(device, in0, in1, out, pc, ckc, kernel_dir=None, defines=(), bcast_batch=True):
    """``matmul`` on the 1D ``mcast_in1`` path through ``generic_op``, cached per shape/config."""
    key = _key(in0, in1, out, pc, ckc, kernel_dir, defines, bcast_batch)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = build(device, in0, in1, out, pc, ckc, kernel_dir, defines,
                                    bcast_batch)
    addrs = (in0.buffer_address(), in1.buffer_address(), out.buffer_address())
    if addrs != entry["addrs"]:
        rebind(entry, *addrs)
    ttnn.generic_op([in0, in1, out], entry["pd"])
    return out
