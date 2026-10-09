#!/usr/bin/env python3
"""ttnn's 2D multicast reuse matmul (``MatmulMultiCoreReuseMultiCastProgramConfig``) re-driven through
``ttnn.generic_op``, with the wheel's own six kernel sources unmodified.

A transcription of ``matmul_multicore_reuse_mcast_2d_program_factory.cpp`` at the ``v0.68.0`` tag (F2D below;
``git -C ~/tt-metal show v0.68.0:ttnn/cpp/ttnn/operations/matmul/device/factory/<that file>``) into a Python
``ttnn.ProgramDescriptor``. It is the S0 of the fc12g route (state/spd-swiglu-fused.md): if this reproduces the
stock op's bytes and time, the gated compute kernel is a swap of K6 on top of it (``compute_src`` and friends).

Covered, everything else asserted out:
  in0 INTERLEAVED (DRAM or L1, TensorAccessorArgs reads which), in1 INTERLEAVED with batch 1 (bcast_B),
  output BLOCK_SHARDED L1 on exactly the num_blocks_x x num_blocks_y grid at (0, 0), ROW_MAJOR,
  no bias, fused activation None or SILU, fuse_batch (B = 1), transpose_mcast False, no transpose_a/b,
  no sparsity, no all-gather / reduce-scatter fusion, untilize_out False, 32x32 tiles,
  bf16 / bfp8_b operands and output, Wormhole or Blackhole (the BH INTERMEDIATE_CB_READ branch, F2D:644-655,
  never fires for a 2048- or 1088-byte tile and is asserted).
num_blocks = Kt / in0_block_w is general: the in0/in1 double-buffer rule (F2D:118-129), PACKER_L1_ACC
(F2D:88) and the interm0 CB placement (F2D:90-93, 937-976) follow the factory.

Kernels (paths under ``ttnn/cpp/ttnn/operations/matmul/device/kernels/``), for the WH 8x9 fc12 case:
  K0 dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp           x0, all y        RISCV_1 in0_noc
  K1 dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp    y0, all x        RISCV_0 in1_noc
  K2 dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp  x<=half, y>=1    RISCV_0 in1_noc
  K3 dataflow/reader_bmm_tile_layout_in0_receiver.cpp                 x>=1 & (y0 | x<=half)  RISCV_1 in0_noc
  K4 = K2 source                                                      x>half, y>=1     RISCV_0 in1_split_noc
  K5 = K3 source                                                      x>half, y>=1     RISCV_1 in0_split_noc
  K6 compute/bmm_large_block_zm_fused_bias_activation.cpp             all cores        compute
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from .mm_generic import INVALID, TILE_HW, ckc_args, tile_bytes, ttnn_cpp_root

# tt_metal/api/tt-metalium/kernel_types.hpp:126-138 -- WH and BH both take these.
NOC_FOR_DRAM_READ = ttnn.NOC.NOC_0
NOC_FOR_DRAM_WRITE = ttnn.NOC.NOC_1

KDIR = "cpp/ttnn/operations/matmul/device/kernels"
K_IN0_SENDER = "dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp"
K_IN0_RECV = "dataflow/reader_bmm_tile_layout_in0_receiver.cpp"
K_IN1_SENDER = "dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp"
K_IN1_RECV = "dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp"
K_COMPUTE = "compute/bmm_large_block_zm_fused_bias_activation.cpp"

# Indices of the per-core runtime args the cache rebinds (F2D:1497-1532 does the same writes).
RT_IN0_SENDER_ADDR = 0          # K0: in0_tensor_addr
RT_IN1_SENDER_ADDR = 0          # K1: in1_tensor_addr
RT_IN1_SENDER_OUT = 7           # K1: out_tensor_addr
RT_IN1_RECV_OUT = 2             # K2/K4: out_tensor_addr
# out_num_nonzero_subblocks_w and out_last_num_nonzero_subblocks_w. Under OUT_SHARDED only the first is
# read, in the final cb_out.wait_front (in1_sender_writer_padding.cpp:677-679,
# in1_receiver_writer_padding.cpp:234-236); the `out_nzsb_w` override writes both.
RT_IN1_SENDER_NZSB_W = (13, 14)
RT_IN1_RECV_NZSB_W = (8, 9)

_CACHE: dict = {}


def kernel_root() -> Path:
    """The ``ttnn`` dir whose ``cpp/`` the stock op compiles from.

    The factory names its kernels by the relative path ``ttnn/cpp/...``, which tt-metal resolves under
    ``TT_METAL_RUNTIME_ROOT``. tt_bio points that at the metal overlay at import (tenstorrent.py:554,
    silu_f32), and the overlay PATCHES K6 (metal_overlay.BMM). Compiling the wheel's copy instead would
    put a different compute kernel under the generic op than under the stock one, so resolve the same
    root first and fall back to ttnn_cpp_root() when none is set.
    """
    rr = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if rr and (Path(rr) / "ttnn" / KDIR).is_dir():
        return Path(rr) / "ttnn"
    return ttnn_cpp_root()


def _kpath(rel: str) -> str:
    return str(kernel_root() / KDIR / rel)


def _int(v) -> int:
    try:
        return int(v)
    except TypeError:
        return int(v.value)


def _cr(x0, y0, x1, y1):
    return ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))


def _fmt(idx, dtype, page):
    return ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)


def _cb(idx, cores, dtype, page, total):
    return ttnn.CBDescriptor(total_size=total, core_ranges=cores, format_descriptors=[_fmt(idx, dtype, page)])


def _arch() -> str:
    a = str(ttnn.get_arch_name()).lower()
    assert a in ("wormhole_b0", "blackhole"), a
    return a


def _act_defines(act, out_dtype):
    """F2D:566-578 for the activations in scope; unary_op_utils.cpp:737 (SILU init/func),
    :100 (default include macro), :781-792 (get_defines_impl) at v0.68.0."""
    if act is None:
        return {}
    op = act.op_type
    assert op == ttnn.UnaryOpType.SILU, f"fused activation {op} not transcribed (SILU or None only)"
    # UnaryWithParam binds op_type only (activation.cpp:53); params show in its __repr__.
    assert "params=" not in repr(act), f"{act!r}: the parameterized path is not transcribed"
    return {"SFPU_OP_INIT_ACTIVATION": "silu_tile_init();",
            "SFPU_OP_FUNC_ACTIVATION": "silu_tile(i);",
            "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE": "1"}


def _throttle_stagger_defines(arch, n_cores, ckc):
    """compute_throttle_utils.cpp:11-90: stagger only from env, throttle from ckc.throttle_level or env,
    both only above 48 cores on WH and always on BH."""
    d = {}
    needed = (arch == "wormhole_b0" and n_cores > 48) or arch == "blackhole"
    st = os.environ.get("TT_MM_STAGGER_TYPE")
    if st and needed:
        d["MM_STAGGER_TYPE"] = st
        d["MM_STAGGER_VALUE"] = os.environ.get("TT_MM_STAGGER_VALUE", "0")
    if needed:
        lvl = _int(getattr(ckc, "throttle_level", 0) or 0)
        env = os.environ.get("TT_MM_THROTTLE_PERF")
        if env:
            lvl = int(env)
        if lvl:
            d["MM_THROTTLE"] = str(lvl if 1 <= lvl <= 5 else 0)
    return d


def build(device, in0, in1, out, pc, ckc, out_nzsb_w=None, compute_src=None, compute_defines=(),
          compute_ct_override=None):
    arch = _arch()
    math_fidelity, math_approx_mode, fp32_dest_acc_en, _ = ckc_args(ckc)
    packer_l1_acc = bool(getattr(ckc, "packer_l1_acc", False))

    # ---- program config (F2D:1560-1572) ----
    in0_block_w = int(pc.in0_block_w)
    out_subblock_h, out_subblock_w = int(pc.out_subblock_h), int(pc.out_subblock_w)
    out_block_h, out_block_w = int(pc.out_block_h), int(pc.out_block_w)
    per_core_M, per_core_N = int(pc.per_core_M), int(pc.per_core_N)
    grid = pc.compute_with_storage_grid_size
    assert not pc.transpose_mcast, "transpose_mcast=True not transcribed"
    assert pc.fuse_batch, "fuse_batch=False not transcribed"
    act = pc.fused_activation

    # ---- operands (F2D:1581-1677) ----
    for t in (in0, in1, out):
        tile = getattr(t, "tile", None)
        if tile is not None and hasattr(tile, "tile_shape"):
            assert tuple(tile.tile_shape) == (TILE_HW, TILE_HW), tile.tile_shape
        assert t.layout == ttnn.TILE_LAYOUT
    for t in (in0, in1, out):
        assert t.dtype in (ttnn.bfloat16, ttnn.bfloat8_b), t.dtype
    mc0, mc1, mco = in0.memory_config(), in1.memory_config(), out.memory_config()
    assert mc0.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc0
    assert mc1.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc1
    assert mco.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED and mco.buffer_type == ttnn.BufferType.L1, mco

    a_shape = [int(d) for d in in0.padded_shape]
    b_shape = [int(d) for d in in1.padded_shape]
    assert a_shape[-1] == b_shape[-2], (a_shape, b_shape)
    b_batch = 1
    for d in b_shape[:-2]:
        b_batch *= d
    assert b_batch == 1, "in1 with a batch: bcast_batch False is not transcribed"
    bcast_batch = 1                                          # matmul_device_operation.cpp:36-41
    B = 1                                                    # fuse_batch, F2D:1673
    M = 1
    for d in a_shape[:-1]:
        M *= d
    M //= TILE_HW                                            # get_M_dim(fuse_batch=True)
    K = a_shape[-1] // TILE_HW
    N = b_shape[-1] // TILE_HW
    in0_last_ktile_w = int(in0.shape[-1]) % TILE_HW          # F2D:1599
    in0_last_ktile_h = 0
    assert K % in0_block_w == 0, (K, in0_block_w)            # F2D:1679

    num_blocks_y = (M - 1) // per_core_M + 1                 # F2D:1686
    num_blocks_x = (N - 1) // per_core_N + 1
    assert num_blocks_x <= int(grid.x) and num_blocks_y <= int(grid.y), (num_blocks_x, num_blocks_y, grid)

    in0_dt, in1_dt, out_dt = in0.dtype, in1.dtype, out.dtype
    in0_ts, in1_ts, out_ts = tile_bytes(in0_dt), tile_bytes(in1_dt), tile_bytes(out_dt)
    if arch == "blackhole":                                  # F2D:644-655
        assert in0_ts % 64 == 0 and in1_ts % 64 == 0, "BH INTERMEDIATE_CB_READ (c_8/c_9) not transcribed"

    # ---- create_program_mcast_in0_in1 (F2D:29-) ----
    num_blocks = K // in0_block_w                                                   # F2D:82
    packer_l1_acc_en = packer_l1_acc and num_blocks > 2                             # F2D:88 (no bias)
    if packer_l1_acc_en:                                                            # F2D:91-93
        interm_dt = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    else:
        interm_dt = ttnn.float32 if fp32_dest_acc_en else out_dt
    interm_ts = tile_bytes(interm_dt)

    do_not_inplace_interm0_out_CB = per_core_M != out_block_h                       # F2D:109 (out sharded)
    in0_block_h, in1_block_w = out_block_h, out_block_w
    assert per_core_M % out_block_h == 0 and per_core_N % out_block_w == 0
    out_num_blocks_y = per_core_M // out_block_h                                    # F2D:113-116
    out_num_blocks_x = per_core_N // out_block_w

    in0_block_tiles = out_block_h * in0_block_w                                     # F2D:118-129
    in0_CB_tiles = in0_block_tiles * (2 if B * num_blocks > 1 else 1)               # MCAST_INPUT_BUFFERING_DEPTH
    in1_block_tiles = out_block_w * in0_block_w
    in1_CB_tiles = in1_block_tiles * (2 if B * num_blocks > 1 else 1)
    out_block_tiles = out_block_h * out_block_w
    out_CB_tiles = per_core_M * per_core_N                                          # F2D:131-137, sharded
    interm0_CB_tiles = out_block_tiles                                              # F2D:138

    c, r = num_blocks_x, num_blocks_y                       # num_cores_with_work_{c,r}, F2D:160-163
    all_cores_cr = _cr(0, 0, c - 1, r - 1)                  # F2D:233 (in0 interleaved: = all_cores_with_work)
    all_cores = ttnn.CoreRangeSet([all_cores_cr])
    n_cores = c * r

    # The output must be the factory's sharding of exactly this grid: c_4 is pinned to it.
    sspec = mco.shard_spec
    assert sspec is not None and sspec.orientation == ttnn.ShardOrientation.ROW_MAJOR, sspec
    bb = sspec.grid.bounding_box()
    assert (int(bb.start.x), int(bb.start.y), int(bb.end.x), int(bb.end.y)) == (0, 0, c - 1, r - 1), bb
    assert sspec.grid.num_cores() == n_cores, sspec.grid
    nzsb_w_full = out_block_w // out_subblock_w
    nzsb_w = nzsb_w_full if out_nzsb_w is None else int(out_nzsb_w)
    shard_w_tiles = per_core_N if out_nzsb_w is None else nzsb_w * out_subblock_w
    assert list(sspec.shape) == [per_core_M * TILE_HW, shard_w_tiles * TILE_HW], (sspec.shape, per_core_M,
                                                                                   shard_w_tiles)
    if out_nzsb_w is not None:
        out_CB_tiles = per_core_M * shard_w_tiles           # the CB is the shard it is pinned to

    # ---- core groups (F2D:240-307) ----
    in0_sender_interleaved = _cr(0, 0, 0, r - 1)
    in1_sender = _cr(0, 0, c - 1, 0)
    in0_sender_in1_receiver = _cr(0, 1, 0, r - 1) if r > 1 else None
    in0_receiver_in1_sender = _cr(1, 0, c - 1, 0) if c > 1 else None
    split_half = c > 2 and r > 1                            # F2D:276, in0 not sharded
    half_core = c // 2 if split_half else c - 1
    left_half = _cr(1, 1, half_core, r - 1) if (c > 1 and r > 1) else None
    in0_recv_ranges = [x for x in (in0_receiver_in1_sender, left_half) if x is not None]
    in1_recv_ranges = [x for x in (in0_sender_in1_receiver, left_half) if x is not None]
    other_cores = _cr(half_core + 1, 1, c - 1, r - 1) if split_half else None

    # ---- semaphores (F2D:310-313): ids 0..3 in creation order, all on all_cores, INVALID ----
    in0_sender_sem, in0_recv_sem, in1_sender_sem, in1_recv_sem = range(4)
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_cores, initial_value=INVALID) for i in range(4)]

    in0_num_subblocks = out_block_h // out_subblock_h                               # F2D:317-318
    in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks
    assert out_block_h % out_subblock_h == 0 and out_block_w % out_subblock_w == 0  # F2D:320-325

    in0_stride_w, in0_stride_h = 1, K                                               # F2D:343-347
    in0_next_block_stride = in0_block_w * in0_stride_w
    in0_next_h_dim_block_stride = in0_block_h * in0_stride_h
    in0_start_tile_id_stride = per_core_M * in0_stride_h
    in1_stride_w, in1_stride_h = 1, N                                               # F2D:349-353
    in1_next_block_stride = in0_block_w * in1_stride_h
    in1_next_w_dim_block_stride = in1_block_w * in1_stride_w
    in1_start_tile_id_stride = per_core_N * in1_stride_w

    ta_in0 = list(ttnn.TensorAccessorArgs(in0).get_compile_time_args())
    ta_in1 = list(ttnn.TensorAccessorArgs(in1).get_compile_time_args())
    ta_out = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    # TensorAccessorArgs() with no buffer: args_config None and aligned page size 0, TWO words
    # (tensor_accessor_args.cpp append_to, non-sharded branch; device side reads 2 for !is_sharded,
    # tensor_accessor_args.h:76).
    ta_placeholder = [0, 0]

    # K0 CT (F2D:391-431) -> reader_bmm_tile_layout_in0_sender_padding.cpp get_compile_time_arg_val(i)
    k0_ct = [
        in0_stride_w,                  # 0  in0_tensor_stride_w
        in0_stride_h,                  # 1  in0_tensor_stride_h
        in0_next_block_stride,         # 2  in0_tensor_next_inner_dim_block_stride
        in0_next_h_dim_block_stride,   # 3  in0_tensor_next_h_dim_block_stride
        in0_block_w,                   # 4  in0_block_w
        in0_block_h,                   # 5  in0_block_h
        in0_block_num_tiles,           # 6  in0_block_num_tiles
        in0_last_ktile_w,              # 7  in0_last_ktile_w
        in0_last_ktile_h,              # 8  in0_last_ktile_h
        0,                             # 9  extract_shard_sub_blocks
        0,                             # 10 shard_width_in_tiles
        0,                             # 11 shard_height_in_tiles
        num_blocks,                    # 12 num_blocks_inner_dim
        out_num_blocks_x,              # 13 num_blocks_w_dim
        out_num_blocks_y,              # 14 num_blocks_h_dim
        in0_sender_sem,                # 15 sender_sem
        in0_recv_sem,                  # 16 receiver_sem
        num_blocks_x - 1,              # 17 in0_mcast_num_dests
        num_blocks_x - 1,              # 18 in0_mcast_num_cores
        M * K,                         # 19 MtKt
        B,                             # 20 in0_B
        B,                             # 21 in1_B
        0,                             # 22 in0_reuse_in_CB
        0,                             # 23 batchB
        0,                             # 24 sparsity_pagesize
        1,                             # 25 bcast_A
        0,                             # 26 get_batch_from_reader
        0,                             # 27 fuse_op (all-gather), F2D:429
    ] + ta_in0 + ta_placeholder        # 28.. TensorAccessorArgs<28> in0, then sparsity (F2D:430-431)

    # K1 CT (F2D:433-491) -> reader_bmm_tile_layout_in1_sender_writer_padding.cpp
    k1_ct = [
        in1_stride_w,                  # 0  in1_tensor_stride_w
        in1_stride_h,                  # 1  in1_tensor_stride_h
        in1_next_block_stride,         # 2  in1_tensor_next_block_stride
        in1_next_w_dim_block_stride,   # 3  in1_tensor_next_w_dim_block_stride
        in1_block_w,                   # 4  in1_block_w
        in0_block_w,                   # 5  in1_block_h
        in1_block_w * in0_block_w,     # 6  in1_block_num_tiles
        num_blocks,                    # 7  num_blocks_inner_dim
        out_num_blocks_x,              # 8  num_blocks_w_dim
        out_num_blocks_y,              # 9  num_blocks_h_dim
        in1_sender_sem,                # 10 sender_sem
        in1_recv_sem,                  # 11 receiver_sem
        num_blocks_y - 1,              # 12 in1_mcast_num_dests
        num_blocks_y - 1,              # 13 in1_mcast_num_cores
        K * N,                         # 14 KtNt
        B,                             # 15 batch
        bcast_batch,                   # 16 bcast_B
        0,                             # 17 batchB
        0,                             # 18 sparsity_pagesize
        1,                             # 19 out_tensor_stride_w
        N,                             # 20 out_tensor_stride_h
        out_subblock_w,                # 21 out_tensor_next_subblock_stride_w
        out_subblock_h * N,            # 22 out_tensor_next_subblock_stride_h
        out_block_w,                   # 23 out_tensor_next_w_dim_block_stride
        out_block_h * N,               # 24 out_tensor_next_h_dim_block_stride
        out_subblock_w,                # 25 out_subblock_w
        out_subblock_h,                # 26 out_subblock_h
        out_subblock_w * out_subblock_h,  # 27 out_subblock_tile_count
        M * N,                         # 28 MtNt
        0,                             # 29 in3_tensor_stride_w (placeholder, no bias)
        0,                             # 30 fuse_op_all_gather
        0,                             # 31 fuse_op_reduce_scatter
    ] + ta_in1 + ta_placeholder + ta_out  # 32.. in1, sparsity placeholder, out (F2D:486-488)

    # K3/K5 CT (F2D:503-516) -> reader_bmm_tile_layout_in0_receiver.cpp
    k3_ct = [
        in0_block_w * in0_block_h,     # 0 in0_block_num_tiles
        num_blocks,                    # 1 num_blocks_inner_dim
        out_num_blocks_x,              # 2 num_blocks_w_dim
        out_num_blocks_y,              # 3 num_blocks_h_dim
        in0_sender_sem,                # 4 sender_sem
        in0_recv_sem,                  # 5 receiver_sem
        B,                             # 6 batch
        0,                             # 7 get_batch_from_reader
    ]

    # K2/K4 CT (F2D:517-552) -> reader_bmm_tile_layout_in1_receiver_writer_padding.cpp
    k2_ct = [
        in1_block_w * in0_block_w,     # 0  in1_block_num_tiles
        num_blocks,                    # 1  num_blocks_inner_dim
        out_num_blocks_x,              # 2  num_blocks_w_dim
        out_num_blocks_y,              # 3  num_blocks_h_dim
        in1_sender_sem,                # 4  sender_sem
        in1_recv_sem,                  # 5  receiver_sem
        B,                             # 6  batch
        1,                             # 7  out_tensor_stride_w
        N,                             # 8  out_tensor_stride_h
        out_subblock_w,                # 9  out_tensor_next_subblock_stride_w
        out_subblock_h * N,            # 10 out_tensor_next_subblock_stride_h
        out_block_w,                   # 11 out_tensor_next_w_dim_block_stride
        out_block_h * N,               # 12 out_tensor_next_h_dim_block_stride
        out_subblock_w,                # 13 out_subblock_w
        out_subblock_h,                # 14 out_subblock_h
        out_subblock_w * out_subblock_h,  # 15 out_subblock_tile_count
        M * N,                         # 16 MtNt
        0,                             # 17 in3_block_w (placeholder, no bias)
        0,                             # 18 fuse_op_reduce_scatter
    ] + ta_out                         # 19.. TensorAccessorArgs<19> out

    # K6 CT (F2D:817-848) -> bmm_large_block_zm_fused_bias_activation.cpp
    in1_num_subblocks = out_block_w // out_subblock_w
    k6_ct = [
        in0_block_w,                                     # 0  in0_block_w
        in0_num_subblocks,                               # 1  in0_num_subblocks
        in0_block_num_tiles,                             # 2  in0_block_num_tiles
        out_subblock_h * in0_block_w,                    # 3  in0_subblock_num_tiles
        in1_num_subblocks,                               # 4  in1_num_subblocks
        out_subblock_w * in0_block_w * in1_num_subblocks,  # 5  in1_block_num_tiles
        out_subblock_w * in1_num_subblocks,              # 6  in1_block_w (in1_per_core_w)
        num_blocks,                                      # 7  num_blocks_inner_dim
        out_num_blocks_x,                                # 8  num_blocks_w_dim
        out_num_blocks_y,                                # 9  num_blocks_h_dim
        out_subblock_h,                                  # 10 out_subblock_h
        out_subblock_w,                                  # 11 out_subblock_w
        out_subblock_h * out_subblock_w,                 # 12 out_subblock_num_tiles
        B,                                               # 13 batch
        out_block_tiles,                                 # 14 out_block_num_tiles
        0,                                               # 15 untilize_out
        0,                                               # 16 get_batch_from_reader
        0,                                               # 17 in0_transpose_tile
    ]
    if compute_ct_override is not None:
        if isinstance(compute_ct_override, dict):
            for i, v in compute_ct_override.items():
                k6_ct[int(i)] = int(v)
        else:
            k6_ct = [int(v) for v in compute_ct_override]

    # ---- defines (F2D:554-620) ----
    mm_defines = dict(_act_defines(act, out_dt))
    if packer_l1_acc_en:
        mm_defines["PACKER_L1_ACC"] = "1"
    if fp32_dest_acc_en:
        mm_defines["FP32_DEST_ACC_EN"] = "1"
    mm_defines.update(_throttle_stagger_defines(arch, n_cores, ckc))               # F2D:589-592
    mm_defines.update({str(k): str(v) for k, v in dict(compute_defines).items()})
    in0_sender_defines = {} if in0_recv_ranges else {"SKIP_MCAST": "1"}           # F2D:594-596
    in1_sender_defines = {"OUT_SHARDED": "1"}                                       # F2D:616-618
    if not in1_recv_ranges:
        in1_sender_defines["SKIP_MCAST"] = "1"                                      # F2D:601-603
    in1_recv_defines = {"OUT_SHARDED": "1"}

    # ---- NOCs (F2D:658-661) ----
    in0_noc, in1_noc = NOC_FOR_DRAM_WRITE, NOC_FOR_DRAM_READ
    in0_split_noc, in1_split_noc = NOC_FOR_DRAM_READ, NOC_FOR_DRAM_WRITE

    # ---- CBs (F2D:875-1031), in the factory's creation order ----
    cbs = [_cb(0, all_cores, in0_dt, in0_ts, in0_CB_tiles * in0_ts),
           _cb(1, all_cores, in1_dt, in1_ts, in1_CB_tiles * in1_ts)]
    out_cb_size = out_CB_tiles * out_ts
    out_cb = ttnn.cb_descriptor_from_sharded_tensor(4, out, total_size=out_cb_size, core_ranges=all_cores)
    if do_not_inplace_interm0_out_CB or interm_dt != out_dt:                        # F2D:942-966
        cbs.append(_cb(5, all_cores, interm_dt, interm_ts, interm0_CB_tiles * interm_ts))
    else:                                                                           # F2D:967-976, shared
        out_cb.format_descriptors = list(out_cb.format_descriptors) + [_fmt(5, interm_dt, interm_ts)]
    out_cb_pos = len(cbs)
    cbs.append(out_cb)

    # ---- per-core runtime args (F2D:1033-1432) ----
    last_per_core_M = M % per_core_M or per_core_M
    last_per_core_N = N % per_core_N or per_core_N
    last_out_block_h = last_per_core_M % out_block_h or out_block_h
    last_out_block_w = last_per_core_N % out_block_w or out_block_w
    last_block_num_nonzero_subblocks_h = (last_out_block_h - 1) // out_subblock_h + 1
    last_block_num_nonzero_subblocks_w = (last_out_block_w - 1) // out_subblock_w + 1
    last_subblock_of_last_block_h = last_out_block_h % out_subblock_h or out_subblock_h
    last_subblock_of_last_block_w = last_out_block_w % out_subblock_w or out_subblock_w
    last_block_padded_subblock_tiles_addr_skip = out_ts * (out_subblock_w - last_subblock_of_last_block_w)
    last_block_padded_block_tiles_w_skip = (out_subblock_w * out_subblock_h) * (
        out_block_w // out_subblock_w - last_block_num_nonzero_subblocks_w)
    last_block_padded_block_tiles_h_skip = (out_block_h // out_subblock_h - last_block_num_nonzero_subblocks_h) * (
        out_block_w * out_subblock_h)
    in0_end_idx, in1_end_idx = num_blocks_y - 1, num_blocks_x - 1

    def phys(x, y):
        p = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return int(p.x), int(p.y)

    in0_addr, in1_addr, out_addr = in0.buffer_address(), in1.buffer_address(), out.buffer_address()
    rt = {n: [] for n in ("k0", "k1", "k2", "k3", "k4", "k5")}
    for cy in range(r):                       # grid_to_cores(row_wise=true); order is immaterial
        for cx in range(c):
            cc = ttnn.CoreCoord(cx, cy)
            left, left1, right = phys(0, cy), phys(1, cy), phys(c - 1, cy)
            top, top1, bottom = phys(cx, 0), phys(cx, 1), phys(cx, r - 1)
            in0_idx, in1_idx = cy, cx
            in0_mcast_start, in0_mcast_end = left1, right
            if in0_noc == ttnn.NOC.NOC_1:                                           # F2D:1103-1105
                in0_mcast_start, in0_mcast_end = in0_mcast_end, in0_mcast_start
            in1_mcast_start, in1_mcast_end = bottom, top1
            if in1_noc == ttnn.NOC.NOC_0:                                           # F2D:1110-1112
                in1_mcast_start, in1_mcast_end = in1_mcast_end, in1_mcast_start

            if in1_idx == 0:                  # K0 RT (F2D:1154-1179) -> in0_sender_padding.cpp:23-34
                rt["k0"].append((cc, [
                    in0_addr,                                    # 0 in0_tensor_addr
                    in0_start_tile_id_stride * in0_idx,          # 1 in0_tensor_start_tile_id
                    in0_mcast_start[0], in0_mcast_start[1],      # 2,3 in0_mcast_dest_noc_start_x/y
                    in0_mcast_end[0], in0_mcast_end[1],          # 4,5 in0_mcast_dest_noc_end_x/y
                    last_out_block_h if in0_idx == in0_end_idx else out_block_h,  # 6 last_block_h
                    0,                                           # 7 sparsity_addr
                ]))
            else:                             # K3/K5 RT (F2D:1182-1196) -> in0_receiver.cpp:17-18
                args = [left[0], left[1]]
                rt["k3" if (cx <= half_core or cy == 0) else "k5"].append((cc, args))

            if in0_idx == 0:                  # K1 RT (F2D:1201-1259) -> in1_sender_writer_padding.cpp:21-45,111
                last_w = in1_idx == in1_end_idx
                rt["k1"].append((cc, [
                    in1_addr,                                    # 0  in1_tensor_addr
                    in1_start_tile_id_stride * in1_idx,          # 1  in1_tensor_start_tile_id
                    in1_mcast_start[0], in1_mcast_start[1],      # 2,3 in1_mcast_dest_noc_start_x/y
                    in1_mcast_end[0], in1_mcast_end[1],          # 4,5 in1_mcast_dest_noc_end_x/y
                    0,                                           # 6  sparsity_addr
                    out_addr,                                    # 7  out_tensor_addr
                    in1_idx * per_core_N + in0_idx * per_core_M * N,  # 8 out_tensor_start_tile_id
                    last_out_block_w if last_w else out_block_w,  # 9 last_block_w
                    out_block_h // out_subblock_h,               # 10 out_num_nonzero_subblocks_h
                    out_subblock_h,                              # 11 out_last_subblock_h
                    0,                                           # 12 padded_block_tiles_h_skip
                    nzsb_w_full,                                 # 13 out_num_nonzero_subblocks_w
                    last_block_num_nonzero_subblocks_w if last_w else nzsb_w_full,  # 14 out_last_num_nonzero_sb_w
                    last_subblock_of_last_block_w if last_w else out_subblock_w,  # 15 out_last_subblock_w
                    last_block_padded_subblock_tiles_addr_skip if last_w else 0,  # 16 padded_subblock_tiles_addr_skip
                    last_block_padded_block_tiles_w_skip if last_w else 0,  # 17 padded_block_tiles_w_skip
                    0,                                           # 18 bias addr (skipped: rt_args_idx += 2)
                    0,                                           # 19 bias start tile id
                ]))                                              # out sharded: no last_num_blocks_w_dim
            else:                             # K2/K4 RT (F2D:1342-1429) -> in1_receiver_writer_padding.cpp:19-37
                last_h, last_w = in0_idx == in0_end_idx, in1_idx == in1_end_idx
                args = [
                    top[0], top[1],                              # 0,1 in1_mcast_sender_noc_x/y
                    out_addr,                                    # 2 out_tensor_addr
                    in1_idx * per_core_N + in0_idx * per_core_M * N,  # 3 out_tensor_start_tile_id
                    out_block_h // out_subblock_h,               # 4 out_num_nonzero_subblocks_h
                    last_block_num_nonzero_subblocks_h if last_h else out_block_h // out_subblock_h,  # 5
                    last_subblock_of_last_block_h if last_h else out_subblock_h,  # 6 out_last_subblock_h
                    last_block_padded_block_tiles_h_skip if last_h else 0,  # 7 padded_block_tiles_h_skip
                    nzsb_w_full,                                 # 8 out_num_nonzero_subblocks_w
                    last_block_num_nonzero_subblocks_w if last_w else nzsb_w_full,  # 9 out_last_num_nonzero_sb_w
                    last_subblock_of_last_block_w if last_w else out_subblock_w,  # 10 out_last_subblock_w
                    last_block_padded_subblock_tiles_addr_skip if last_w else 0,  # 11 padded_subblock_tiles_addr_skip
                    last_block_padded_block_tiles_w_skip if last_w else 0,  # 12 padded_block_tiles_w_skip
                ]                                                # out sharded: no last_num_blocks_{h,w}_dim
                rt["k2" if cx <= half_core else "k4"].append((cc, args))

    if out_nzsb_w is not None:
        for _, a in rt["k1"]:
            for i in RT_IN1_SENDER_NZSB_W:
                a[i] = nzsb_w
        for name in ("k2", "k4"):
            for _, a in rt[name]:
                for i in RT_IN1_RECV_NZSB_W:
                    a[i] = nzsb_w

    # ---- kernels, in the factory's creation order (F2D:712-873) ----
    def dm(src, ranges, ct, named, defines, args, risc, noc):
        return ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(ranges), compile_time_args=ct,
            named_compile_time_args=list(named.items()), defines=list(defines.items()), runtime_args=args,
            config=ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc))

    R0, R1 = ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1
    named_k0 = {"cb_in0": 0, "cb_in0_sharded": 2, "cb_sparsity": 6, "cb_in0_intermediate": 8}  # F2D:721-726
    named_k1 = {"cb_in1": 1, "cb_bias": 3, "cb_out": 4, "cb_sparsity": 7, "cb_in1_intermediate": 9}  # F2D:738-744
    named_k2 = {"cb_in1": 1, "cb_bias": 3, "cb_out": 4}                                       # F2D:759-763
    named_k3 = {"cb_in0": 0}                                                                  # F2D:777-779
    named_k6 = {"cb_in0": 0, "cb_in1": 1, "cb_bias": 3, "cb_out": 4, "cb_intermed0": 5,
                "cb_in0_intermediate": 8, "cb_in1_intermediate": 9, "cb_in0_transposed": 10}  # F2D:864-873

    kernels, knames = [], []

    def add(name, kd):
        kernels.append(kd)
        knames.append(name)

    add("k0", dm(_kpath(K_IN0_SENDER), [in0_sender_interleaved], k0_ct, named_k0, in0_sender_defines,
                 rt["k0"], R1, in0_noc))
    add("k1", dm(_kpath(K_IN1_SENDER), [in1_sender], k1_ct, named_k1, in1_sender_defines, rt["k1"], R0, in1_noc))
    if in1_recv_ranges:
        add("k2", dm(_kpath(K_IN1_RECV), in1_recv_ranges, k2_ct, named_k2, in1_recv_defines, rt["k2"], R0, in1_noc))
    if in0_recv_ranges:
        add("k3", dm(_kpath(K_IN0_RECV), in0_recv_ranges, k3_ct, named_k3, {}, rt["k3"], R1, in0_noc))
    if other_cores is not None:
        add("k4", dm(_kpath(K_IN1_RECV), [other_cores], k2_ct, named_k2, in1_recv_defines, rt["k4"], R0,
                     in1_split_noc))
        add("k5", dm(_kpath(K_IN0_RECV), [other_cores], k3_ct, named_k3, {}, rt["k5"], R1, in0_split_noc))
    for name in ("k2", "k3", "k4", "k5"):
        assert rt[name] == [] or name in knames, (name, "has runtime args but no kernel")
    if compute_src is None:
        csrc = _kpath(K_COMPUTE)
    else:
        p = Path(compute_src)
        csrc = str(p if p.is_absolute() or p.exists() else kernel_root() / KDIR / p)
    # ComputeConfig gets fidelity, fp32 acc and approx only (F2D:858-863); dst_full_sync_en,
    # unpack_to_dest_mode and bfp8_pack_precise stay at their defaults.
    add("k6", ttnn.KernelDescriptor(
        kernel_source=csrc, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=all_cores, compile_time_args=k6_ct, named_compile_time_args=list(named_k6.items()),
        defines=list(mm_defines.items()), runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, math_approx_mode=math_approx_mode,
                                            fp32_dest_acc_en=fp32_dest_acc_en)))

    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    return {"pd": pd, "kernels": kernels, "knames": knames, "cbs": cbs, "semaphores": semaphores, "rt": rt,
            "out_cb_pos": out_cb_pos, "out_cb_size": out_cb_size, "all_cores": all_cores,
            "interm_fmt": None if (do_not_inplace_interm0_out_CB or interm_dt != out_dt) else (interm_dt, interm_ts),
            "addrs": (in0_addr, in1_addr, out_addr), "out": out,
            "info": {"M": M, "K": K, "N": N, "num_blocks": num_blocks, "grid": (c, r), "split_half": split_half,
                     "half_core": half_core, "packer_l1_acc_en": packer_l1_acc_en, "interm": str(interm_dt),
                     "interm_separate": do_not_inplace_interm0_out_CB or interm_dt != out_dt,
                     "nzsb_w": nzsb_w, "defines": mm_defines, "kernel_root": str(kernel_root()),
                     "k0_ct": k0_ct, "k1_ct": k1_ct, "k2_ct": k2_ct, "k3_ct": k3_ct, "k6_ct": k6_ct}}


def _ckc_key(ckc):
    return tuple(str(getattr(ckc, f, None)) for f in
                 ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc",
                  "dst_full_sync_en", "throttle_level"))


def _pc_key(pc):
    g = pc.compute_with_storage_grid_size
    act = pc.fused_activation
    return (int(g.x), int(g.y), int(pc.in0_block_w), int(pc.out_subblock_h), int(pc.out_subblock_w),
            int(pc.out_block_h), int(pc.out_block_w), int(pc.per_core_M), int(pc.per_core_N),
            bool(pc.transpose_mcast), bool(pc.fuse_batch),
            None if act is None else repr(act))


def _key(in0, in1, out, pc, ckc, out_nzsb_w, compute_src, compute_defines, compute_ct_override):
    spec = lambda t: (str(t.padded_shape), str(t.shape), str(t.dtype), str(t.memory_config()))
    ov = compute_ct_override
    if isinstance(ov, dict):
        ov = tuple(sorted((int(k), int(v)) for k, v in ov.items()))
    elif ov is not None:
        ov = tuple(int(v) for v in ov)
    env = tuple(os.environ.get(k) for k in ("TT_MM_STAGGER_TYPE", "TT_MM_STAGGER_VALUE", "TT_MM_THROTTLE_PERF"))
    return (spec(in0), spec(in1), spec(out), _pc_key(pc), _ckc_key(ckc), out_nzsb_w,
            None if compute_src is None else str(compute_src),
            tuple(sorted((str(k), str(v)) for k, v in dict(compute_defines).items())), ov, env, str(kernel_root()))


def generic_matmul_2d(device, in0, in1, out, program_config, compute_kernel_config, *, out_nzsb_w=None,
                      compute_src=None, compute_defines=(), compute_ct_override=None):
    """``ttnn.linear(in0, in1, program_config=..., compute_kernel_config=..., memory_config=out's,
    dtype=out's)`` written into the pre-allocated block-sharded ``out``, through ``generic_op``.

    ``out_nzsb_w`` replaces the writers' out_num_nonzero_subblocks_w (and its last-block twin), which under
    OUT_SHARDED only sets how many tiles they wait for at the end; ``out`` must then be sharded
    [per_core_M, out_nzsb_w * out_subblock_w] tiles and c_4 is pinned to that. ``compute_src`` (a path,
    absolute or relative to the matmul kernels dir), ``compute_defines`` (merged over the factory's) and
    ``compute_ct_override`` (a full list, or {index: value}) swap K6 for a patched kernel.
    """
    key = _key(in0, in1, out, program_config, compute_kernel_config, out_nzsb_w, compute_src, compute_defines,
               compute_ct_override)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = build(device, in0, in1, out, program_config, compute_kernel_config, out_nzsb_w,
                                    compute_src, compute_defines, compute_ct_override)
    addrs = (in0.buffer_address(), in1.buffer_address(), out.buffer_address())
    if addrs != entry["addrs"] or out is not entry["out"]:
        rebind(entry, in0, in1, out)
    ttnn.generic_op([in0, in1, out], entry["pd"])
    return out


def rebind(entry, in0, in1, out):
    """New buffer addresses into the cached runtime args (the writes F2D:1497-1536 makes), and c_4
    re-pinned to ``out``.

    The pinned CB descriptor carries a raw Buffer pointer, so it is rebuilt whenever ``out`` is a
    different tensor object, not only when the address moves; the entry keeps a reference to the
    ``out`` it was pinned to so that pointer cannot dangle under it.
    """
    in0_addr, in1_addr, out_addr = in0.buffer_address(), in1.buffer_address(), out.buffer_address()
    rt = entry["rt"]
    for _, a in rt["k0"]:
        a[RT_IN0_SENDER_ADDR] = in0_addr
    for _, a in rt["k1"]:
        a[RT_IN1_SENDER_ADDR] = in1_addr
        a[RT_IN1_SENDER_OUT] = out_addr
    for name in ("k2", "k4"):
        for _, a in rt[name]:
            a[RT_IN1_RECV_OUT] = out_addr
    for kd, name in zip(entry["kernels"], entry["knames"]):
        if name in ("k0", "k1", "k2", "k4"):
            kd.runtime_args = rt[name]
    if out is not entry["out"]:
        cb = ttnn.cb_descriptor_from_sharded_tensor(4, out, total_size=entry["out_cb_size"],
                                                    core_ranges=entry["all_cores"])
        if entry["interm_fmt"] is not None:
            cb.format_descriptors = list(cb.format_descriptors) + [_fmt(5, *entry["interm_fmt"])]
        entry["cbs"][entry["out_cb_pos"]] = cb
        entry["out"] = out
    entry["pd"] = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=entry["semaphores"],
                                         cbs=entry["cbs"])
    entry["addrs"] = (in0_addr, in1_addr, out_addr)
