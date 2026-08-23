"""`ttnn.transformer.scaled_dot_product_attention` re-driven through `ttnn.generic_op`.

A transcription of `sdpa_program_factory.cpp` at the `v0.68.0` tag (the wheel's kernel sources are
byte-identical to that tag) into a Python `ttnn.ProgramDescriptor`, for exactly the call triangle
attention makes and nothing else:

    non-causal, provided mask, no chunking, no paging, no MLA, no attention sink, bf16 in/out,
    interleaved DRAM, one q chunk per core.

Under those conditions two of the factory's three hard parts vanish, both checked rather than
assumed (`state/triatt-fused-kernel-final.md` §11):

  * `can_use_streaming_compute` returns false the moment a mask is provided, so the streaming
    compute path and its lightweight mask are dead.
  * KV chain forwarding is enabled for every non-causal call and lines 811-1279 build its topology,
    but chains group by `(batch, head)` and are skipped below two segments. Every `(batch, head)` is
    owned by one core here, so the device prints `Multicast eligibility: 0/0 chains using mcast`
    and every chain runtime arg is its no-chain default.

What is left is the CB table, three kernels and a work split. This module exists so K2 can change
the work split to head-contiguous and hold the mask in a permanently fronted CB; on its own, with
the wheel's kernels unmodified, it must reproduce the native op bit-exactly and at the same speed.
That is the S4 gate.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

INVALID, VALID = 0, 1
TILE = 32
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}
# What TensorAccessorArgs(nullptr) appends: ArgConfig::None (0) and a zero page size.
NULL_ACCESSOR = [0, 0]

_CACHE: dict = {}


def _kdir():
    from .mm_generic import ttnn_cpp_root
    return ttnn_cpp_root() / "cpp/ttnn/operations/transformer/sdpa/device/kernels"


def _div_up(a, b):
    return (a + b - 1) // b


def largest_subblock(block_h, block_w, dst_size, max_h=None):
    """`detail::determine_largest_subblock_size`, candidate order included."""
    cands = [(2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1),
             (1, 5), (5, 1), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1)]
    for h, w in cands:
        if h * w > dst_size:
            continue
        if max_h is not None and h > max_h:
            continue
        if block_h % h or block_w % w:
            continue
        return h, w
    return 1, 1


def valid_granularity(tile_count, max_granularity):
    """`detail::find_valid_granularity`."""
    g = min(tile_count, max_granularity)
    while g > 1 and tile_count % g:
        g -= 1
    return g


def dht_granularity(DHt, vDHt, dst_size):
    """`compute_dht_granularity`."""
    g = min(DHt, vDHt, dst_size)
    while g > 1 and (DHt % g or vDHt % g):
        g -= 1
    return g


def _f32_bits(x):
    import struct
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _packed_identity_scalar():
    """`pack_two_bfloat16_into_uint32({1.0f, 1.0f})` -- bf16 1.0 is 0x3F80."""
    return (0x3F80 << 16) | 0x3F80


def plan(q, k, v, mask, out, q_chunk_size, k_chunk_size, grid, ckc, scale, split=None,
         kv_buffer_factor=2):
    """Everything the descriptor needs, derived exactly as the factory derives it.

    `split` overrides `(batch_parallel_factor, nh_parallel_factor, q_parallel_factor)`. The
    factory's own choice saturates batch first -- `batch_pf = min(B, num_cores)` -- which at
    B = 512 on 110 cores leaves `nh_pf = 1` and hands every core all 8 heads. K2 needs one head per
    core, i.e. `(num_cores // NQH, NQH, 1)`, and the per-core index arithmetic is unchanged.
    """
    gx, gy = grid
    num_cores = gx * gy
    B, NQH, Sq, DH = (int(d) for d in q.padded_shape)
    NKH = int(k.padded_shape[1])
    NVH = int(v.padded_shape[1])
    Sk = int(k.padded_shape[2])

    padded_Sq = _div_up(Sq, q_chunk_size) * q_chunk_size
    padded_Sk = _div_up(Sk, k_chunk_size) * k_chunk_size
    Sqt, Skt = padded_Sq // TILE, padded_Sk // TILE
    DHt = DH // TILE
    vDHt = DHt
    valid_Sqt, valid_Skt = _div_up(Sq, TILE), _div_up(Sk, TILE)
    use_padded_mask = (padded_Sk != Sk) or (padded_Sq != Sq)
    Sq_chunk_t, Sk_chunk_t = q_chunk_size // TILE, k_chunk_size // TILE
    q_num_chunks, k_num_chunks = padded_Sq // q_chunk_size, padded_Sk // k_chunk_size

    mshape = [int(d) for d in mask.shape]
    bcast_batch = mshape[0] == 1
    bcast_heads = mshape[1] == 1

    if split is None:
        batch_pf = min(B, num_cores)
        nh_pf = min(num_cores // batch_pf, NQH)
        q_pf = min(num_cores // (batch_pf * nh_pf), q_num_chunks)
    else:
        batch_pf, nh_pf, q_pf = split
        assert batch_pf * nh_pf * q_pf <= num_cores, (split, num_cores)
    batch_per_core = _div_up(B, batch_pf)
    nh_per_core = _div_up(NQH, nh_pf)
    q_per_core = _div_up(q_num_chunks, q_pf)
    q_buffer_factor = 2 if q_per_core > 1 else 1

    math_fidelity, math_approx, fp32_dest_acc, dst_full_sync = ckc
    dst_size = 4 if fp32_dest_acc else 8
    qk_in0_block_w = DHt
    qk_sb_h, qk_sb_w = largest_subblock(Sq_chunk_t, Sk_chunk_t, dst_size)
    qk_in0_ns, qk_in1_ns = Sq_chunk_t // qk_sb_h, Sk_chunk_t // qk_sb_w
    qk_num_blocks = DHt // qk_in0_block_w
    out_in0_block_w = Sk_chunk_t
    out_sb_h, out_sb_w = largest_subblock(Sq_chunk_t, vDHt, dst_size)
    out_in0_ns, out_in1_ns = Sq_chunk_t // out_sb_h, vDHt // out_sb_w
    out_num_blocks = Sk_chunk_t // out_in0_block_w

    return dict(
        gx=gx, gy=gy, num_cores=num_cores, B=B, NQH=NQH, NKH=NKH, NVH=NVH, Sq=Sq, Sk=Sk, DH=DH,
        Sqt=Sqt, Skt=Skt, DHt=DHt, vDHt=vDHt, valid_Sqt=valid_Sqt, valid_Skt=valid_Skt,
        use_padded_mask=use_padded_mask, Sq_chunk_t=Sq_chunk_t, Sk_chunk_t=Sk_chunk_t,
        q_num_chunks=q_num_chunks, k_num_chunks=k_num_chunks,
        bcast_batch=bcast_batch, bcast_heads=bcast_heads,
        batch_pf=batch_pf, nh_pf=nh_pf, q_pf=q_pf,
        batch_per_core=batch_per_core, nh_per_core=nh_per_core, q_per_core=q_per_core,
        q_buffer_factor=q_buffer_factor, dst_size=dst_size,
        qk_in0_block_w=qk_in0_block_w, qk_sb_h=qk_sb_h, qk_sb_w=qk_sb_w,
        qk_in0_ns=qk_in0_ns, qk_in1_ns=qk_in1_ns, qk_num_blocks=qk_num_blocks,
        out_in0_block_w=out_in0_block_w, out_sb_h=out_sb_h, out_sb_w=out_sb_w,
        out_in0_ns=out_in0_ns, out_in1_ns=out_in1_ns, out_num_blocks=out_num_blocks,
        math_fidelity=math_fidelity, math_approx=math_approx, fp32_dest_acc=fp32_dest_acc,
        dst_full_sync=dst_full_sync, scale=scale,
        # the CB tile counts, straight off :405-414
        q_tiles=Sq_chunk_t * DHt * q_buffer_factor,
        k_tiles=Sk_chunk_t * DHt * kv_buffer_factor,
        v_tiles=Sk_chunk_t * vDHt * kv_buffer_factor,
        mask_tiles=Sq_chunk_t * Sk_chunk_t * 2,
        qk_tiles=Sq_chunk_t * Sk_chunk_t,
        out_im_tiles=Sq_chunk_t * vDHt,
        out0_t=Sq_chunk_t * vDHt,
        statistics_tiles=Sq_chunk_t,
    )


def build(device, q, k, v, mask, out, q_chunk_size, k_chunk_size, grid, ckc, scale,
          exp_approx_mode=False, mask_cb_tiles=None, defines_extra=None, kernel_dir=None,
          split=None, kv_buffer_factor=2):
    """The ProgramDescriptor for the fold's SDPA call.

    `mask_cb_tiles` overrides the size of `cb_mask_in` (K2 makes it the whole head's grid instead of
    a double-buffered chunk; at the shipped config those are the same 256 tiles). `kernel_dir` swaps
    in patched kernel sources. With both left alone this is the wheel's own program.
    """
    p = plan(q, k, v, mask, out, q_chunk_size, k_chunk_size, grid, ckc, scale, split,
             kv_buffer_factor)
    gx, gy, num_cores = p["gx"], p["gy"], p["num_cores"]
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    q_ts = _TILE_BYTES[q.dtype]
    k_ts = _TILE_BYTES[k.dtype]
    v_ts = _TILE_BYTES[v.dtype]
    mask_ts = _TILE_BYTES[mask.dtype]
    out_ts = _TILE_BYTES[out.dtype]
    im_df, stats_df, scalar_df = ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16   # :651-653, always bf16
    im_ts = stats_ts = scalar_ts = 2048

    def cb(idx, n_tiles, page, fmt):
        f = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)
        return ttnn.CBDescriptor(total_size=n_tiles * page, core_ranges=core_grid,
                                 format_descriptors=[f])

    nmask = p["mask_tiles"] if mask_cb_tiles is None else mask_cb_tiles
    cbs = [
        cb(0, p["q_tiles"], q_ts, q.dtype),
        cb(1, p["k_tiles"], k_ts, k.dtype),
        cb(2, p["v_tiles"], v_ts, v.dtype),
        cb(3, nmask, mask_ts, mask.dtype),
        cb(5, 1, scalar_ts, scalar_df),
        cb(7, 1, scalar_ts, scalar_df),
        cb(4, 1, im_ts, im_df),                       # c_recip_scratch, :746 (no attention sink)
        cb(24, p["qk_tiles"], im_ts, im_df),
        cb(25, p["out_im_tiles"], im_ts, im_df),
        cb(26, p["out_im_tiles"], im_ts, im_df),
        cb(27, p["statistics_tiles"], stats_ts, stats_df),
        cb(28, p["statistics_tiles"], stats_ts, stats_df),
        cb(29, p["statistics_tiles"], stats_ts, stats_df),
        cb(30, p["statistics_tiles"], stats_ts, stats_df),
        cb(31, p["statistics_tiles"], stats_ts, stats_df),
        cb(16, p["out0_t"], out_ts, out.dtype),
    ]

    # Three semaphores, created for every non-causal call (:539), ids 0..2 in creation order.
    semaphores = [
        ttnn.SemaphoreDescriptor(id=0, core_ranges=core_grid, initial_value=INVALID),
        ttnn.SemaphoreDescriptor(id=1, core_ranges=core_grid, initial_value=INVALID),
        ttnn.SemaphoreDescriptor(id=2, core_ranges=core_grid, initial_value=VALID),
    ]

    def acc(t):
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_ct = [
        p["B"], p["NQH"], p["NKH"], p["NVH"], p["Sqt"], p["Skt"], p["valid_Sqt"], p["valid_Skt"],
        p["DHt"], p["vDHt"], p["Sq_chunk_t"], p["q_num_chunks"], p["Sk_chunk_t"],
        p["k_num_chunks"], num_cores,
        0,                                   # is_causal
        1,                                   # use_provided_mask
        int(p["bcast_batch"]), int(p["bcast_heads"]), int(p["use_padded_mask"]),
        0,                                   # is_chunked
        0,                                   # block_size_t
        0,                                   # page_table_stick_size
        0,                                   # use_attention_sink
        0,                                   # use_mla
        0,                                   # mla_kv_overlap
        p["qk_sb_h"],
        0, 1, 2,                             # sender / receiver / valid semaphore ids
        0,                                   # mcast_enabled
    ]
    # page_table, attention_sink and chunk_start_idx are all absent here. The factory still
    # appends an accessor for each (:522-530), and a null buffer appends exactly
    # [ArgConfig::None.raw(), 0] = [0, 0] -- tensor_accessor_args.cpp:128-131 and :179-186,
    # arg_config.hpp:18. The Python binding only takes a Tensor, so they are spelled out.
    reader_ct += acc(q) + acc(k) + acc(v) + acc(mask) + NULL_ACCESSOR * 3

    writer_ct = [
        p["B"], p["NQH"], p["NKH"], p["Sqt"], p["valid_Sqt"], p["Sk"], p["DHt"], p["vDHt"],
        p["Sq_chunk_t"], p["q_num_chunks"], p["Sk_chunk_t"], p["k_num_chunks"],
        _packed_identity_scalar(), _f32_bits(p["scale"]), num_cores,
        0,                                   # is_causal
        1,                                   # use_provided_mask
        int(p["use_padded_mask"]),
        0,                                   # is_chunked
        0,                                   # sliding_window_size
        0,                                   # lightweight mask (streaming only)
    ] + acc(out)

    compute_ct = [
        p["B"], p["NQH"], p["NKH"], p["Skt"], p["DHt"], p["vDHt"], p["Sq_chunk_t"],
        p["q_num_chunks"], p["Sk_chunk_t"], p["k_num_chunks"],
        p["qk_in0_block_w"], p["qk_sb_w"], p["qk_sb_h"], p["qk_in0_ns"], p["qk_in1_ns"],
        p["qk_num_blocks"],
        p["out_in0_block_w"], p["out_sb_w"], p["out_sb_h"], p["out_in0_ns"], p["out_in1_ns"],
        p["out_num_blocks"],
        num_cores,
        0,                                   # is_causal
        1,                                   # use_provided_mask
        int(p["use_padded_mask"]),
        0,                                   # is_chunked
        _f32_bits(p["scale"]),
        0,                                   # sliding_window_size
        0,                                   # use_attention_sink
        0,                                   # use_streaming_compute
        p["valid_Skt"],
        int(_uniform_dataformat(q, k, v, out, mask)),
    ] + acc(out)

    ds = p["dst_size"]
    defines = {
        "STATS_GRANULARITY": str(valid_granularity(p["Sq_chunk_t"], ds)),
        "SUB_EXP_GRANULARITY": str(valid_granularity(p["Sk_chunk_t"], ds)),
        "MUL_BCAST_GRANULARITY": str(valid_granularity(p["Sq_chunk_t"] * p["Sk_chunk_t"], ds)),
        "DHT_GRANULARITY": str(dht_granularity(p["DHt"], p["vDHt"], ds)),
        "REDUCE_GRANULARITY": str(valid_granularity(p["Sq_chunk_t"], ds // 2)),
        "EXP_APPROX_MODE": str(int(exp_approx_mode)),
    }
    if defines_extra:
        defines.update({str(a): str(b) for a, b in dict(defines_extra).items()})
    dlist = sorted(defines.items())

    kd = kernel_dir or _kdir()
    stock = _kdir()
    reader_src = str(Path(kd) / "dataflow/reader_interleaved.cpp")
    writer_src = str(stock / "dataflow/writer_interleaved.cpp")
    compute_src = str(Path(kd) / "compute/sdpa.cpp")
    if not Path(reader_src).exists():
        reader_src = str(stock / "dataflow/reader_interleaved.cpp")
    if not Path(compute_src).exists():
        compute_src = str(stock / "compute/sdpa.cpp")

    q_a, k_a, v_a = q.buffer_address(), k.buffer_address(), v.buffer_address()
    m_a, o_a = mask.buffer_address(), out.buffer_address()

    rr, wr, cr = [], [], []
    for i in range(num_cores):
        core = ttnn.CoreCoord(i % gx, i // gx)
        lb = min((i // (p["nh_pf"] * p["q_pf"])) * p["batch_per_core"], p["B"])
        lbe = min(lb + p["batch_per_core"], p["B"])
        ln = min(((i // p["q_pf"]) % p["nh_pf"]) * p["nh_per_core"], p["NQH"])
        lne = min(ln + p["nh_per_core"], p["NQH"])
        lq = min((i % p["q_pf"]) * p["q_per_core"], p["q_num_chunks"])
        lqe = min(lq + p["q_per_core"], p["q_num_chunks"])
        # num_phases=1, chunked_q_chunk_offset=0, read/write_offset=0 (:807-809)
        rr.append((core, [q_a, k_a, v_a, m_a, 0, 0, 0, i, lb, lbe, ln, lne, lq, lqe, 1, 0, 0]
                   + [0] * 14))                        # chain metadata, all no-chain (0/0 chains)
        wr.append((core, [o_a, i, lb, lbe, ln, lne, lq, lqe, 1, 0, 0, 0]))
        cr.append((core, [i, lb, lbe, ln, lne, lq, lqe, 1, 0, 0]))

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=reader_src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=reader_ct, defines=dlist, runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=writer_src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=writer_ct, defines=dlist, runtime_args=wr,
            config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=compute_src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=compute_ct, defines=dlist, runtime_args=cr,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=p["math_fidelity"], math_approx_mode=p["math_approx"],
                fp32_dest_acc_en=p["fp32_dest_acc"], dst_full_sync_en=p["dst_full_sync"])),
    ]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    return {"pd": pd, "kernels": kernels, "cbs": cbs, "semaphores": semaphores, "plan": p,
            "rt": (rr, wr, cr), "addrs": (q_a, k_a, v_a, m_a, o_a)}


def _uniform_dataformat(q, k, v, out, mask):
    """`check_uniform_dataformat` -- with streaming off it compares q/k/v/out and the mask."""
    dts = {str(t.dtype) for t in (q, k, v, out, mask)}
    return len(dts) == 1


def sdpa(device, q, k, v, mask, out, q_chunk_size, k_chunk_size, grid, ckc, scale, **kw):
    key = (str(q.padded_shape), str(k.padded_shape), str(mask.padded_shape), str(out.padded_shape),
           str(q.dtype), q_chunk_size, k_chunk_size, grid, tuple(str(c) for c in ckc),
           tuple(sorted((kw.get("defines_extra") or {}).items())),
           kw.get("mask_cb_tiles"), str(kw.get("kernel_dir")), kw.get("split"),
           kw.get("kv_buffer_factor"))
    e = _CACHE.get(key)
    if e is None:
        e = _CACHE[key] = build(device, q, k, v, mask, out, q_chunk_size, k_chunk_size, grid,
                                ckc, scale, **kw)
    addrs = (q.buffer_address(), k.buffer_address(), v.buffer_address(),
             mask.buffer_address(), out.buffer_address())
    if addrs != e["addrs"]:
        rr, wr, cr = e["rt"]
        for _, a in rr:
            a[0], a[1], a[2], a[3] = addrs[0], addrs[1], addrs[2], addrs[3]
        for _, a in wr:
            a[0] = addrs[4]
        e["kernels"][0].runtime_args = rr
        e["kernels"][1].runtime_args = wr
        e["pd"] = ttnn.ProgramDescriptor(kernels=e["kernels"], semaphores=e["semaphores"],
                                         cbs=e["cbs"])
        e["addrs"] = addrs
    ttnn.generic_op([q, k, v, mask, out], e["pd"])
    return out
