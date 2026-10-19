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

import os
import re
from pathlib import Path

import ttnn

from .mm_generic import bound_descriptor, remember_bound, tile_bytes, ttnn_cpp_root

INVALID, VALID = 0, 1
TILE = 32
# What TensorAccessorArgs(nullptr) appends: ArgConfig::None (0) and a zero page size.
NULL_ACCESSOR = [0, 0]

_CACHE: dict = {}


def _kdir():
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


# Physical L1 per Tensix core, and the slice of it a program has already spent before its first
# circular buffer. Both measured, on qb1 card 2 (p150a, ttnn 0.68.0):
#
#     TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
#     grow to 4844032 B which is beyond max L1 size of 1572864 B
#
# and the figure it reports is `cb_bytes` plus a constant 109056 B, exact on all ten refusals
# measured against it (eight from a chunk-size probe, two from a live BoltzGen design at 2208
# padded tokens) and on the 3424768 B clash `tests/test_capacity_gate.py` already quotes.
# Those are Blackhole figures. A Wormhole core enforces 1499136 B and its CB region starts 103712 B
# in, so it has 1395424 B for CBs (a Galaxy's own allocator report, `perf/ceilrfd3/results`), 68384 B
# less than this pair allows. Priced against Blackhole, OpenFold3's fused-HiFi triangle attention
# at 512 tokens built q512/k512 on a Galaxy and the device refused it twice per process (ends
# 1584416 and 1518880 B, exactly 103712 + `cb_bytes`), and every caught refusal is a hang risk
# (spd-galaxy .103 A/B 10-18: 6 caught throws a run hung 6 of 54 runs, 2 a run 1 of 57).
# `device->l1_size_per_core()` is not exposed to Python, hence constants keyed on the arch (the
# same gap `softmax_generic` documents). `MAX_L1` is re-read from any refusal that names it, so a
# part with a different L1 corrects the constant instead of being mispriced by it.
L1_PER_CORE = 1572864
PROGRAM_RESERVE = 109056
WH_CB_BUDGET = 1395424


def cb_budget(wormhole=None) -> int:
    """Bytes per core the static circular buffers may take on this part."""
    if wormhole is None:
        from .tenstorrent import is_wormhole
        wormhole = is_wormhole()
    return WH_CB_BUDGET if wormhole else L1_PER_CORE - PROGRAM_RESERVE


class _ShapeOnly:
    """Enough of a ttnn tensor for `plan`, which reads padded_shape, shape and dtype only.

    So the CB surface can be priced on the host: `perf/bgsdpa/cb_model.py` enumerates it and
    `triatt_sdpa.fused_pairs` picks from it without allocating anything.
    """

    def __init__(self, shape, dtype=None):
        self.padded_shape = self.shape = list(shape)
        self.dtype = dtype if dtype is not None else ttnn.bfloat16


_KV_WINDOW_ORIG = (
    "const uint32_t k_start_tile_id = k_tile_shape.id_of(nb, k_head, kv_row_start_tile, 0);",
    "const uint32_t v_start_tile_id = v_tile_shape.id_of(nb, v_head, kv_row_start_tile, 0);")
_Q_WINDOW_ORIG = "uint32_t q_read_tile_id = q_tile_shape.id_of(nb, nq, read_offset + q_row_start_tile, 0);"


def kv_window_kernel_dir() -> Path:
    """The wheel's own SDPA reader with K and V read as sliding windows of a frame (`kv_window`).

    Generated from the running wheel's source, so it follows the wheel instead of pinning a copy:
    the two lines that place a K/V chunk get an `#ifdef KV_WINDOW_NB` twin, nothing else changes,
    and a wheel whose reader no longer has those lines fails here, not on the device. Head `n` of
    the virtual K is frame head `n / KV_WINDOW_NB`, starting `n % KV_WINDOW_NB` tile rows down. With
    `KV_WINDOW_Q_LEAD_T` the query is a frame too: head `n` starts `KV_WINDOW_Q_LEAD_T` tile rows
    further down, one window of Sq rows per head."""
    src = (_kdir() / "dataflow/reader_interleaved.cpp").read_text()
    common = (_kdir() / "dataflow/dataflow_common.hpp").read_text()
    for line in _KV_WINDOW_ORIG + (_Q_WINDOW_ORIG,):
        assert src.count(line) == 1, ("kv_window: the wheel's SDPA reader changed", line)
    win = ("#ifdef KV_WINDOW_NB\n"
           "                        const uint32_t k_start_tile_id = ((nb * KV_WINDOW_NKH + k_head / KV_WINDOW_NB) *\n"
           "                            KV_WINDOW_FRAME_T + k_head % KV_WINDOW_NB + kv_row_start_tile) * DHt;\n"
           "                        const uint32_t v_start_tile_id = ((nb * KV_WINDOW_NKH + v_head / KV_WINDOW_NB) *\n"
           "                            KV_WINDOW_FRAME_T + v_head % KV_WINDOW_NB + kv_row_start_tile) * DHt;\n"
           "#else\n")
    src = src.replace(_KV_WINDOW_ORIG[0], win + _KV_WINDOW_ORIG[0], 1)
    src = src.replace(_KV_WINDOW_ORIG[1], _KV_WINDOW_ORIG[1] + "\n#endif", 1)
    src = src.replace(_Q_WINDOW_ORIG, (
        "#ifdef KV_WINDOW_Q_LEAD_T\n"
        "                    uint32_t q_read_tile_id = ((nb * KV_WINDOW_NKH + nq / KV_WINDOW_NB) * KV_WINDOW_FRAME_T +\n"
        "                        KV_WINDOW_Q_LEAD_T + nq % KV_WINDOW_NB + read_offset + q_row_start_tile) * DHt;\n"
        "#else\n") + _Q_WINDOW_ORIG + "\n#endif", 1)
    import hashlib
    d = (Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tt_bio" / "sdpa_kv_window"
         / hashlib.sha1((src + common).encode()).hexdigest()[:12])
    if not (d / "dataflow/reader_interleaved.cpp").exists():
        (d / "dataflow").mkdir(parents=True, exist_ok=True)
        (d / "dataflow/dataflow_common.hpp").write_text(common)
        tmp = d / f"dataflow/.reader.{os.getpid()}"
        tmp.write_text(src)
        tmp.replace(d / "dataflow/reader_interleaved.cpp")
    return d


CB_CSS = 13
_CSS_LOOP_ORIG = "for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {"
_CSS_FIRST_ORIG = "if (k_chunk == 0) {"
_CSS_END_ORIG = "k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;"


def css_kernel_dir() -> Path:
    """The wheel's SDPA with each q chunk visiting only a listed set of key chunks (`css`).

    Generated from the running wheel's reader and compute source, three lines patched, so it follows
    the wheel. Row `q_chunk` of the list tensor is `[n, c_0, ..., c_{n-1}, pad]`: the reader walks
    those key chunks in that order, reading K, V and the mask tiles in place, and the compute kernel
    runs `n` iterations, its per-q-chunk count coming from runtime args. A skipped chunk is one whose
    mask is -1e4 everywhere: in the online softmax it adds exp(-1e4 - max) = 0 to the sum and a zero
    block to the output, so leaving it out changes nothing but the work."""
    kd = _kdir()
    reader = (kd / "dataflow/reader_interleaved.cpp").read_text()
    common = (kd / "compute/compute_common.hpp").read_text()
    for src, line in ((reader, _CSS_LOOP_ORIG), (reader, _CSS_FIRST_ORIG), (common, _CSS_END_ORIG)):
        assert src.count(line) == 1, ("css: the wheel's SDPA kernels changed", line)
    reader = reader.replace(_CSS_LOOP_ORIG, (
        "#ifdef CSS_W\n"
        "                    constexpr auto css_args = TensorAccessorArgs<CSS_ACC>();\n"
        "                    const auto css_reader = TensorAccessor(css_args, get_arg_val<uint32_t>(CSS_ADDR_ARG), CSS_W * 4);\n"
        "                    cb_reserve_back(CSS_CB, 1);\n"
        "                    const uint32_t css_l1 = get_write_ptr(CSS_CB);\n"
        "                    noc_async_read(css_reader.get_noc_addr(q_chunk), css_l1, CSS_W * 4);\n"
        "                    noc_async_read_barrier();\n"
        "                    volatile tt_l1_ptr uint32_t* css = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(css_l1);\n"
        "                    const uint32_t css_n = css[0];\n"
        "                    for (uint32_t k_pos = 0; k_pos < css_n; ++k_pos) {\n"
        "                        const uint32_t k_chunk = css[1 + k_pos];\n"
        "#else\n"
        "                    " + _CSS_LOOP_ORIG + "\n"
        "                        const uint32_t k_pos = k_chunk;\n"
        "#endif"), 1)
    reader = reader.replace(_CSS_FIRST_ORIG, "if (k_pos == 0) {", 1)
    common = common.replace(_CSS_END_ORIG, (
        "#ifdef CSS_W\n"
        "            k_chunk_end = get_arg_val<uint32_t>(CSS_COUNT_ARG + q_low_idx / Sq_chunk_t);\n"
        "#else\n"
        "            " + _CSS_END_ORIG + "\n"
        "#endif"), 1)
    files = {"dataflow/reader_interleaved.cpp": reader, "compute/compute_common.hpp": common}
    for name in ("dataflow/dataflow_common.hpp", "compute/sdpa.cpp", "compute/compute_streaming.hpp"):
        files[name] = (kd / name).read_text()
    import hashlib
    d = (Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tt_bio" / "sdpa_css"
         / hashlib.sha1("".join(files[k] for k in sorted(files)).encode()).hexdigest()[:12])
    if not (d / "compute/compute_common.hpp").exists():
        for name, src in files.items():
            (d / name).parent.mkdir(parents=True, exist_ok=True)
            tmp = d / (name + f".{os.getpid()}")
            tmp.write_text(src)
            tmp.replace(d / name)
    return d


def css_width(k_num_chunks):
    return _div_up(k_num_chunks + 1, 16) * 16


def css_lists(occ):
    """Bool `[nq, nk]` chunk occupancy (q chunk i reads key chunk j) -> (host int32 [1, 1, nq, W] list
    rows, counts).

    A row is its count, then its key chunk ids ascending. W fits every chunk plus the count word,
    rounded to 16 words so a row is a 64-byte page (the DRAM read alignment on Blackhole, twice
    Wormhole's), and depends only on the key axis, so one compiled program serves every step. Every
    row must be non-empty: a q chunk with no keys would leave its output rows unnormalised."""
    import torch
    nq, nk = occ.shape
    counts = occ.sum(1, dtype=torch.int32)
    assert bool((counts > 0).all()), "css: a q chunk with no key chunk"
    ids = torch.where(occ, torch.arange(nk, dtype=torch.int32), nk).sort(1).values
    rows = torch.zeros(nq, css_width(nk), dtype=torch.int32)
    rows[:, 0] = counts
    rows[:, 1:1 + nk] = torch.where(ids < nk, ids, 0)
    return rows.reshape(1, 1, nq, -1), tuple(counts.tolist())


_QKV_ROW_SKIPS = (
    # read_chunk_with_padding: Q, K and V are CBs 0-2; the mask is read inline, not through here.
    ("        start_tile_id += skip_src_cols;\n",
     "        start_tile_id += skip_src_cols;\n#ifdef QKV_FUSED_W\n"
     "        if (cb_id <= 2) { start_tile_id += QKV_FUSED_W - DHT_FUSED; }\n#endif\n"),
    # read_q_subblock
    ("            }\n            // Zero-pad extra columns (src_cols < dst_cols case)\n",
     "            }\n#ifdef QKV_FUSED_W\n            start_tile_id += QKV_FUSED_W - DHT_FUSED;\n#endif\n"
     "            // Zero-pad extra columns (src_cols < dst_cols case)\n"),
    # write_block (the TensorAccessor one, the output's)
    ("        }\n    }\n    noc_async_write_barrier();\n    cb_pop_front(cb_out, out_chunk_tiles);\n",
     "        }\n#ifdef QKV_FUSED_W\n        tile_id += OUT_FUSED_W - DHT_FUSED;\n#endif\n"
     "    }\n    noc_async_write_barrier();\n    cb_pop_front(cb_out, out_chunk_tiles);\n"),
)
_OUT_ID_ORIG = "uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile, 0);"


def _twin(src, orig, alt):
    """`orig` gets an `#ifdef QKV_FUSED_W` twin `alt`; a wheel without the line fails here."""
    assert src.count(orig) == 1, ("qkv_fused: the wheel's SDPA kernels changed", orig)
    return src.replace(orig, f"#ifdef QKV_FUSED_W\n{alt}\n#else\n{orig}\n#endif", 1)


def qkv_fused_kernel_dir() -> Path:
    """The wheel's SDPA reader and writer reading q, k and v straight out of one fused projection.

    `qkv_fused`: the tensor is `[B, 1, S, 3 * NH * DH]` (q heads, then k, then v, DH-wide each, the
    layout a fused qkv linear writes) and the output is written `[B, 1, S, NH * DH]`, the layout
    `nlp_concat_heads` gives. Only tile ADDRESSES change: head `n`'s chunk starts at column tile
    `n * DHt` of its row and every row skips the other heads' columns. The arithmetic is the stock
    program's, so the result is bit-identical to create-heads, SDPA, concat-heads."""
    common = (_kdir() / "dataflow/dataflow_common.hpp").read_text()
    for orig, alt in _QKV_ROW_SKIPS:
        assert common.count(orig) == 1, ("qkv_fused: the wheel's dataflow_common changed", orig)
        common = common.replace(orig, alt, 1)
    rows = "(nb * QKV_ROWS_T + {r}) * QKV_FUSED_W"
    reader = (_kdir() / "dataflow/reader_interleaved.cpp").read_text()
    reader = _twin(reader, _Q_WINDOW_ORIG, "                    uint32_t q_read_tile_id = "
                   + rows.format(r="read_offset + q_row_start_tile") + " + nq * DHt;")
    reader = _twin(reader, _KV_WINDOW_ORIG[0], "                        const uint32_t k_start_tile_id = "
                   + rows.format(r="kv_row_start_tile") + " + (NQH + k_head) * DHt;")
    reader = _twin(reader, _KV_WINDOW_ORIG[1], "                        const uint32_t v_start_tile_id = "
                   + rows.format(r="kv_row_start_tile") + " + (2 * NQH + v_head) * DHt;")
    writer = _twin((_kdir() / "dataflow/writer_interleaved.cpp").read_text(), _OUT_ID_ORIG,
                   "                    uint32_t out_tile_id = (nb * QKV_ROWS_T + write_offset + "
                   "out_row_start_tile) * OUT_FUSED_W + nq * vDHt;")
    import hashlib
    d = (Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tt_bio" / "sdpa_qkv_fused"
         / hashlib.sha1((reader + writer + common).encode()).hexdigest()[:12])
    if not (d / "dataflow/writer_interleaved.cpp").exists():
        (d / "dataflow").mkdir(parents=True, exist_ok=True)
        for name, s in (("dataflow_common.hpp", common), ("reader_interleaved.cpp", reader),
                        ("writer_interleaved.cpp", writer)):
            tmp = d / f"dataflow/.{name}.{os.getpid()}"
            tmp.write_text(s)
            tmp.replace(d / "dataflow" / name)
    return d


def plan_for_shape(seq, heads, head_dim, q_chunk, k_chunk, grid=(11, 10), split=None,
                   dtype=None, batch=0):
    """`plan` for a square attention call at `seq` padded tokens, without a device. The batch is
    `seq` (triangle attention's rows) unless given."""
    qkv = _ShapeOnly([batch or seq, heads, seq, head_dim], dtype)
    bias = _ShapeOnly([1, heads, seq, seq], dtype)
    ckc = (ttnn.MathFidelity.HiFi2, True, False, False)
    return plan(qkv, qkv, qkv, bias, qkv, q_chunk, k_chunk, grid, ckc, 1.0, split)


def chunk_divisors(seq, tile=TILE):
    """The chunk sizes that DIVIDE `seq` and are themselves a multiple of the 32-row tile,
    widest first. A chunk that does not divide sets `use_padded_mask`, which the fused kernel
    declines outright and which makes the stock op read a mask padded past the sequence."""
    return tuple(sorted({seq // n for n in range(1, seq // tile + 1)
                         if seq % n == 0 and (seq // n) % tile == 0}, reverse=True))


def cb_table(p, q_dtype, k_dtype, v_dtype, mask_dtype, out_dtype, mask_cb_tiles=None,
             extra_cbs=(), im_dtype=None, out_im_dtype=None, stats_dtype=None, one_k_trim=False):
    """(buffer index, tiles, page bytes, data format) for every CB the factory creates, :405-414.

    `mask_cb_tiles` overrides `cb_mask_in`: K2 fronts the whole head's mask grid
    (`k_num_chunks * Sq_chunk_t * Sk_chunk_t` tiles) where the stock op double-buffers one chunk.
    `extra_cbs` are rows in the same shape, appended -- the qkv fold's resident x block and weight
    slices go through here so `cb_fits_l1` prices them with everything else.

    `im_dtype` sets the score CB (`cb_qk_im`, the Sq_chunk x Sk_chunk tile block) and
    `out_im_dtype` the two output accumulators. The factory pins both to bf16 (:651-653); `None`
    keeps that. The standard compute path reconfigures unpack/pack formats at every stage boundary,
    so a narrower intermediate is a page-size change, not a kernel change. `stats_dtype` sets the
    five row-statistics CBs (running max, running sum and their previous values), which an fp32
    attention needs: a bf16 running sum keeps 8 bits of the softmax denominator. The scalar tiles
    stay bf16, they hold exact constants.
    """
    im_df = scalar_df = ttnn.bfloat16                 # :651-653, always bf16
    im_ts = scalar_ts = 2048
    stats_df = stats_dtype or ttnn.bfloat16
    stats_ts = tile_bytes(stats_df)
    qk_df = im_dtype or im_df
    oim_df = out_im_dtype or im_df
    nmask = p["mask_tiles"] if mask_cb_tiles is None else mask_cb_tiles
    # `one_k_trim` prices THIS kernel as `build` allocates it. With one k chunk the online softmax
    # never rescales: the first chunk writes the B side (c_26 out_im_B, c_28 max_B, c_30 sum_B) and
    # the tail reads it back after the swap, so the A side and c_31 (exp_max_diff) are declared but
    # never touched. One tile each frees 32 KB + 3 statistics blocks at q512, which is what lets
    # (q512, k512) fit in bf16. Off by default: the measured stock-op refusals and `fused_pairs`'
    # ranking are priced on the full set, and trimming there would move picks nobody has measured.
    one_k = one_k_trim and p["k_num_chunks"] == 1
    oim_a = 1 if one_k else p["out_im_tiles"]
    st_a = 1 if one_k else p["statistics_tiles"]
    return [
        (0, p["q_tiles"], tile_bytes(q_dtype), q_dtype),
        (1, p["k_tiles"], tile_bytes(k_dtype), k_dtype),
        (2, p["v_tiles"], tile_bytes(v_dtype), v_dtype),
        (3, nmask, tile_bytes(mask_dtype), mask_dtype),
        (5, 1, scalar_ts, scalar_df),
        (7, 1, scalar_ts, scalar_df),
        (4, 1, im_ts, im_df),                         # c_recip_scratch, :746 (no attention sink)
        (24, p["qk_tiles"], tile_bytes(qk_df), qk_df),
        (25, oim_a, tile_bytes(oim_df), oim_df),
        (26, p["out_im_tiles"], tile_bytes(oim_df), oim_df),
        (27, st_a, stats_ts, stats_df),
        (28, p["statistics_tiles"], stats_ts, stats_df),
        (29, st_a, stats_ts, stats_df),
        (30, p["statistics_tiles"], stats_ts, stats_df),
        (31, st_a, stats_ts, stats_df),
        (16, p["out0_t"], tile_bytes(out_dtype), out_dtype),
    ] + list(extra_cbs)


def cb_bytes(p, q_dtype=ttnn.bfloat16, k_dtype=ttnn.bfloat16, v_dtype=ttnn.bfloat16,
             mask_dtype=ttnn.bfloat16, out_dtype=ttnn.bfloat16, mask_cb_tiles=None,
             extra_cbs=(), im_dtype=None, out_im_dtype=None, stats_dtype=None,
             one_k_trim=False) -> int:
    """L1 the static circular buffers hold, per core."""
    return sum(n * page for _i, n, page, _f in cb_table(
        p, q_dtype, k_dtype, v_dtype, mask_dtype, out_dtype, mask_cb_tiles, extra_cbs,
        im_dtype, out_im_dtype, stats_dtype, one_k_trim))


def cb_fits_l1(p, **kw) -> bool:
    """Whether tt-metal will accept this config's CBs. `mask_cb_tiles` selects the fused kernel."""
    return cb_bytes(p, **kw) <= cb_budget()


CB_X_IN, CB_W_IN = 10, 11


def fuse_qkv_cbs(p, Ct, x_buffer_factor=2, dtype=ttnn.bfloat16):
    """The two CBs the qkv fold adds: the resident x row-block and the head's weight slices.

    x is [B, S, C] and one core owns one (batch, head) at a time, so the block is
    `Sq_chunk_t * Ct` tiles; `x_buffer_factor` 2 lets the reader fetch the next batch row while the
    compute kernel is still contracting this one. The weights are three `Ct x DHt` slices, read once
    per core and never refilled.
    """
    page = tile_bytes(dtype)
    return [(CB_X_IN, x_buffer_factor * p["Sq_chunk_t"] * Ct, page, dtype),
            (CB_W_IN, 3 * Ct * p["DHt"], page, dtype)]


CB_GATE_IN = 12


def gate_cbs(p, gate_buffer_factor=2, dtype=ttnn.bfloat16):
    """The one CB the gate epilogue adds: the q chunk's gate block, `Sq_chunk_t * vDHt` tiles.

    `gate_buffer_factor` 2 lets the reader fetch the next chunk's gate while the compute kernel is
    still on this one; 1 is the fall-back when L1 will not hold two.
    """
    return [(CB_GATE_IN, gate_buffer_factor * p["Sq_chunk_t"] * p["vDHt"], tile_bytes(dtype),
             dtype)]


def proj_subblock_h(p, Ct):
    """The tall side of the projection subblock: the widest that divides the M tiles and fits DST."""
    for h in range(min(p["Sq_chunk_t"], p["dst_size"] // max(p["DHt"], 1)), 0, -1):
        if p["Sq_chunk_t"] % h == 0:
            return h
    return 1


_MAX_L1_RX = re.compile(r"max L1 size of (\d+) B")


def note_l1_refusal(message: str) -> None:
    """Take `L1_PER_CORE` from a refusal that names it, so a part whose per-core L1 is not
    1.5 MiB is priced by its own number instead of by this file's constant."""
    global L1_PER_CORE
    m = _MAX_L1_RX.search(message)
    if m:
        L1_PER_CORE = int(m.group(1))


def build(device, q, k, v, mask, out, q_chunk_size, k_chunk_size, grid, ckc, scale,
          exp_approx_mode=False, mask_cb_tiles=None, defines_extra=None, kernel_dir=None,
          split=None, kv_buffer_factor=2, fuse_qkv=None, gate=None, im_dtype=None,
          out_im_dtype=None, stats_dtype=None, kv_window=None, qkv_fused=None, css=None,
          keypad=None, merged_out=False):
    """The ProgramDescriptor for the fold's SDPA call.

    `mask_cb_tiles` overrides the size of `cb_mask_in` (K2 makes it the whole head's grid instead of
    a double-buffered chunk; at the shipped config those are the same 256 tiles). `kernel_dir` swaps
    in patched kernel sources. With both left alone this is the wheel's own program.

    `gate` is `(g, gate_buffer_factor)`: triangle attention's `[B, NQH, Sq, DH]` gate, which the
    compute kernel's pack stage multiplies in as `out * sigmoid(g)` instead of the caller running
    that multiply as its own program. Head-major and DHt == 1, so gate tile i is output tile i.

    `css` is `(lists, counts)` from `css_lists` (lists on device, ROW_MAJOR uint32): q chunk i visits
    only the key chunks its row lists (`css_kernel_dir`).

    `qkv_fused` is the head count NH: `q` is a fused `[B, 1, S, 3*NH*DH]` projection (`k` and `v`
    are the same tensor) and `out` is `[B, 1, S, NH*DH]` (`qkv_fused_kernel_dir`).

    `fuse_qkv` is `(x, w, x_buffer_factor)`: the pre-projection pair tensor `[B, S, C]` and its
    `[C, 3*NQH*DH]` weight. q, k and v are then made inside this program instead of read, and the
    three tensors passed as `q`, `k`, `v` are used for their SHAPES and dtypes only -- the reader
    never touches their addresses. The whole K = Ct contraction is one block, which is the order
    `_MM_BLOCK[(4, 12)]` already gives the standalone projection at this site.
    """
    kp, vp = k, v
    q_fused = out_fused = None
    out_merged = None
    if merged_out:
        # `out` is `[B, 1, S, NH * DH]`, the layout nlp_concat_heads gives: the writer half of `qkv_fused`
        # (only its output addresses; q, k and v stay per head). Bit-identical to SDPA then concat-heads.
        B_, NH_, S_, DH_ = (int(d) for d in q.padded_shape)
        assert qkv_fused is None and [int(d) for d in out.padded_shape] == [B_, 1, S_, NH_ * DH_], (out.shape, q.shape)
        out_merged, out = out, _ShapeOnly([B_, NH_, S_, DH_], out.dtype)
    if qkv_fused is not None:
        assert kv_window is None and fuse_qkv is None and gate is None
        B_, one, S_, W_ = (int(d) for d in q.padded_shape)
        DH_ = W_ // (3 * qkv_fused)
        assert one == 1 and W_ == 3 * qkv_fused * DH_ and DH_ % TILE == 0, (q.shape, qkv_fused)
        assert [int(d) for d in out.padded_shape] == [B_, 1, S_, qkv_fused * DH_], (out.shape, q.shape)
        q_fused, out_fused = q, out
        q = kp = vp = _ShapeOnly([B_, qkv_fused, S_, DH_], q.dtype)
        out = _ShapeOnly([B_, qkv_fused, S_, DH_], out.dtype)
        kernel_dir = kernel_dir or qkv_fused_kernel_dir()
    if kv_window is not None:
        # `kv_window` is `(nb, W)`: k and v are FRAMES [B, NKH, F, DH], and the attention runs as if
        # K/V were [B, NKH * nb, W, DH] with head h * nb + i = frame head h, rows i*32 .. i*32 + W.
        # The windows overlap, so building them would copy every frame row W/32 times; the reader
        # reads them in place instead (`kv_window_kernel_dir`).
        # `(nb, W, q_lead, Sq)` makes q a frame of the same shape too: head h * nb + i is frame rows
        # q_lead + i*32 .. q_lead + i*32 + Sq.
        nbw, W = kv_window[:2]
        B_, NKH_, F_, DH_ = (int(d) for d in k.padded_shape)
        assert F_ % TILE == 0 and W % TILE == 0 and (nbw - 1) * TILE + W <= F_, (k.shape, kv_window)
        kp, vp = _ShapeOnly([B_, NKH_ * nbw, W, DH_], k.dtype), _ShapeOnly([B_, NKH_ * nbw, W, DH_], v.dtype)
        if len(kv_window) == 4:
            ql, Sq_ = kv_window[2:]
            assert list(q.padded_shape) == list(k.padded_shape) and ql % TILE == 0 and Sq_ % TILE == 0
            assert ql + (nbw - 1) * TILE + Sq_ <= F_, ("q window runs off the frame", q.shape, kv_window)
            q_frame, q = q, _ShapeOnly([B_, NKH_ * nbw, Sq_, DH_], q.dtype)
        else:
            assert int(q.padded_shape[1]) == NKH_ * nbw, (q.shape, k.shape, kv_window)
        kernel_dir = kernel_dir or kv_window_kernel_dir()
    p = plan(q, kp, vp, mask, out, q_chunk_size, k_chunk_size, grid, ckc, scale, split,
             kv_buffer_factor)
    if css is not None:
        assert kv_window is None and qkv_fused is None and fuse_qkv is None and gate is None
        css_w = css_width(p["k_num_chunks"])
        assert [int(d) for d in css[0].padded_shape] == [1, 1, p["q_num_chunks"], css_w], css[0].shape
        assert len(css[1]) == p["q_num_chunks"], (len(css[1]), p["q_num_chunks"])
        kernel_dir = kernel_dir or css_kernel_dir()
    if kv_window is not None and len(kv_window) == 4:
        q = q_frame
    if qkv_fused is not None:
        q, kp, vp, out = q_fused, q_fused, q_fused, out_fused
    if out_merged is not None:
        out = out_merged
    gx, gy, num_cores = p["gx"], p["gy"], p["num_cores"]
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])


    extra_cbs = []
    Ct = proj_h = None
    if fuse_qkv is not None:
        x, w, x_buffer_factor = fuse_qkv
        C = int(x.padded_shape[-1])
        assert C % TILE == 0, C
        Ct = C // TILE
        assert p["DHt"] == 1, ("the fold's K layout is the reader's tile-order transpose, which is "
                               "the identity only at DHt == 1", p["DHt"])
        assert p["q_num_chunks"] == 1 and p["q_per_core"] == 1 and p["nh_per_core"] == 1
        assert p["Skt"] == p["Sq_chunk_t"], (p["Skt"], p["Sq_chunk_t"])
        # q, k and v are pushed whole, so cb_k_in / cb_v_in must hold the sequence, not a chunk.
        assert kv_buffer_factor >= p["k_num_chunks"], (kv_buffer_factor, p["k_num_chunks"])
        assert int(w.padded_shape[-2]) == C and int(w.padded_shape[-1]) == 3 * p["NQH"] * p["DH"]
        extra_cbs = fuse_qkv_cbs(p, Ct, x_buffer_factor, x.dtype)
        proj_h = proj_subblock_h(p, Ct)

    if gate is not None:
        # One reader tail, two levers: both append their addresses after the chain args, so only
        # one of them can own that tail. The qkv fold is an unshipped arm and the host refuses the
        # combination rather than guessing an order.
        assert fuse_qkv is None, "the gate epilogue and the qkv fold share the reader's tail args"
        gate_t, gate_buffer_factor = gate
        assert p["DHt"] == 1 and p["vDHt"] == 1, (p["DHt"], p["vDHt"])
        assert [int(d) for d in gate_t.padded_shape] == [p["B"], p["NQH"], p["Sq"], p["DH"]], \
            (gate_t.padded_shape, (p["B"], p["NQH"], p["Sq"], p["DH"]))
        extra_cbs = list(extra_cbs) + gate_cbs(p, gate_buffer_factor, gate_t.dtype)
    if css is not None:
        extra_cbs = list(extra_cbs) + [(CB_CSS, 1, 4 * css_w, ttnn.uint32)]
    kp_w = None if keypad is None else int(keypad.padded_shape[-1])
    if keypad is not None:
        assert 2 * p["B"] <= kp_w and kp_w % 8 == 0, (keypad.padded_shape, p["B"])
        extra_cbs = list(extra_cbs) + [(CB_CSS, 1, 4 * kp_w, ttnn.uint32)]

    cbs = [ttnn.CBDescriptor(
        total_size=n_tiles * page, core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt,
                                                    page_size=page)])
        for idx, n_tiles, page, fmt in cb_table(
            p, q.dtype, k.dtype, v.dtype, mask.dtype, out.dtype, mask_cb_tiles, extra_cbs,
            im_dtype, out_im_dtype, stats_dtype, one_k_trim=True)]

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
    if fuse_qkv is not None:
        reader_ct += acc(fuse_qkv[0]) + acc(fuse_qkv[1])
    if gate is not None:
        reader_ct += acc(gate[0])
    css_acc = len(reader_ct)
    if css is not None:
        reader_ct += acc(css[0])
    kp_acc = len(reader_ct)
    if keypad is not None:
        reader_ct += acc(keypad)

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
        # The mask add and the running-sum/max adds share one helper, so the granularity has to be
        # legal for the smallest block either of them passes -- that is Sq_chunk_t, the same count
        # STATS_GRANULARITY is priced against. Batching it is worth 1.0317x on the fused SDPA on
        # Blackhole (2.8299 -> 2.7430 ms at 512x512, perf/b2z_levers/addgran_512_qb2c0.json) and
        # 1.0267x on Wormhole, and is bit-exact against the per-tile loop at every granularity,
        # which `TT_BIO_SDPA_ADD_GRANULARITY=1` restores.
        "ADD_BLOCK_GRANULARITY": os.environ.get(
            "TT_BIO_SDPA_ADD_GRANULARITY") or str(valid_granularity(p["Sq_chunk_t"], ds)),
        "EXP_APPROX_MODE": str(int(exp_approx_mode)),
    }
    if fuse_qkv is not None:
        defines["FUSE_QKV"] = str(Ct)
        defines["PROJ_SUBBLOCK_H"] = str(proj_h)
    if gate is not None:
        defines["GATE_EPILOGUE"] = "1"
        # A binary SFPU op takes half of DST per operand, so the epilogue processes dst_size / 2
        # output tiles at a time -- narrowed to a divisor of Sq_chunk_t by the same helper the
        # kernel's other granularities use.
        defines["GATE_GRANULARITY"] = str(
            valid_granularity(p["Sq_chunk_t"], max(p["dst_size"] // 2, 1)))
    if im_dtype not in (None, ttnn.bfloat16):
        # The block row-max LLK streams the score tiles at bf16's tile stride (its own header says
        # the operand must be bfloat16_b), so a narrower score CB takes the per-tile reduce.
        defines["QK_TILEWISE_MAX"] = "1"
    if kv_window is not None:
        defines.update(KV_WINDOW_NB=str(kv_window[0]), KV_WINDOW_NKH=str(int(k.padded_shape[1])),
                       KV_WINDOW_FRAME_T=str(int(k.padded_shape[2]) // TILE))
        if len(kv_window) == 4:
            defines["KV_WINDOW_Q_LEAD_T"] = str(kv_window[2] // TILE)
    if qkv_fused is not None:
        defines.update(QKV_FUSED_W=str(int(q.padded_shape[-1]) // TILE), DHT_FUSED=str(p["DHt"]),
                       OUT_FUSED_W=str(int(out.padded_shape[-1]) // TILE),
                       QKV_ROWS_T=str(int(q.padded_shape[2]) // TILE))
    if out_merged is not None:
        # QKV_FUSED_W = DHT_FUSED leaves every read-side skip of the twin at zero; only the writer moves.
        defines.update(QKV_FUSED_W=str(p["DHt"]), DHT_FUSED=str(p["DHt"]),
                       OUT_FUSED_W=str(int(out.padded_shape[-1]) // TILE),
                       QKV_ROWS_T=str(int(out.padded_shape[2]) // TILE))
    if css is not None:
        # Reader args: 17 base + 14 chain, then the list address; compute args: 10, then the counts.
        defines.update(CSS_W=str(css_w), CSS_CB=str(CB_CSS), CSS_ACC=str(css_acc),
                       CSS_ADDR_ARG="31", CSS_COUNT_ARG="10")
    if keypad is not None:
        # `shsdpa`: a [1, 1, 1, W] uint32 table, per batch (key chunk count, boundary chunk), after the
        # 31 base reader args. It shares CB 13 with css, which it excludes.
        assert fuse_qkv is None and gate is None and css is None
        defines.update(KEYPAD="1", KEYPAD_ACC=str(kp_acc), KEYPAD_ADDR_ARG="31", KEYPAD_CB=str(CB_CSS),
                       KEYPAD_W=str(kp_w))
    if defines_extra:
        defines.update({str(a): str(b) for a, b in dict(defines_extra).items()})
    dlist = sorted(defines.items())

    kd = kernel_dir or _kdir()
    stock = _kdir()
    reader_src = str(Path(kd) / "dataflow/reader_interleaved.cpp")
    writer_src = str(Path(kd) / "dataflow/writer_interleaved.cpp")
    compute_src = str(Path(kd) / "compute/sdpa.cpp")
    if not Path(reader_src).exists():
        reader_src = str(stock / "dataflow/reader_interleaved.cpp")
    if not Path(writer_src).exists():
        writer_src = str(stock / "dataflow/writer_interleaved.cpp")
    if out_merged is not None:
        writer_src = str(qkv_fused_kernel_dir() / "dataflow/writer_interleaved.cpp")
    if not Path(compute_src).exists():
        compute_src = str(stock / "compute/sdpa.cpp")

    q_a, k_a, v_a = q.buffer_address(), k.buffer_address(), v.buffer_address()
    m_a, o_a = mask.buffer_address(), out.buffer_address()
    x_a = w_a = g_a = 0
    if fuse_qkv is not None:
        x_a, w_a = fuse_qkv[0].buffer_address(), fuse_qkv[1].buffer_address()
    if gate is not None:
        g_a = gate[0].buffer_address()
    css_tail = ([], []) if css is None else ([css[0].buffer_address()], list(css[1]))

    kp_tail = [] if keypad is None else [keypad.buffer_address()]
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
                   + [0] * 14                          # chain metadata, all no-chain (0/0 chains)
                   + ([] if fuse_qkv is None else [x_a, w_a])
                   + ([] if gate is None else [g_a]) + css_tail[0] + kp_tail))
        wr.append((core, [o_a, i, lb, lbe, ln, lne, lq, lqe, 1, 0, 0, 0]))
        cr.append((core, [i, lb, lbe, ln, lne, lq, lqe, 1, 0, 0] + css_tail[1]))

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
            "rt": (rr, wr, cr), "addrs": (q_a, k_a, v_a, m_a, o_a),
            "fuse": None if fuse_qkv is None else (Ct, proj_h, x_a, w_a),
            "gate": None if gate is None else g_a,
            "css": None if css is None else (css_tail[0][0], tuple(css_tail[1])),
            "kp": None if keypad is None else kp_tail[0]}


def _uniform_dataformat(q, k, v, out, mask):
    """`check_uniform_dataformat` -- with streaming off it compares q/k/v/out and the mask."""
    dts = {str(t.dtype) for t in (q, k, v, out, mask)}
    return len(dts) == 1


def sdpa(device, q, k, v, mask, out, q_chunk_size, k_chunk_size, grid, ckc, scale, **kw):
    key = (str(q.padded_shape), str(k.padded_shape), str(mask.padded_shape), str(out.padded_shape),
           # EVERY operand's dtype, not just q's. `build` compiles
           # `int(_uniform_dataformat(q, k, v, out, mask))` in as `check_uniform_dataformat`
           # (:440) and sizes each operand CB from `tile_bytes(that operand's dtype)`, so a key
           # carrying only q's format lets two calls that differ in the MASK's format collide and
           # hands the second one the first one's program -- a mask read at the wrong page size,
           # under a uniformity define computed for different operands. It does not decline, raise
           # or NaN: it returns finite garbage. Measured by `bfp8-accuracy-envelope` on
           # TT_BIO_TRIATT_BIAS_B8 at cdk2x2_298, same commit/card/seed: 22.21 A and plDDT
           # 0.387-0.452 interleaved after a base fold in one process, against 0.92671 A and plDDT
           # 0.901-0.910 for the identical flag alone in its own process, both digests
           # reproducible and the solo one bit-identical to origin/main. The flag was fine and
           # this key was not.
           #
           # Cost, stated precisely because `fast_dtypes_ok(dest=...)` now exempts the destination
           # from uniformity: with every bfp8 region OFF -- the shipped default -- the read
           # operands are uniform and both callers allocate `out` with the query's own dtype
           # (`triatt_sdpa.py:382` takes `q.dtype`, `:511` takes `x.dtype`), so all five strings
           # below are equal and this key partitions calls exactly as the q-only key did: same
           # entry count, same program-compile count, no perf risk on the default path. With a
           # region ON and only the DESTINATION narrowed, `out.dtype` differs from the reads and
           # this key correctly issues a second program for it -- that extra compile is the point,
           # since it is what stops the narrowed destination from colliding with the bf16 one.
           str(q.dtype), str(k.dtype), str(v.dtype), str(mask.dtype), str(out.dtype),
           # Each operand's accessor compiles in whether it is DRAM or L1, so a program built for DRAM q/k
           # reads the wrong memory when handed L1 ones (SaProt's RoPE leaves q, k in L1).
           tuple(str(t.memory_config().buffer_type) for t in (q, k, v, mask, out)),
           q_chunk_size, k_chunk_size, grid, tuple(str(c) for c in ckc),
           # `build` compiles the softmax scale into the writer and compute kernels. Without it
           # here two sites with equal padded shapes and different head dims share a program:
           # OpenFold3's template triangle attention (head_dim 16, padded to 32) ran first and
           # handed its 1/4 to the trunk's head_dim-32 calls, rel_rms 1.07 against float64
           # where the same call alone reads 0.022 (perf/spd/triatt_headdim.py).
           float(scale),
           tuple(sorted((kw.get("defines_extra") or {}).items())),
           kw.get("mask_cb_tiles"), str(kw.get("kernel_dir")), kw.get("split"),
           kw.get("kv_buffer_factor"),
           None if kw.get("fuse_qkv") is None else (
               str(kw["fuse_qkv"][0].padded_shape), str(kw["fuse_qkv"][1].padded_shape),
               kw["fuse_qkv"][2]),
           None if kw.get("gate") is None else (
               str(kw["gate"][0].padded_shape), kw["gate"][1]),
           # `build` reads this one define from the environment, and the cache is keyed on what
           # `build` was given -- so without it here an A/B that flips the env gets the FIRST arm's
           # compiled program back for both legs and reads a 1.000x that means nothing.
           os.environ.get("TT_BIO_SDPA_ADD_GRANULARITY"),
           str(kw.get("im_dtype")), str(kw.get("out_im_dtype")), str(kw.get("stats_dtype")),
           kw.get("kv_window"), kw.get("qkv_fused"), kw.get("css") is not None, bool(kw.get("merged_out")),
           None if kw.get("keypad") is None else str(kw["keypad"].padded_shape))
    e = _CACHE.get(key)
    if e is None:
        e = _CACHE[key] = build(device, q, k, v, mask, out, q_chunk_size, k_chunk_size, grid,
                                ckc, scale, **kw)
    addrs = (q.buffer_address(), k.buffer_address(), v.buffer_address(),
             mask.buffer_address(), out.buffer_address())
    fuse = kw.get("fuse_qkv")
    fuse_addrs = None if fuse is None else (fuse[0].buffer_address(), fuse[1].buffer_address())
    gate = kw.get("gate")
    gate_addr = None if gate is None else gate[0].buffer_address()
    css = kw.get("css")
    css_bind = None if css is None else (css[0].buffer_address(), tuple(css[1]))
    kp = kw.get("keypad")
    kp_addr = None if kp is None else kp.buffer_address()
    bound = (addrs, fuse_addrs, gate_addr, css_bind, kp_addr)
    held = (e["addrs"], None if fuse is None else e["fuse"][2:], None if gate is None else e["gate"],
            e.get("css"), e.get("kp"))
    if bound != held:
        pd = bound_descriptor(e, bound)
        if pd is None:
            remember_bound(e, held, e["pd"])     # the one `build` made, too
            rr, wr, cr = e["rt"]
            for _, a in rr:
                a[0], a[1], a[2], a[3] = addrs[0], addrs[1], addrs[2], addrs[3]
                # The two levers are mutually exclusive (`build` asserts it), so whichever is on
                # owns the tail of the reader's runtime args.
                if fuse is not None:
                    a[-2], a[-1] = fuse_addrs
                elif gate is not None:
                    a[-1] = gate_addr
                elif css is not None:
                    a[-1] = css_bind[0]
                elif kp is not None:
                    a[-1] = kp_addr
            for _, a in wr:
                a[0] = addrs[4]
            if css is not None:
                for _, a in cr:
                    a[10:] = css_bind[1]
                e["kernels"][2].runtime_args = cr
            e["kernels"][0].runtime_args = rr
            e["kernels"][1].runtime_args = wr
            e["pd"] = ttnn.ProgramDescriptor(kernels=e["kernels"], semaphores=e["semaphores"],
                                             cbs=e["cbs"])
            remember_bound(e, bound, e["pd"])
        else:
            # An address set this program launched at before: its descriptor, not a rebuild
            # (`mm_generic.bound_descriptor`). `rt` and the kernels keep the last REBUILT
            # addresses, which is safe because the rebuild above rewrites every address slot.
            e["pd"] = pd
        e["addrs"] = addrs
        if fuse is not None:
            e["fuse"] = e["fuse"][:2] + fuse_addrs
        if gate is not None:
            e["gate"] = gate_addr
        e["css"] = css_bind
        e["kp"] = kp_addr
    # The fold's reader never touches q/k/v -- they are here for their shapes, and the tensors the
    # program actually reads are x and w.
    operands = [q, k, v, mask, out] if fuse is None else [fuse[0], fuse[1], mask, out]
    if kw.get("qkv_fused") is not None:     # q, k and v are one tensor
        operands = [q, mask, out]
    if gate is not None:
        operands = operands + [gate[0]]
    if css is not None:
        operands = operands + [css[0]]
    if kp is not None:
        operands = operands + [kp]
    ttnn.generic_op(operands, e["pd"])
    return out
