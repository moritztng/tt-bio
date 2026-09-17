#!/usr/bin/env python3
"""Generate `tt_bio/kernels/mm_split/` from the wheel's own `minimal_matmul` kernels.

Two guarded arms, both no-ops unless their macro is defined, so the generated files are the
wheel's kernels byte for byte on the default path:

  MM_NOWRITE   the output CB is drained and popped exactly as it is today, but the DRAM write
               is not issued. Prices the write half of the op, which is the whole question the
               DRAM-write-serialisation diagnosis turns on. Output is garbage by construction,
               so this arm is a stopwatch, never a result.
  MM_SPLIT_LAST_TILES
               the split writer divides N into N_chunks EQUAL chunks. With this defined the
               LAST chunk carries that many N tiles instead, so two projections of different
               widths over one activation are two destinations of one matmul and the activation
               is read once. Addressing only: same tiles, same order, two buffers.
  MM_GATE      the output stage gates the accumulated block instead of copying it: the N axis
               carries tile-interleaved (value, gate) pairs and the epilogue packs
               `p * sigmoid(g)`, so the op writes HALF the tiles it computes. That deletes the
               DRAM round trip the separate `reblock_permute_gated` move pays for its operands.
               Not bit-exact, and deliberately so -- `compute_reblock_permute_gated.cpp`'s header
               is explicit that ttnn's sigmoid is only reproduced under a 16-bit DST, and the
               in-projection runs `fp32_dest_acc_en=True`. Scored in Angstrom, not by digest.
               Halving the write is a compile-time factor of two on the OUTPUT side only: the
               weight, the in1 reads and the core split all stay on the pre-gate N.
  MM_DUAL_NOC  every second output tile goes out on the OTHER NOC. The writer RISC carries
               512 MiB on NOC_1 while the reader RISC carries 128 MiB on NOC_0, and
               `noc_async_write_tile` takes an explicit noc index, so half the drain can move
               across without a second RISC, a second CB or a semaphore. Bit-exact by
               construction: same tiles, same addresses, same bytes, different wire.

Run from the repo root; overwrites the generated files.
"""

import sys
from pathlib import Path

OUT = Path("tt_bio/kernels/mm_split")

HELPER = r"""
// --- tt-bio: the two screen arms, see tt_bio/kernels/mm_split/patch_mm_split.py -----------------
#if defined(MM_DUAL_NOC)
// Alternate the wire, not the transaction. `noc_index` is this kernel's own NOC; 1 - noc_index is
// the one the other DM RISC uses for its DRAM reads and which carries a quarter of the traffic.
#define MM_WRITE_TILE(tid, acc, ptr, i) \
    noc_async_write_tile((tid), (acc), (ptr), (uint8_t)(((i) & 1) ? (1 - noc_index) : noc_index))
#define MM_WRITES_FLUSHED()                            \
    do {                                               \
        noc_async_writes_flushed(noc_index);           \
        noc_async_writes_flushed(1 - noc_index);       \
    } while (0)
#define MM_WRITE_BARRIER()                             \
    do {                                               \
        noc_async_write_barrier(noc_index);            \
        noc_async_write_barrier(1 - noc_index);        \
    } while (0)
#elif defined(MM_NOWRITE)
#define MM_WRITE_TILE(tid, acc, ptr, i) do { (void)(tid); (void)(ptr); } while (0)
#define MM_WRITES_FLUSHED() noc_async_writes_flushed()
#define MM_WRITE_BARRIER() noc_async_write_barrier()
#else
#define MM_WRITE_TILE(tid, acc, ptr, i) noc_async_write_tile((tid), (acc), (ptr))
#define MM_WRITES_FLUSHED() noc_async_writes_flushed()
#define MM_WRITE_BARRIER() noc_async_write_barrier()
#endif

// A trailing output chunk of a different width. Without MM_SPLIT_LAST_TILES every chunk is
// N_tiles_per_chunk wide and this is the expression the stock writer already evaluates.
#ifdef MM_SPLIT_LAST_TILES
#define MM_CHUNK_TILES(c, n_chunks, uniform) \
    ((uint32_t)((c) + 1 == (n_chunks) ? (uint32_t)(MM_SPLIT_LAST_TILES) : (uint32_t)(uniform)))
#else
#define MM_CHUNK_TILES(c, n_chunks, uniform) ((uint32_t)(uniform))
#endif

// The gated output stage packs one tile per (value, gate) pair, so every OUTPUT-side N quantity
// halves while the in1 reads and the core split keep the pre-gate N. Tile-interleaved pairs make
// the map exact: pre-gate tile t belongs to gated tile t >> 1, so a per-core range [2i, 2i+2)
// becomes [i, i+1) with no remainder. Undefined here is the stock kernel's own expression.
#ifdef MM_GATE
#define MM_OUT_N(n) ((uint32_t)(n) >> 1)
#else
#define MM_OUT_N(n) ((uint32_t)(n))
#endif
// -----------------------------------------------------------------------------------------------
"""

# (file, old, new). Every edit is exact-match and asserted, so a wheel bump that moves any of
# these lines fails loudly instead of silently generating the stock kernel.
HPP_EDITS = [
    # write_block_sync: block-at-a-time drain, used on the deferred-write path.
    ("""            uint32_t tile_id = i * shape.logical_d1 + j;
            noc_async_write_tile(tile_id, tensor_accessor, read_ptr);
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_writes_flushed();""",
     """            uint32_t tile_id = i * shape.logical_d1 + j;
            MM_WRITE_TILE(tile_id, tensor_accessor, read_ptr, mm_write_seq);
            mm_write_seq++;
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    MM_WRITES_FLUSHED();"""),
    # write_block_sync_granular: row-at-a-time drain, the path N_chunks == 1 actually takes.
    ("""                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
                noc_async_write_tile(tile_id, tensor_accessor, out_read_ptr);
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_pop_front(cb_id_out, N_block_tiles);
    }
    noc_async_writes_flushed();""",
     """                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
                MM_WRITE_TILE(tile_id, tensor_accessor, out_read_ptr, mm_write_seq);
                mm_write_seq++;
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_pop_front(cb_id_out, N_block_tiles);
    }
    MM_WRITES_FLUSHED();"""),
    # write_tile_to_chunk: the split path drains on one NOC where the single-output path already
    # alternates, so a two-chunk call would have lost MM_DUAL_NOC purely by taking a different
    # branch. Same tiles, same addresses.
    ("""    ((chunk_idx == Is ? (noc_async_write_tile(tile_id, std::get<Is>(accessors), read_ptr), void()) : void()), ...);""",
     """    ((chunk_idx == Is ? (MM_WRITE_TILE(tile_id, std::get<Is>(accessors), read_ptr, mm_write_seq), void()) : void()), ...);"""),
    # write_block_sync_split: walk the chunk widths instead of dividing by one of them.
    ("""    const uint32_t chunk_idx_start = d1_start / N_tiles_per_chunk;
    const uint32_t tile_idx_in_chunk_start = d1_start % N_tiles_per_chunk;

    for (uint32_t i = d0_start; i < d0_end; i++) {
        // Assumes that all chunks have same number of tiles on the M-axis
        if (i >= chunk_shape.logical_d0) {
            break;
        }

        uint32_t chunk_idx = chunk_idx_start;
        uint32_t tile_idx_in_chunk = tile_idx_in_chunk_start;

        for (uint32_t j = d1_start; j < d1_end; j++, tile_idx_in_chunk++) {
            // If we've reached the end of the current chunk, move to the next one
            if (tile_idx_in_chunk >= chunk_shape.logical_d1) {
                tile_idx_in_chunk = 0;
                chunk_idx++;  // Move to next chunk; if chunk is past last one then next branch will skip padding
            }

            // Skip padding
            if (chunk_idx >= N_chunks) {
                read_ptr += tile_size_bytes;
                continue;
            }

            uint32_t tile_id_in_chunk = i * chunk_shape.logical_d1 + tile_idx_in_chunk;

            // Compile-time dispatch preserving concrete types
            write_tile_to_chunk(
                accessors, chunk_idx, tile_id_in_chunk, read_ptr, std::index_sequence_for<Accessors...>{});
            read_ptr += tile_size_bytes;
        }""",
     """    uint32_t chunk_idx_start = 0;
    uint32_t tile_idx_in_chunk_start = d1_start;
    while (chunk_idx_start < N_chunks &&
           tile_idx_in_chunk_start >= MM_CHUNK_TILES(chunk_idx_start, N_chunks, N_tiles_per_chunk)) {
        tile_idx_in_chunk_start -= MM_CHUNK_TILES(chunk_idx_start, N_chunks, N_tiles_per_chunk);
        chunk_idx_start++;
    }

    for (uint32_t i = d0_start; i < d0_end; i++) {
        // Assumes that all chunks have same number of tiles on the M-axis
        if (i >= chunk_shape.logical_d0) {
            break;
        }

        uint32_t chunk_idx = chunk_idx_start;
        uint32_t tile_idx_in_chunk = tile_idx_in_chunk_start;

        for (uint32_t j = d1_start; j < d1_end; j++, tile_idx_in_chunk++) {
            // If we've reached the end of the current chunk, move to the next one
            if (tile_idx_in_chunk >= MM_CHUNK_TILES(chunk_idx, N_chunks, N_tiles_per_chunk)) {
                tile_idx_in_chunk = 0;
                chunk_idx++;  // Move to next chunk; if chunk is past last one then next branch will skip padding
            }

            // Skip padding
            if (chunk_idx >= N_chunks) {
                read_ptr += tile_size_bytes;
                continue;
            }

            uint32_t tile_id_in_chunk =
                i * MM_CHUNK_TILES(chunk_idx, N_chunks, N_tiles_per_chunk) + tile_idx_in_chunk;

            // Compile-time dispatch preserving concrete types
            write_tile_to_chunk(
                accessors, chunk_idx, tile_id_in_chunk, read_ptr, std::index_sequence_for<Accessors...>{});
            mm_write_seq++;
            read_ptr += tile_size_bytes;
        }"""),
    # write_block_sync_granular_split: the same walk on the row-at-a-time drain.
    ("""    const uint32_t chunk_idx_start = d1_start / N_tiles_per_chunk;
    const uint32_t tile_idx_in_chunk_start = d1_start % N_tiles_per_chunk;

    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {""",
     """    uint32_t chunk_idx_start = 0;
    uint32_t tile_idx_in_chunk_start = d1_start;
    while (chunk_idx_start < N_chunks &&
           tile_idx_in_chunk_start >= MM_CHUNK_TILES(chunk_idx_start, N_chunks, N_tiles_per_chunk)) {
        tile_idx_in_chunk_start -= MM_CHUNK_TILES(chunk_idx_start, N_chunks, N_tiles_per_chunk);
        chunk_idx_start++;
    }

    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {"""),
    ("""            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++, tile_idx_in_chunk++) {
                // If we've reached the end of the current chunk, move to the next one
                if (tile_idx_in_chunk >= chunk_shape.logical_d1) {
                    tile_idx_in_chunk = 0;
                    chunk_idx++;  // Move to next chunk; if chunk is past last one then next branch will skip padding
                }

                if (chunk_idx >= N_chunks) {
                    break;
                }

                uint32_t tile_id = m_tile * chunk_shape.logical_d1 + tile_idx_in_chunk;
                // Compile-time dispatch preserving concrete types
                write_tile_to_chunk(
                    accessors, chunk_idx, tile_id, out_read_ptr, std::index_sequence_for<Accessors...>{});

                out_read_ptr += tile_size_bytes;
            }""",
     """            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++, tile_idx_in_chunk++) {
                // If we've reached the end of the current chunk, move to the next one
                if (tile_idx_in_chunk >= MM_CHUNK_TILES(chunk_idx, N_chunks, N_tiles_per_chunk)) {
                    tile_idx_in_chunk = 0;
                    chunk_idx++;  // Move to next chunk; if chunk is past last one then next branch will skip padding
                }

                if (chunk_idx >= N_chunks) {
                    break;
                }

                uint32_t tile_id =
                    m_tile * MM_CHUNK_TILES(chunk_idx, N_chunks, N_tiles_per_chunk) + tile_idx_in_chunk;
                // Compile-time dispatch preserving concrete types
                write_tile_to_chunk(
                    accessors, chunk_idx, tile_id, out_read_ptr, std::index_sequence_for<Accessors...>{});
                mm_write_seq++;

                out_read_ptr += tile_size_bytes;
            }"""),
]

# A single counter for the whole kernel, so the alternation keeps striping across block
# boundaries instead of restarting in phase on every block.
# The OUTPUT half of both DM kernels, halved on the N axis under MM_GATE and the stock expression
# without it. Both files carry byte-identical writer bodies, and either can be the output writer
# depending on `transpose_core_grid`, so both get the same treatment. MM_GATE with FUSE_TERNARY is
# not supported: `read_ternary_blocks_sync` reads `out_shape`, which is the post-gate shape here.
OUT_EDITS = [
    ("""    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);""",
     """    const TensorShape2D out_shape(
        M_tiles, MM_OUT_N(N_tiles), padded_M_tiles, MM_OUT_N(padded_N_tiles));"""),
    ("""    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;""",
     """    constexpr uint32_t out_N_block_tiles = MM_OUT_N(N_block_tiles);
    constexpr uint32_t out_block_num_tiles = M_block_tiles * out_N_block_tiles;"""),
    # deferred write, single destination
    ("""                            write_block_sync<M_block_tiles, N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);""",
     """                            write_block_sync<M_block_tiles, out_N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                MM_OUT_N(defer_write_n_tile),
                                MM_OUT_N(defer_write_n_tile_end));"""),
    # deferred write, split destinations
    ("""                            write_block_sync_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);""",
     """                            write_block_sync_split<M_block_tiles, out_N_block_tiles, N_chunks, N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                MM_OUT_N(defer_write_n_tile),
                                MM_OUT_N(defer_write_n_tile_end));"""),
    # immediate write, single destination
    ("""                        write_block_sync_granular<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);""",
     """                        write_block_sync_granular<M_block_tiles, out_N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            MM_OUT_N(n_tile),
                            MM_OUT_N(n_tile_end));"""),
    # immediate write, split destinations
    ("""                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);""",
     """                        write_block_sync_granular_split<M_block_tiles, out_N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            MM_OUT_N(n_tile),
                            MM_OUT_N(n_tile_end));"""),
]

# The gate epilogue itself. `copy_block` is UNTOUCHED and keeps doing what it always did -- read the
# fp32 accumulator, pack to bf16 -- only into a scratch CB instead of straight out. So the gate's
# operands carry exactly today's bf16 mantissa, which is variant B of the pre-registration; gating
# off the fp32 accumulator directly is variant A and is a different kernel, not a flag.
GATE_BLOCK = r"""
// --- tt-bio MM_GATE: the fused gate epilogue, see kernels/mm_split/patch_mm_split.py ------------
// out[m, n/2] = p * sigmoid(g), over tile-interleaved (p, g) pairs on the N axis.
//
// This is stages 1-2 of `tt_bio/kernels/reblock_permute_gated/compute_reblock_permute_gated.cpp`
// with its two CB round trips collapsed into DST. Its stage 3 (`transpose_wh`) is NOT here: this
// arm writes the original [.., .., slice_c] layout and leaves the reblock to the plain move.
//
// THREE DEVIATIONS from the move it replaces, all documented rather than discovered:
//   1. `calculate_sigmoid` branches on `fp32_dest_acc_en` at compile time and the in-projection
//      runs it TRUE, so this takes the accurate branch where ttnn takes the cheap one -- a full
//      bf16 ulp on 10.4 % of elements (measured over 3.28 M, perf/trimul_f2/e6_diag.py).
//   2. sigmoid(g) stays in DST instead of being packed to bf16 before the multiply, so the
//      product sees more mantissa than ttnn's. Costs nothing; a bf16-matched variant is one
//      extra CB round trip and is named, not built.
//   3. the packer breaks ties away from zero where ttnn breaks to even, 0.91 % of elements at a
//      1.85 % tie rate.
// All three push TOWARD float64, which is exactly why they are scored in Angstrom against a seed
// floor and not argued from a reference. `c12-fused-eltwise-at-pin` failed while being more
// accurate than what it replaced.
void gate_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    constexpr uint32_t P_DST = 0;
    constexpr uint32_t G_DST = 1;
    pack_reconfig_data_format(out_cb);
    reconfig_data_format_srca(in_cb);

    const uint32_t out_N_block_tiles = N_block_tiles >> 1;
    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < out_N_block_tiles; n++) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(in_cb);
            copy_tile(in_cb, tile_id, P_DST);
            copy_tile(in_cb, tile_id + 1, G_DST);
            sigmoid_tile_init();
            sigmoid_tile(G_DST);
            mul_binary_tile_init();
            mul_binary_tile(P_DST, G_DST, P_DST);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(P_DST, out_cb);
            tile_regs_release();
            tile_id += 2;
        }
        cb_push_back(out_cb, out_N_block_tiles);
    }
}
// -----------------------------------------------------------------------------------------------
"""

COMPUTE_EDITS = [
    # the epilogue itself, ahead of kernel_main
    ("""void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);""",
     GATE_BLOCK + """
void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);"""),
    # c_7 is the scratch, the first index minimal_matmul leaves free (c_4..c_6 are bias/ternary).
    ("""    constexpr uint32_t ternary_b_cb = tt::CBIndex::c_6;""",
     """    constexpr uint32_t ternary_b_cb = tt::CBIndex::c_6;
    constexpr uint32_t gate_cb = tt::CBIndex::c_7;"""),
    # route the no-bias no-ternary output stage through the scratch CB and then the gate
    ("""            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);""",
     """#ifdef MM_GATE
            cb_reserve_back(gate_cb, out_block_num_tiles);
            copy_block(intermediate_cb, gate_cb, M_block_tiles, N_block_tiles);
            cb_wait_front(gate_cb, out_block_num_tiles);
            gate_block(gate_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_pop_front(gate_cb, out_block_num_tiles);
#else
            copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
#endif"""),
]


HPP_COUNTER = ("""template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync(""",
               """static uint32_t mm_write_seq = 0;

template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync(""")

CPP_EDITS = [("""    noc_async_write_barrier();
    noc_async_atomic_barrier();""",
              """    MM_WRITE_BARRIER();
    noc_async_atomic_barrier();""")]


def apply(text, edits, path):
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit("patch site not unique in %s: %r" % (path, old[:60]))
        text = text.replace(old, new)
    return text


def main():
    sys.path.insert(0, ".")
    from tt_bio.mm_generic import _kernel_dir
    src = _kernel_dir()
    OUT.mkdir(parents=True, exist_ok=True)

    anchor = '#include "api/dataflow/dataflow_api.h"\n'
    hpp = (src / "matmul_dataflow_common.hpp").read_text()
    if hpp.count(anchor) != 1:
        raise SystemExit("include anchor not unique in matmul_dataflow_common.hpp")
    hpp = hpp.replace(anchor, anchor + HELPER)
    hpp = apply(hpp, [HPP_COUNTER] + HPP_EDITS, "matmul_dataflow_common.hpp")
    (OUT / "matmul_dataflow_common.hpp").write_text(hpp)

    for name in ("dm_in0_sender.cpp", "dm_in1_sender_out.cpp"):
        t = (src / name).read_text()
        t = apply(t, CPP_EDITS + OUT_EDITS, name)
        (OUT / name).write_text(t)

    compute = apply((src / "compute.cpp").read_text(), COMPUTE_EDITS, "compute.cpp")
    (OUT / "compute.cpp").write_text(compute)

    print("wrote", OUT, "from", src)


if __name__ == "__main__":
    main()
