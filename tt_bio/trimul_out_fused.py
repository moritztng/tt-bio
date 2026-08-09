"""Fused trimul output op: the channel move and the concat in one launch.

After the triangle contraction the trimul holds [1, C, N, N] per channel chunk and needs
[1, N, N, hidden]. ttnn does that in two passes: permute(0,2,3,1) to bring the channel chunk
back to the last axis, then a running concat that copies the accumulator again on every
chunk. At 298 aa (N=320, C=64, four chunks) that is 1.18 ms and 334 MB of traffic per
trimul, and the permute half runs at 17% of this card's L1 copy roof because it moves 32 B
pieces (W12: the piece is set by tile face geometry, not by the tensor).

This op reads each chunk once and writes it straight into its column stripe of a
destination allocated up front. The concat disappears -- there is no accumulator to copy --
and the channel move becomes whole-tile NOC traffic with the sub-tile work done in local L1.

Shapes, per group of 32 tiles:
  source tile (c, it, jt) holds (i, j) intra-tile, page = c*NT*NT + it*NT + jt
  dest   tile (i, jt, ct) holds (j, c) intra-tile, page = i*NT*HT + jt*HT + ct

Only j is shared between the two, so exactly one tile-index/intra-row exchange is
unavoidable (W4 proved this for the mirror-image input side). The reader does it: whole
tiles come in over the NOC, the exchange runs in local L1 at 32 B a piece, and the compute
kernel's unpacker transpose finishes the job. The writer then moves whole tiles only.

Unlike the input-side fusion this one *reduces* the live set: it deletes the permuted chunk
(13.1 MB in L1 at 298 aa) and the growing concat accumulator, and adds nothing but its CBs.
"""

import ttnn

TILE_BYTES = 2048

READER_SRC = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_g  = get_arg_val<uint32_t>(1);
    const uint32_t n_g      = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_raw = 0;   // reader scratch: 32 whole source tiles
    constexpr uint32_t cb_in  = 1;   // exchanged tiles, handed to the compute kernel
    constexpr auto src_args = TensorAccessorArgs<0>();
    const uint32_t tb = get_local_cb_interface(cb_in).fifo_page_size;
    const auto src = TensorAccessor(src_args, src_addr, tb);
    const uint32_t raw = get_write_ptr(cb_raw);

    for (uint32_t g = first_g; g < first_g + n_g; ++g) {
        const uint32_t jt = g % N_JT;
        const uint32_t it = (g / N_JT) % N_IT;
        const uint32_t ct = g / (N_JT * N_IT);          // channel tile inside this chunk
#ifdef NOEXCHANGE
        // Timing control only: the same whole-tile NOC traffic and the same transpose, with
        // the exchange skipped. The result is WRONG on purpose -- this arm exists to split
        // the op's cost into whole-tile movement and exchange.
        cb_reserve_back(cb_in, 32);
        const uint32_t direct = get_write_ptr(cb_in);
        for (uint32_t cc = 0; cc < 32; ++cc) {
#ifdef SEQREAD
            noc_async_read(src.get_noc_addr(g * 32 + cc), direct + cc * 2048, tb);
#else
            noc_async_read(src.get_noc_addr((ct * 32 + cc) * (N_IT * N_JT) + it * N_JT + jt),
                           direct + cc * 2048, tb);
#endif
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, 32);
    }
}
#else
        // 32 whole tiles, one per channel in this channel tile.
        for (uint32_t cc = 0; cc < 32; ++cc) {
            noc_async_read(src.get_noc_addr((ct * 32 + cc) * (N_IT * N_JT) + it * N_JT + jt),
                           raw + cc * 2048, tb);
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_in, 32);
        const uint32_t dst = get_write_ptr(cb_in);
        // Exchange: tile index c <-> intra-tile row i. Tile cc row ii becomes tile ii row cc.
        // A 32 B piece is one face row = 16 j values; both sides keep the same column face,
        // so j never moves and there is nothing wider to move.
        //
        // Both addresses are strength-reduced to pointer increments. Within a source tile the
        // rows of one face are 32 B apart, and the destinations for consecutive ii are one
        // tile apart, so the inner loop is two adds -- no multiply, no shift, no mask. The
        // first form of this loop recomputed both addresses per piece and the op measured
        // ~30 ns per piece per core.
        noc_async_read_one_packet_set_state(get_noc_addr(raw), 32);
        for (uint32_t cc = 0; cc < 32; ++cc) {
            const uint32_t s_base = raw + cc * 2048;
            uint32_t sa = s_base;                                     // face (0,0), row 0
            uint32_t da = dst + ((cc >> 4) * 1024) + ((cc & 15) * 32);  // row cc of tile 0
            for (uint32_t ii = 0; ii < 16; ++ii) {
                noc_async_read_one_packet_with_state(sa, da);
                noc_async_read_one_packet_with_state(sa + 512, da + 512);
                sa += 32;
                da += 2048;
            }
            sa = s_base + 1024;                                       // face (1,0), row 0
            for (uint32_t ii = 16; ii < 32; ++ii) {
                noc_async_read_one_packet_with_state(sa, da);
                noc_async_read_one_packet_with_state(sa + 512, da + 512);
                sa += 32;
                da += 2048;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, 32);
    }
}
#endif
"""

COMPUTE_SRC = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t cb_in  = 1;
    constexpr uint32_t cb_mid = 2;
    const uint32_t n_g = get_arg_val<uint32_t>(0);
    init_sfpu(cb_in, cb_mid);
    for (uint32_t g = 0; g < n_g; ++g) {
        cb_wait_front(cb_in, 32);
        cb_reserve_back(cb_mid, 32);
        // RBATCH tiles per dst-register acquire. One tile per acquire makes the
        // acquire/commit/wait/release handshake the per-tile cost, and this op pushes 25,600
        // tiles per trimul through it.
        for (uint32_t t0 = 0; t0 < 32; t0 += RBATCH) {
            tile_regs_acquire();
            transpose_wh_init_short(cb_in);
            for (uint32_t r = 0; r < RBATCH; ++r) {
                transpose_wh_tile(cb_in, t0 + r, r);   // (c,j) -> (j,c)
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < RBATCH; ++r) {
                pack_tile(r, cb_mid, t0 + r);
            }
            tile_regs_release();
        }
        cb_push_back(cb_mid, 32);
        cb_pop_front(cb_in, 32);
    }
}
"""

WRITER_SRC = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_g  = get_arg_val<uint32_t>(1);
    const uint32_t n_g      = get_arg_val<uint32_t>(2);
    const uint32_t ct_off   = get_arg_val<uint32_t>(3);   // column tile offset = the concat
    constexpr uint32_t cb_mid = 2;
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const uint32_t tb = get_local_cb_interface(cb_mid).fifo_page_size;
    const auto dst = TensorAccessor(dst_args, dst_addr, tb);
    for (uint32_t g = first_g; g < first_g + n_g; ++g) {
        const uint32_t jt = g % N_JT;
        const uint32_t it = (g / N_JT) % N_IT;
        const uint32_t ct = g / (N_JT * N_IT);
        cb_wait_front(cb_mid, 32);
        const uint32_t mid = get_read_ptr(cb_mid);
        for (uint32_t ii = 0; ii < 32; ++ii) {
#ifdef SEQWRITE
            // Timing control only, WRONG data. Walks the destination with a chosen page
            // stride instead of the real stride-(N_JT*HT) one. Stride 1 touches every DRAM
            // bank, stride 8 touches one; the curve between them says how much bank spread
            // is worth, which is what a cross-chunk batching change would buy.
            noc_async_write(mid + ii * 2048,
                            dst.get_noc_addr(((g * 32 + ii) * SEQSTRIDE) % DSTPAGES), tb);
#else
            noc_async_write(mid + ii * 2048,
                            dst.get_noc_addr((it * 32 + ii) * (N_JT * HT)
                                             + jt * HT + ct_off + ct),
                            tb);
#endif
        }
        noc_async_write_barrier();
        cb_pop_front(cb_mid, 32);
    }
}
"""

# A ProgramDescriptor bakes in the buffer addresses and mutating its runtime args is
# silently ignored (W4 lost a day to that), so cache on the addresses. The allocator hands
# back the same addresses for a repeated alloc/dealloc pattern, so a fold builds these once.
_PROGRAM_CACHE: dict[tuple, object] = {}


def _program(src, dst, n, c, ct_off, grid, no_exchange=False, rbatch=8, depth=2,
             seq_read=False, seq_write=False, seq_stride=1):
    gx, gy = grid
    key = (src.buffer_address(), dst.buffer_address(), n, c, ct_off, gx, gy,
           int(dst.shape[-1]), src.memory_config().buffer_type, dst.memory_config().buffer_type,
           no_exchange, rbatch, depth, seq_read, seq_write, seq_stride)
    got = _PROGRAM_CACHE.get(key)
    if got is not None:
        return got
    nt = n // 32
    ct_per_chunk = c // 32
    ht = int(dst.shape[-1]) // 32
    n_groups = ct_per_chunk * nt * nt
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                              ttnn.CoreCoord(gx - 1, gy - 1))])
    base, rem = divmod(n_groups, gx * gy)

    def fmt(i):
        return ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16,
                                       page_size=TILE_BYTES)

    # cb_raw is reader scratch and is never handed on, so it stays one group deep. cb_in and
    # cb_mid are `depth` groups deep: at depth 1 the reader, compute kernel and writer run
    # strictly one after another per group and the op costs read + transpose + write instead
    # of their maximum. At depth 2 that is (32 + 64 + 64) x 2 KB = 320 KB per core, against
    # the ~340 KB a real Pairformer block leaves above its buffer high-water mark.
    cbs = [ttnn.CBDescriptor(total_size=32 * TILE_BYTES, core_ranges=cores,
                             format_descriptors=[fmt(0)])]
    cbs += [ttnn.CBDescriptor(total_size=depth * 32 * TILE_BYTES, core_ranges=cores,
                              format_descriptors=[fmt(i)]) for i in (1, 2)]
    r_rt, c_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    idx, first = 0, 0
    for y in range(gy):
        for x in range(gx):
            gpc = base + (1 if idx < rem else 0)
            r_rt[x][y] = [src.buffer_address(), first, gpc]
            c_rt[x][y] = [gpc]
            w_rt[x][y] = [dst.buffer_address(), first, gpc, ct_off]
            first += gpc
            idx += 1
    defines = [("N_IT", str(nt)), ("N_JT", str(nt)), ("HT", str(ht)), ("RBATCH", str(rbatch))]
    if no_exchange:
        defines.append(("NOEXCHANGE", "1"))
    if seq_read:
        defines.append(("SEQREAD", "1"))
    if seq_write:
        defines.append(("SEQWRITE", "1"))
        defines.append(("SEQSTRIDE", str(seq_stride)))
        defines.append(("DSTPAGES", str(int(dst.shape[1]) * nt * ht)))
    K = ttnn.KernelDescriptor
    kernels = [
        K(kernel_source=READER_SRC, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
          compile_time_args=list(ttnn.TensorAccessorArgs(src).get_compile_time_args()),
          defines=defines, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
        K(kernel_source=COMPUTE_SRC, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
          compile_time_args=[], defines=defines, runtime_args=c_rt,
          config=ttnn.ComputeConfigDescriptor()),
        K(kernel_source=WRITER_SRC, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
          compile_time_args=list(ttnn.TensorAccessorArgs(dst).get_compile_time_args()),
          defines=defines, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
    ]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    _PROGRAM_CACHE[key] = pd
    return pd


def applicable(n, c, hidden, dtype, fast_mode):
    """Tile-aligned dims and bf16. The destination may be in L1 or DRAM."""
    return (n % 32 == 0 and c % 32 == 0 and hidden % c == 0
            and dtype == ttnn.bfloat16 and not fast_mode)


def fused_output(x_chunk, dst, chunk_index, grid=(13, 10), no_exchange=False,
                 rbatch=8, depth=2, seq_read=False, seq_write=False, seq_stride=1):
    """Write permute(x_chunk, (0,2,3,1)) into columns [chunk_index*C, +C) of `dst`.

    x_chunk is [1, C, N, N]; dst is [1, N, N, hidden] and is returned unchanged so the
    caller can keep the ttnn dataflow shape.

    `no_exchange` skips the sub-tile exchange and produces WRONG data. It exists so the op's
    cost can be split into whole-tile movement and exchange; never set it in the model.
    """
    n = int(x_chunk.shape[2])
    c = int(x_chunk.shape[1])
    ct_off = chunk_index * (c // 32)
    pd = _program(x_chunk, dst, n, c, ct_off, grid, no_exchange, rbatch, depth,
                  seq_read, seq_write, seq_stride)
    ttnn.generic_op([x_chunk, dst], pd)
    return dst
