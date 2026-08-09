"""Fused trimul input op: one kernel for the chunk, the gate, the mask and the channel move.

The triangle contraction needs channel-major operands, and getting there from the fused
projection output costs five full passes over the pair tensor in ttnn: a 4-way chunk, two
gated multiplies, a mask multiply and two permutes. Per channel chunk at 298 aa that is
0.6725 ms and 210 MB of L1 traffic. This op does it in one launch that reads the projection
output once and writes both operands, 79 MB, in 0.3404 ms including the one transpose left
outside, so 1.98x, bit-exact against the ttnn chain.

The kernel is built with ttnn.generic_op from inline sources, so there is no tt-metal build
step. Three things about it are worth knowing before changing it:

* The channel move is a cube transpose, and it needs one tile-index to intra-tile-row
  exchange that cannot be expressed as whole-tile traffic (the input has (j,c) intra-tile,
  the output needs (i,j), only j is shared, and no tile-local op changes which axes are
  intra-tile). The writer does that exchange in local L1 at 32 B per piece, which is 12% of
  the op. Widening the piece needs an untilize/tilize pair around it.
* The mask is pre-broadcast to [1,H,H,32] so it rides the same unpack-transpose path as the
  g and p tiles. That keeps the compute kernel free of unpacker reconfiguration.
* A ProgramDescriptor is immutable: mutating its runtime args in place is silently ignored
  and produces wrong results. Hence the address-keyed cache below. It works because the L1
  allocator hands back the same addresses for a repeated alloc/dealloc pattern.
"""

import ttnn

TILE_BYTES = 2048

READER_SRC = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t gp_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_g = get_arg_val<uint32_t>(1);
    const uint32_t n_g     = get_arg_val<uint32_t>(2);
    const uint32_t mk_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_in = 0;
    constexpr auto gp_args = TensorAccessorArgs<0>();
    const uint32_t tb = get_local_cb_interface(cb_in).fifo_page_size;
    const auto gp = TensorAccessor(gp_args, gp_addr, tb);
#ifdef MASKED
    // mask_bc is [1,H,H,32]: one col-tile, value mask[i,j] repeated across the 32
    // channels, so transpose_wh gives (c,j) with mask[i,j] -- the same unpack path the
    // g and p tiles take, no config switch in the compute kernel.
    const auto mk = TensorAccessor(gp_args, mk_addr, tb);
#endif
    for (uint32_t g = first_g; g < first_g + n_g; ++g) {
        const uint32_t o   = g / GROUPS_PER_OP;
        const uint32_t id  = g % GROUPS_PER_OP;
        const uint32_t Cg  = id % CT_PER_OP;
        const uint32_t J   = (id / CT_PER_OP) % N_JT;
        const uint32_t I   = id / (CT_PER_OP * N_JT);
        const uint32_t ctg = o * CT_PER_OP + Cg;
        const uint32_t ctp = P_CT_OFF + o * CT_PER_OP + Cg;
        // RB i's per barrier: 2 tiles per barrier is read-latency-bound, not
        // bandwidth-bound (milestone 2 measured 460 vs 906 GB/s for the same structure).
#ifdef MASKED
        const uint32_t per = (o == 0) ? 3 : 2;   // the mask applies to `a` only
#else
        const uint32_t per = 2;
#endif
        for (uint32_t ii = 0; ii < 32; ii += RB) {
            cb_reserve_back(cb_in, per * RB);
            uint32_t l1 = get_write_ptr(cb_in);
            for (uint32_t k = 0; k < RB; ++k) {
                const uint32_t i = I * 32 + ii + k;
                const uint32_t row = i * (N_JT * GP_COL_TILES) + J * GP_COL_TILES;
                noc_async_read(gp.get_noc_addr(row + ctg), l1, tb);
                noc_async_read(gp.get_noc_addr(row + ctp), l1 + tb, tb);
#ifdef MASKED
                if (o == 0) {
                    noc_async_read(mk.get_noc_addr(i * N_JT + J), l1 + 2 * tb, tb);
                }
#endif
                l1 += per * tb;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, per * RB);
        }
    }
}
"""

COMPUTE_SRC = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t cb_in  = 0;
    constexpr uint32_t cb_mid = 1;
    const uint32_t n_tiles   = get_arg_val<uint32_t>(0);
    const uint32_t n_masked  = get_arg_val<uint32_t>(1);
    constexpr uint32_t CB_BATCH = CBATCH;   // tiles per CB handshake
    init_sfpu(cb_in, cb_mid);
    for (uint32_t t0 = 0; t0 < n_tiles; t0 += CB_BATCH) {
    const uint32_t per = (t0 < n_masked) ? MASK_PER : 2;
    cb_wait_front(cb_in, per * CB_BATCH);
    cb_reserve_back(cb_mid, CB_BATCH);
    for (uint32_t t = 0; t < CB_BATCH; ++t) {
        tile_regs_acquire();
        transpose_wh_init_short(cb_in);
#ifdef NOGATE
        transpose_wh_tile(cb_in, per * t + 1, 0);   // p^T only: how much is the gate?
#else
        transpose_wh_tile(cb_in, per * t + 0, 0);   // dst0 = g^T
        transpose_wh_tile(cb_in, per * t + 1, 1);   // dst1 = p^T
        sigmoid_tile_init();
        sigmoid_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
#ifdef MASKED
        if (per == 3) {
            transpose_wh_init_short(cb_in);
            transpose_wh_tile(cb_in, per * t + 2, 2);   // dst2 = broadcast mask
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
        }
#endif
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_mid, t);
        tile_regs_release();
    }
    cb_push_back(cb_mid, CB_BATCH);
    cb_pop_front(cb_in, per * CB_BATCH);
    }
}
"""

WRITER_SRC = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr  = get_arg_val<uint32_t>(0);
    const uint32_t b_addr  = get_arg_val<uint32_t>(1);
    const uint32_t first_g = get_arg_val<uint32_t>(2);
    const uint32_t n_g     = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_mid = 1;
    constexpr uint32_t cb_out = 2;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t tb = get_local_cb_interface(cb_mid).fifo_page_size;
    for (uint32_t g = first_g; g < first_g + n_g; ++g) {
        const uint32_t o  = g / GROUPS_PER_OP;
        const uint32_t id = g % GROUPS_PER_OP;
        const uint32_t Cg = id % CT_PER_OP;
        const uint32_t J  = (id / CT_PER_OP) % N_JT;
        const uint32_t I  = id / (CT_PER_OP * N_JT);
        const auto out = TensorAccessor(out_args, (o == 0) ? a_addr : b_addr, tb);
        cb_wait_front(cb_mid, 32);
        const uint32_t mid = get_read_ptr(cb_mid);
#ifdef EXCHANGE
        // Sub-tile exchange: tile index i <-> intra-tile row c. Source is 32 mid tiles
        // holding (c,j); destination is 32 tiles holding (i,j), one per c. A 32 B piece is
        // one face row = 16 j values.
        const uint32_t dst = get_write_ptr(cb_out);
        noc_async_read_one_packet_set_state(get_noc_addr(mid), 32);
        for (uint32_t cc = 0; cc < 32; ++cc) {
            const uint32_t C2 = cc >> 4, c16 = cc & 15;
            const uint32_t ob = dst + cc * 2048;
            for (uint32_t I2 = 0; I2 < 2; ++I2) {
                for (uint32_t J2 = 0; J2 < 2; ++J2) {
                    const uint32_t sf = (C2 * 2 + J2) * 512 + c16 * 32;
                    const uint32_t df = (I2 * 2 + J2) * 512;
                    for (uint32_t i16 = 0; i16 < 16; ++i16) {
                        noc_async_read_one_packet_with_state(
                            mid + (I2 * 16 + i16) * 2048 + sf, ob + df + i16 * 32);
                    }
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t cc = 0; cc < 32; ++cc) {
            // out is [1,C,H,H]: page = c*(N_IT*N_JT) + I*N_JT + J
            noc_async_write(dst + cc * 2048,
                            out.get_noc_addr((Cg * 32 + cc) * (N_IT * N_JT) + I * N_JT + J),
                            tb);
        }
#else
        for (uint32_t ii = 0; ii < 32; ++ii) {
            // out is [1,H,C,H]: page = i*(CT_PER_OP*N_JT) + Cg*N_JT + J
            noc_async_write(mid + ii * 2048,
                            out.get_noc_addr((I * 32 + ii) * (CT_PER_OP * N_JT)
                                             + Cg * N_JT + J),
                            tb);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_mid, 32);
    }
}
"""

# Keyed on the buffer addresses because the descriptor bakes them in. The allocator repeats
# addresses for a repeated pattern, so a 298 aa fold builds these once, not 3840 times.
_PROGRAM_CACHE: dict[tuple, object] = {}
_MASK_CACHE: dict[tuple, ttnn.Tensor] = {}


def _mask_broadcast(mask: ttnn.Tensor, h: int) -> ttnn.Tensor:
    """mask [1,H,H] to [1,H,H,32], the value repeated across one tile of channels."""
    key = (mask.buffer_address(), h)
    got = _MASK_CACHE.get(key)
    if got is not None:
        return got
    m = ttnn.to_torch(mask).reshape(1, h, h, 1).expand(1, h, h, 32).contiguous()
    # DRAM, not L1: this lives for the whole fold, and 6.5 MB of permanently held L1 pushes
    # the buffer high-water mark past what minimal_matmul's own circular buffers need on a
    # 13x10 grid, which fails at enqueue rather than at allocation.
    bc = ttnn.from_torch(m, layout=ttnn.TILE_LAYOUT, device=mask.device(),
                         dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _MASK_CACHE[key] = bc
    return bc


def _program(gp, out_a, out_b, mask_bc, h, c, grid):
    gx, gy = grid
    key = (gp.buffer_address(), out_a.buffer_address(), out_b.buffer_address(),
           mask_bc.buffer_address() if mask_bc is not None else 0, h, c, gx, gy,
           out_a.memory_config().buffer_type)
    got = _PROGRAM_CACHE.get(key)
    if got is not None:
        return got
    n_t = h // 32
    ct_per_op = c // 32
    groups_per_op = n_t * n_t * ct_per_op
    n_groups = 2 * groups_per_op
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                              ttnn.CoreCoord(gx - 1, gy - 1))])
    base, rem = divmod(n_groups, gx * gy)
    per_max = 3 if mask_bc is not None else 2

    def fmt(i):
        return ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16,
                                       page_size=TILE_BYTES)

    cbs = [
        ttnn.CBDescriptor(total_size=2 * per_max * TILE_BYTES, core_ranges=cores,
                          format_descriptors=[fmt(0)]),
        # 32 tiles, not 64: inside a real block the pair tensor and the projections leave
        # only ~340 KB of per-core L1 above the buffer high-water mark, and a program whose
        # static CB region runs past it is rejected at enqueue. Single-buffering cb_mid
        # costs some reader/writer overlap and buys 64 KB.
        ttnn.CBDescriptor(total_size=32 * TILE_BYTES, core_ranges=cores,
                          format_descriptors=[fmt(1)]),
        ttnn.CBDescriptor(total_size=32 * TILE_BYTES, core_ranges=cores,
                          format_descriptors=[fmt(2)]),
    ]
    r_rt, c_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    idx, first = 0, 0
    for y in range(gy):
        for x in range(gx):
            gpc = base + (1 if idx < rem else 0)
            r_rt[x][y] = [gp.buffer_address(), first, gpc,
                          mask_bc.buffer_address() if mask_bc is not None else 0]
            c_rt[x][y] = [gpc * 32,
                          max(0, min(first + gpc, groups_per_op) - first) * 32]
            w_rt[x][y] = [out_a.buffer_address(), out_b.buffer_address(), first, gpc]
            first += gpc
            idx += 1
    defines = [("GROUPS_PER_OP", str(groups_per_op)), ("CT_PER_OP", str(ct_per_op)),
               ("N_JT", str(n_t)), ("N_IT", str(n_t)),
               ("GP_COL_TILES", str(4 * ct_per_op)), ("P_CT_OFF", str(2 * ct_per_op)),
               ("EXCHANGE", "1"), ("RB", "1"), ("CBATCH", "1"),
               ("MASK_PER", "3" if mask_bc is not None else "2")]
    if mask_bc is not None:
        defines.append(("MASKED", "1"))
    K = ttnn.KernelDescriptor
    kernels = [
        K(kernel_source=READER_SRC, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
          compile_time_args=list(ttnn.TensorAccessorArgs(gp).get_compile_time_args()),
          defines=defines, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
        K(kernel_source=COMPUTE_SRC, source_type=K.SourceType.SOURCE_CODE,
          core_ranges=cores, compile_time_args=[], defines=defines, runtime_args=c_rt,
          config=ttnn.ComputeConfigDescriptor()),
        K(kernel_source=WRITER_SRC, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
          compile_time_args=list(ttnn.TensorAccessorArgs(out_a).get_compile_time_args()),
          defines=defines, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
    ]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    _PROGRAM_CACHE[key] = pd
    return pd


def applicable(h, c, memory_config, dtype, fast_mode):
    """The kernel assumes tile-aligned dims, bf16, and an L1-resident pair tensor."""
    return (h % 32 == 0 and c % 32 == 0 and c >= 32
            and memory_config.buffer_type == ttnn.BufferType.L1
            and dtype == ttnn.bfloat16 and not fast_mode)


def fused_inputs(gp_in_fused, mask, ending, grid=(13, 10), out_in_l1=False):
    """Both gated channel-major trimul operands, from the fused projection output.

    Replaces chunk(4) + two gated multiplies + the mask multiply + both channel moves.
    `a` carries the mask, matching the reference. Bit-exact with the ttnn chain.
    """
    h = int(gp_in_fused.shape[1])
    c = int(gp_in_fused.shape[-1]) // 4
    dev = gp_in_fused.device()
    mask_bc = _mask_broadcast(mask, h) if mask is not None else None
    # The op needs gp_in_fused live while it writes both operands: 79 MB against the 52 MB
    # the ttnn chain holds once it has freed gp_in_fused after the split. At 298 aa that
    # extra 26 MB does not fit in L1 alongside z and x_norm_in, so the operands land in
    # DRAM unless the caller says otherwise.
    mc = ttnn.L1_MEMORY_CONFIG if out_in_l1 else ttnn.DRAM_MEMORY_CONFIG
    shape = ttnn.Shape([1, c, h, h])
    out_a = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
    out_b = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
    pd = _program(gp_in_fused, out_a, out_b, mask_bc, h, c, grid)
    ttnn.generic_op([gp_in_fused, out_a, out_b], pd)
    # The kernel emits both operands as permute(0,3,1,2); exactly one of them needs the
    # inner L,L swap, and that is the cheap whole-tile half of the transform.
    if ending:
        swapped = ttnn.transpose(out_a, -2, -1, memory_config=mc)
        ttnn.deallocate(out_a)
        return swapped, out_b
    swapped = ttnn.transpose(out_b, -2, -1, memory_config=mc)
    ttnn.deallocate(out_b)
    return out_a, swapped
