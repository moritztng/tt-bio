// SPDX-License-Identifier: Apache-2.0
//
// pair_swiglu_ws: fc3(silu(x @ w1) * (x @ w2)) for one 32-row tile at a time, weights resident in L1.
//
// Per row tile, for each group of SW hidden tiles: fc1 group -> silu -> c_3 (gate); fc2 group, times the gate in
// dest (dest reused as srcB, as kernels/fc12g does) -> c_4 (hidden). With all NH hidden tiles of the row in c_4,
// fc3 accumulates the CT output tiles in dest and packs them to c_16. x is read once, the output written once,
// and fc1/fc2/the product never leave the core.
//
// The gate, the hidden and the output are bf16; x and the weights are bf16 or bfp8, each CB in its own format, and
// srcA / srcB are reconfigured at every phase change. SW = 4 hidden tiles per fc1/fc2 pass, fc3 in groups of 4 output tiles
// (one fp32 dest half each), so c_z = 256 (CT = 8) runs two fc3 groups per row tile.
// WS_NO_{FC12,SILU,GATE,FC3} drop one phase for time attribution only (perf/spd_fastmode/ws_attr.py); the output
// is then wrong by design. No production call sets them.
// WS_LN (the `ws_ln` lever): the caller's LayerNorm runs here instead, per row tile, from the raw x in c_5 into
// c_0 (kernels/common/ln_prologue.hpp, shared with trimul T1/T2), so the normalised x never goes to DRAM.
// WS_RMS with it: the norm is an RMSNorm (RFdiffusion3's pair Transition); the beta CB is loaded and unused.
// WS_L1ACC (the `ws_l1acc` lever): every matmul sums at most KC K tiles in dest and the packer adds the chunks into the
// fp32 c_19, which then comes back into dest by a copy (no accumulation) for the silu, the gate multiply or the bf16
// pack. On Wormhole an fp32 dest summing all of K writes exact +-1 / 2 / 4 into ~1e-8 of the outputs (dest_guard.py).
// WS_RESID (the `ws_resid` lever): fc3's fp32 dest is seeded with the residual's row tiles from c_20 (an exact copy)
// and fc3 accumulates on top, so z + t is rounded to bf16 once, by the pack; the writer puts it back over z. The
// transition's `z + t` and the hidden-chunk sum then cost no add pass over the pair. (Adding after fc3 with dest
// reused as srcA truncates t to srcA's bf16 first: rel_rms 0.0037 -> 0.0046 at c 256.)
// WS_RESID2: the seed is c_20 + c_24 (z and the chained partial of the earlier hidden chunks), added by the FPU into
// the fp32 dest, so z + partial + t is rounded once and the transition's residual add is not a pass of its own.
// WS_RELU (AF2's Transition): fc3(relu(x @ w1 + b1)) + b3. c_1 holds w1 alone and there is no gate; each bias is a
// CB of tiles whose 32 rows all carry the vector (c_21 = b1, NH tiles; c_22 = b3, CT tiles), copied into dest as the
// seed its matmul accumulates on, so the bias costs no pass and is summed in fp32 with the products.
// WS_RES32 (with WS_RELU and WS_RESID; RFD4's float32 pair stream): the residual c_20 is float32, unpacked straight to
// dest (UnpackToDestFp32, set by the host) beside fc3's unrounded fp32 result and added by the SFPU, so z + t is one
// float32 rounding; the sum is packed twice, float32 to c_16 (written over z) and bfloat16 to c_25 (written over x,
// the copy the next sublayer reads). Four dest slots a group, so fc3 runs two output tiles at a time.
// WS_RES_NOCOPY (with WS_RES32; RFD4's bfloat16 pair stream, `pair_bf16z`): c_20 and c_16 are bfloat16 and the residual
// is x itself, so the sum is rounded once to bfloat16 and written over x; there is no copy (c_25 is not built).
// WS_RELU_BW (that Transition's backward to its normed input): ((x @ w1 + b1 > 0) * (dy @ w2t)) @ w3 with w12 =
// [w1 | w2^T] and w3 = w1^T, the SwiGLU program with b1 seeding fc1, a step function for the silu, and fc2 reading
// the cotangent dy (c_23, KT tiles a row) instead of x. The hidden pre-activation is recomputed, never stored.
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/comp.h"
#ifdef WS_RES32
#include "api/compute/eltwise_binary_sfpu.h"
#endif

#if defined(WS_PACK_SILU) && defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_silu.h"   // the packer-side SFPU init, as kernels/fc12g
#endif

// Dest 0..N-1 = the residual tiles i0..i0+N-1 of c_20 (+ c_24's under WS_RESID2), then the matmul `in0 x in1` is
// re-armed: the seed fc3 accumulates on.
template <uint32_t N>
void seed_res(uint32_t cb_res, uint32_t in0, uint32_t in1, uint32_t K, uint32_t i0) {
#ifdef WS_RESID2
    constexpr uint32_t cb_res2 = 24;
    add_tiles_init(cb_res, cb_res2);
    reconfig_data_format(in1, cb_res, in0, cb_res2);
    for (uint32_t i = 0; i < N; ++i) {
        add_tiles(cb_res, cb_res2, i0 + i, i0 + i, i);
    }
    mm_block_init_short(in0, in1, false, N, 1, K);
    reconfig_data_format(cb_res, in1, cb_res2, in0);
#else
    copy_tile_to_dst_init_short(cb_res);
    reconfig_data_format_srca(in1, cb_res);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(cb_res, i0 + i, i);
    }
    mm_block_init_short(in0, in1, false, N, 1, K);
    reconfig_data_format_srca(cb_res, in1);
#endif
}
#include "../trimul_gin_moved/sigmoid_poly.hpp"

#ifdef WS_L1ACC
// cb_acc holds SLOT tiles (one fp32 dest half) and every result takes all of them, N <= SLOT used. The packer and
// copy_tile index past the CB's pointer without wrapping, so a 2-tile fc3 result (c_z 64, OG 2) pushed as 2 would leave
// the next 4-tile result straddling the CB's end and writing into c_20, the residual.
constexpr uint32_t KC = 2, cb_acc = 19, SLOT = 4;

// sum over k of in0[k] x in1[k * s + b], N output tiles, into cb_acc: KC K tiles per dest pass, the packer adds the passes
// With `seed` (a CB holding N tiles at its front) the first pass starts from them instead of zero.
template <uint32_t K, uint32_t N>
void mm_acc(uint32_t in0, uint32_t in1, uint32_t s, uint32_t b, int32_t seed = -1) {
    static_assert(K % KC == 0 && N <= SLOT);
    mm_block_init_short(in0, in1, false, N, 1, K);
    reconfig_data_format(in1, in0);
    pack_reconfig_data_format(cb_acc);
    cb_reserve_back(cb_acc, SLOT);
    for (uint32_t k0 = 0; k0 < K; k0 += KC) {
        tile_regs_acquire();
        if (seed >= 0 && k0 == 0) {
            seed_res<N>(seed, in0, in1, K, 0);
        }
        for (uint32_t k = k0; k < k0 + KC; ++k) {
            matmul_block(in0, in1, k, k * s + b, 0, false, N, 1, K);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_l1_acc(k0 > 0);
        for (uint32_t i = 0; i < N; ++i) {
            pack_tile<true>(i, cb_acc, i);
        }
        tile_regs_release();
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, SLOT);
}

// cb_acc's N summed tiles into dest 0..N-1; the caller holds the dest
template <uint32_t N>
void acc_to_dest() {
    cb_wait_front(cb_acc, SLOT);
    copy_tile_to_dst_init_short(cb_acc);
    reconfig_data_format_srca(cb_acc);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(cb_acc, i, i);
    }
    cb_pop_front(cb_acc, SLOT);
}
#endif

#ifdef WS_PACK_SILU
// WS_PACK_SILU (the `ws_psilu` lever; not with RELU or RELU_BW): the silu runs on the packer thread's SFPU
// calls (kernels/fc12g's pattern) instead of the math thread's, and fc1 of group j + SW is issued before fc2 of group
// j, so the FPU's matmuls overlap the silu instead of waiting for it. The gate CB's 2 SW tiles hold the two groups in
// flight. Same SFPU instructions on the same dest values: bit-identical to the math-thread silu. With WS_LN the math
// thread's LayerNorm SFPU work never meets the packer's: a row's last silu is packed into the gate CB before that
// row's fc2 can finish, and the next row's fc1 (the packer's next silu) waits on the LN's packed x. With WS_L1ACC
// the matmul's K chunks go through c_19 as on the math-thread path and the silu reads the same fp32 copy in dest.
template <uint32_t KT, uint32_t NH, uint32_t SW, uint32_t V>
void fc1_gate(uint32_t j) {
    constexpr uint32_t cb_x = 0, cb_w12 = 1, cb_gate = 3;
#ifdef WS_L1ACC
    mm_acc<KT, SW>(cb_x, cb_w12, 2 * NH, j);
    tile_regs_acquire();
    acc_to_dest<SW>();
    pack_reconfig_data_format(cb_gate);
#else
    mm_block_init_short(cb_x, cb_w12, false, SW, 1, KT);
    reconfig_data_format(cb_w12, cb_x);
    tile_regs_acquire();
    for (uint32_t k = 0; k < KT; ++k) {
        matmul_block(cb_x, cb_w12, k, k * 2 * NH + j, 0, false, SW, 1, KT);
    }
#endif
    tile_regs_commit();
    cb_reserve_back(cb_gate, SW);
    // tile_regs_wait() that also points the SFPU at the packer's dest half, then the silu before the pack
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
#ifndef WS_NO_SILU
    for (uint32_t i = 0; i < SW; ++i) {
        silu_poly_v_tile_pack<V>(i);
    }
#endif
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_tile_block(0, cb_gate, SW);
    tile_regs_release();
    cb_push_back(cb_gate, SW);
}
#endif
#ifdef WS_LN
#include "../common/ln_prologue.hpp"

struct LnCbs {
    static constexpr uint32_t Kt = get_compile_time_arg_val(0), x = 5, gamma = 8, beta = 9, scaler = 10, eps = 11,
                              mean = 12, var = 13, rstd = 14, xc = 17, tmp = 18;
};
#endif


void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    // 0: metal's silu_tile, 1: sigmoid_poly.hpp's degree-8 silu (3e-4), 2: its degree-6 fp16-coefficient silu (7.4e-4)
    constexpr uint32_t SILU_POLY = get_compile_time_arg_val(3);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t SW = 4;
#ifdef WS_RES32
    constexpr uint32_t OG = CT < 2 ? CT : 2, cb_outb = 25;
#else
    constexpr uint32_t OG = CT < 4 ? CT : 4;
#endif
    static_assert(NH % SW == 0 && CT % OG == 0);
    constexpr uint32_t cb_x = 0, cb_w12 = 1, cb_w3 = 2, cb_gate = 3, cb_hid = 4, cb_out = 16, cb_res = 20;
#ifdef WS_RELU
    constexpr uint32_t cb_b1 = 21, cb_b3 = 22, W12 = NH;   // c_1 is w1 alone, [KT, NH]
#else
    constexpr uint32_t W12 = 2 * NH;
#endif
#ifdef WS_RELU_BW
    constexpr uint32_t cb_b1 = 21, cb_dy = 23, cb_fc2 = cb_dy;
#else
    constexpr uint32_t cb_fc2 = cb_x;
#endif
    // The gate multiply keeps HiFi4 whatever the matmuls run at: at HiFi2 it would drop the gate's low bits.
    constexpr auto mul_fidelity = MathFidelity::HiFi4;
    if (count == 0) {
        return;
    }
#ifdef WS_RELU
    mm_block_init(cb_x, cb_w12, cb_hid, false, SW, 1, KT);   // no gate CB in this mode
#else
    mm_block_init(cb_x, cb_w12, cb_gate, false, SW, 1, KT);
#endif
#ifdef WS_PACK_SILU
    PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));   // the packer thread's SFPU address modes, as fc12g
#endif
    cb_wait_front(cb_w12, KT * W12);
    cb_wait_front(cb_w3, NH * CT);
#ifdef WS_RELU
    cb_wait_front(cb_b1, NH);
    cb_wait_front(cb_b3, CT);
#endif
#ifdef WS_RELU_BW
    cb_wait_front(cb_b1, NH);
#endif
#ifdef WS_LN
    cb_wait_front(LnCbs::gamma, KT);
    cb_wait_front(LnCbs::beta, KT);
    cb_wait_front(LnCbs::scaler, 1);
    cb_wait_front(LnCbs::eps, 1);
#endif

    for (uint32_t r = 0; r < count; ++r) {
#ifdef WS_LN
        cb_reserve_back(cb_x, KT);
#ifdef WS_RMS
        ln_prologue::rms_norm_row<LnCbs>(cb_x, [](uint32_t i) { pack_tile<true>(0, cb_x, i); });
#else
        ln_prologue::layer_norm_row<LnCbs>(cb_x, [](uint32_t i) { pack_tile<true>(0, cb_x, i); });
#endif
        cb_push_back(cb_x, KT);
#endif
        cb_wait_front(cb_x, KT);
#ifdef WS_RELU_BW
        cb_wait_front(cb_dy, KT);
#endif
#ifdef WS_PACK_SILU
        static_assert(SILU_POLY == 1 || SILU_POLY == 2);
        fc1_gate<KT, NH, SW, SILU_POLY>(0);
        for (uint32_t j = 0; j < NH; j += SW) {
            if (j + SW < NH) {
                fc1_gate<KT, NH, SW, SILU_POLY>(j + SW);
            }
#ifdef WS_L1ACC
            mm_acc<KT, SW>(cb_x, cb_w12, 2 * NH, NH + j);
            tile_regs_acquire();
            acc_to_dest<SW>();
            pack_reconfig_data_format(cb_hid);
            cb_wait_front(cb_gate, SW);
            reconfig_data_format_srca(cb_acc, cb_gate);
#else
            mm_block_init_short(cb_x, cb_w12, false, SW, 1, KT);
            reconfig_data_format(cb_w12, cb_x);
            tile_regs_acquire();
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w12, k, k * 2 * NH + NH + j, 0, false, SW, 1, KT);
            }
            cb_wait_front(cb_gate, SW);
            reconfig_data_format_srca(cb_w12, cb_gate);
#endif
            UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                false, false, cb_gate)));
            MATH((llk_math_eltwise_binary_init<ELWMUL, NONE, mul_fidelity, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                false)));
            for (uint32_t i = 0; i < SW; ++i) {
                UNPACK((llk_unpack_A<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_gate, i)));
                MATH((llk_math_eltwise_binary<ELWMUL, NONE, DST_ACCUM_MODE, mul_fidelity,
                                              EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_gate, cb_gate, i, true)));
            }
            cb_pop_front(cb_gate, SW);
            tile_regs_commit();
            cb_reserve_back(cb_hid, SW);
            tile_regs_wait();
            pack_tile_block(0, cb_hid, SW);
            tile_regs_release();
            cb_push_back(cb_hid, SW);
        }
#else
        for (uint32_t j = 0; j < NH; j += SW) {
#ifdef WS_RELU
            // fc1 group on top of b1 -> relu -> hidden
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_b1);
            reconfig_data_format_srca(cb_b1);
            for (uint32_t i = 0; i < SW; ++i) {
                copy_tile(cb_b1, j + i, i);
            }
            mm_block_init_short(cb_x, cb_w12, false, SW, 1, KT);
            reconfig_data_format(cb_w12, cb_x);
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w12, k, k * W12 + j, 0, false, SW, 1, KT);
            }
            relu_tile_init();
            for (uint32_t i = 0; i < SW; ++i) {
                relu_tile(i);
            }
            tile_regs_commit();
            cb_reserve_back(cb_hid, SW);
            tile_regs_wait();
            pack_tile_block(0, cb_hid, SW);
            tile_regs_release();
            cb_push_back(cb_hid, SW);
#else
            // fc1 group -> silu -> gate
#ifdef WS_L1ACC
            mm_acc<KT, SW>(cb_x, cb_w12, 2 * NH, j);
            tile_regs_acquire();
            acc_to_dest<SW>();
            pack_reconfig_data_format(cb_gate);
#else
#ifdef WS_RELU_BW
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_b1);
            reconfig_data_format_srca(cb_b1);
            for (uint32_t i = 0; i < SW; ++i) {
                copy_tile(cb_b1, j + i, i);
            }
            mm_block_init_short(cb_x, cb_w12, false, SW, 1, KT);
            reconfig_data_format(cb_w12, cb_x);
#else
            mm_block_init_short(cb_x, cb_w12, false, SW, 1, KT);
            reconfig_data_format(cb_w12, cb_x);  // srcA = the weights, srcB = x, each in its own format
            tile_regs_acquire();
#endif
#ifndef WS_NO_FC12
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w12, k, k * 2 * NH + j, 0, false, SW, 1, KT);
            }
#endif
#endif
#if defined(WS_RELU_BW)
            gtz_tile_init();   // relu's derivative: 1 where the pre-activation is positive, else 0
            for (uint32_t i = 0; i < SW; ++i) {
                gtz_tile(i);
            }
#elif !defined(WS_NO_SILU)
            silu_tile_init();  // the dest-reuse multiply's init does not keep the SFPU's
            for (uint32_t i = 0; i < SW; ++i) {
                if constexpr (SILU_POLY == 2) {
                    silu_poly6_tile(i);
                } else if constexpr (SILU_POLY == 1) {
                    silu_poly_tile(i);
                } else {
                    silu_tile(i);
                }
            }
#endif
            tile_regs_commit();
            cb_reserve_back(cb_gate, SW);
            tile_regs_wait();
            pack_tile_block(0, cb_gate, SW);
            tile_regs_release();
            cb_push_back(cb_gate, SW);
            // fc2 group, times the gate in dest -> hidden
#ifdef WS_L1ACC
            mm_acc<KT, SW>(cb_x, cb_w12, 2 * NH, NH + j);
            tile_regs_acquire();
            acc_to_dest<SW>();
            pack_reconfig_data_format(cb_hid);
            cb_wait_front(cb_gate, SW);
            reconfig_data_format_srca(cb_acc, cb_gate);
#else
            tile_regs_acquire();
#ifdef WS_RELU_BW
            mm_block_init_short(cb_dy, cb_w12, false, SW, 1, KT);
            reconfig_data_format(cb_w12, cb_dy);
#endif
#ifndef WS_NO_FC12
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_fc2, cb_w12, k, k * 2 * NH + NH + j, 0, false, SW, 1, KT);
            }
#endif
            cb_wait_front(cb_gate, SW);
            reconfig_data_format_srca(cb_w12, cb_gate);  // bfp8 weights leave srcA in their format
#endif
#ifndef WS_NO_GATE
            UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                false, false, cb_gate)));
            MATH((llk_math_eltwise_binary_init<ELWMUL, NONE, mul_fidelity, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                false)));
            for (uint32_t i = 0; i < SW; ++i) {
                UNPACK((llk_unpack_A<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_gate, i)));
                MATH((llk_math_eltwise_binary<ELWMUL, NONE, DST_ACCUM_MODE, mul_fidelity,
                                              EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_gate, cb_gate, i, true)));
            }
#endif
            cb_pop_front(cb_gate, SW);
            tile_regs_commit();
            cb_reserve_back(cb_hid, SW);
            tile_regs_wait();
            pack_tile_block(0, cb_hid, SW);
            tile_regs_release();
            cb_push_back(cb_hid, SW);
#endif
        }
#endif  // WS_PACK_SILU
        cb_pop_front(cb_x, KT);
#ifdef WS_RELU_BW
        cb_pop_front(cb_dy, KT);
#endif
        // fc3 over the row's NH hidden tiles, OG output tiles (one fp32 dest half) at a time
        cb_wait_front(cb_hid, NH);
#ifndef WS_L1ACC
        mm_block_init_short(cb_hid, cb_w3, false, OG, 1, NH);
        reconfig_data_format(cb_w3, cb_hid);
#endif
        for (uint32_t c = 0; c < CT; c += OG) {
#ifdef WS_RESID
            cb_wait_front(cb_res, OG);
#ifdef WS_RESID2
            cb_wait_front(24, OG);
#endif
#endif
#ifdef WS_L1ACC
#ifdef WS_RESID
            mm_acc<NH, OG>(cb_hid, cb_w3, CT, c, cb_res);
#else
            mm_acc<NH, OG>(cb_hid, cb_w3, CT, c);
#endif
            tile_regs_acquire();
            acc_to_dest<OG>();
            pack_reconfig_data_format(cb_out);
#else
            tile_regs_acquire();
#ifdef WS_RELU
            copy_tile_to_dst_init_short(cb_b3);
            reconfig_data_format_srca(cb_b3);
            for (uint32_t i = 0; i < OG; ++i) {
                copy_tile(cb_b3, c + i, i);
            }
            mm_block_init_short(cb_hid, cb_w3, false, OG, 1, NH);
            reconfig_data_format(cb_w3, cb_hid);
#endif
#if defined(WS_RESID) && !defined(WS_RES32)
            seed_res<OG>(cb_res, cb_hid, cb_w3, NH, 0);
#endif
#ifndef WS_NO_FC3
            for (uint32_t k = 0; k < NH; ++k) {
                matmul_block(cb_hid, cb_w3, k, k * CT + c, 0, false, OG, 1, NH);
            }
#endif
#endif
#ifdef WS_RES32
            copy_tile_to_dst_init_short_with_dt(cb_w3, cb_res);
            for (uint32_t i = 0; i < OG; ++i) {
                copy_tile(cb_res, i, OG + i);
            }
            add_binary_tile_init();
            for (uint32_t i = 0; i < OG; ++i) {
                add_binary_tile(i, OG + i, i);
            }
#endif
#ifdef WS_RESID
            cb_pop_front(cb_res, OG);
#ifdef WS_RESID2
            cb_pop_front(24, OG);
#endif
#endif
            tile_regs_commit();
            cb_reserve_back(cb_out, OG);
#if defined(WS_RES32) && !defined(WS_RES_NOCOPY)
            cb_reserve_back(cb_outb, OG);
#endif
            tile_regs_wait();
#ifdef WS_RES32
            pack_reconfig_data_format(cb_out);
#endif
            pack_tile_block(0, cb_out, OG);
#if defined(WS_RES32) && !defined(WS_RES_NOCOPY)
            pack_reconfig_data_format(cb_outb);
            pack_tile_block(0, cb_outb, OG);
#endif
            tile_regs_release();
            cb_push_back(cb_out, OG);
#if defined(WS_RES32) && !defined(WS_RES_NOCOPY)
            cb_push_back(cb_outb, OG);
#endif
        }
        cb_pop_front(cb_hid, NH);
    }
}
