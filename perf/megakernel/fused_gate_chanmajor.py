#!/usr/bin/env python3
"""W4 milestone 2: the fused trimul input-side op, as a generic_op.

Replaces, per channel chunk, `ttnn.chunk(gp_in_fused, 4, -1)` + two
`multiply_(p, g, SIGMOID)` + two channel-major permutes -- 0.585 ms and 210 MB of L1
traffic at N=320 -- with one launch that reads `gp_in_fused` once and writes both gated
operands, 79 MB.

Two arms, so the design bisects:

  v1a  read the g and p tiles for one channel chunk, gate them (sigmoid on g, multiply),
       transpose each tile with the unpacker, write whole tiles. Output is [1,H,C,H],
       i.e. (i,c,j): the whole-tile half of the channel-major transform. The caller still
       needs one `permute(0,2,1,3)` to finish. Proves the reader/compute/writer pipeline
       and the gate arithmetic.
  v1b  the same, plus the sub-tile exchange in the writer, so the output is [1,C,H,H] and
       nothing is left for the caller. Milestone 1 measured this exchange at 955 GB/s with
       256 B pieces; here the pieces are 32 B, because a tile row of 32 bf16 spans two
       16-wide faces and nothing has been untilized yet. 32 B is the slow end of that
       curve (213.9 GB/s) and widening it is milestone 2b.

Correctness is checked two ways: against torch (`p * sigmoid(g)` then the permute) and
against the actual ttnn op chain being replaced, bit-exactly where that holds. The gate is
the one place bit-exactness can legitimately fail: ttnn multiplies on the FPU after an SFPU
sigmoid, this kernel does both on the SFPU, and the two need not round identically. That is
measured here, not assumed.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/fused_gate_chanmajor.py --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

from tt_bio.tenstorrent import get_device

TB = 2048  # bf16 tile bytes

READER = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t gp_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_g = get_arg_val<uint32_t>(1);
    const uint32_t n_g     = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_in = 0;
    constexpr auto gp_args = TensorAccessorArgs<0>();
    const uint32_t tb = get_local_cb_interface(cb_in).fifo_page_size;
    const auto gp = TensorAccessor(gp_args, gp_addr, tb);
    for (uint32_t g = first_g; g < first_g + n_g; ++g) {
        const uint32_t o   = g / GROUPS_PER_OP;
        const uint32_t id  = g % GROUPS_PER_OP;
        const uint32_t Cg  = id % CT_PER_OP;
        const uint32_t J   = (id / CT_PER_OP) % N_JT;
        const uint32_t I   = id / (CT_PER_OP * N_JT);
        const uint32_t ctg = o * CT_PER_OP + Cg;
        const uint32_t ctp = P_CT_OFF + o * CT_PER_OP + Cg;
        for (uint32_t ii = 0; ii < 32; ++ii) {
            const uint32_t row = (I * 32 + ii) * (N_JT * GP_COL_TILES) + J * GP_COL_TILES;
            cb_reserve_back(cb_in, 2);
            const uint32_t l1 = get_write_ptr(cb_in);
            noc_async_read(gp.get_noc_addr(row + ctg), l1, tb);
            noc_async_read(gp.get_noc_addr(row + ctp), l1 + tb, tb);
            noc_async_read_barrier();
            cb_push_back(cb_in, 2);
        }
    }
}
"""

COMPUTE = r"""
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
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    init_sfpu(cb_in, cb_mid);
    for (uint32_t t = 0; t < n_tiles; ++t) {
        cb_wait_front(cb_in, 2);
        tile_regs_acquire();
        transpose_wh_init_short(cb_in);
#ifdef NOGATE
        transpose_wh_tile(cb_in, 1, 0);   // p^T only: how much of the op is the gate?
#else
        transpose_wh_tile(cb_in, 0, 0);   // dst0 = g^T
        transpose_wh_tile(cb_in, 1, 1);   // dst1 = p^T
        sigmoid_tile_init();
        sigmoid_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
#endif
        tile_regs_commit();
        cb_reserve_back(cb_mid, 1);
        tile_regs_wait();
        pack_tile(0, cb_mid);
        tile_regs_release();
        cb_push_back(cb_mid, 1);
        cb_pop_front(cb_in, 2);
    }
}
"""

WRITER = r"""
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


def build(dev, gp, out_a, out_b, gx, gy, exchange, n_jt, n_it, ct_per_op,
          gp_col_tiles, nogate=False):
    groups_per_op = n_it * n_jt * ct_per_op
    n_groups = 2 * groups_per_op
    ncores = gx * gy
    assert n_groups % ncores == 0, f"{n_groups} groups over {ncores} cores is uneven"
    gpc = n_groups // ncores
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                              ttnn.CoreCoord(gx - 1, gy - 1))])
    fmt = lambda i: ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16,
                                            page_size=TB)
    cbs = [
        ttnn.CBDescriptor(total_size=4 * TB, core_ranges=cores, format_descriptors=[fmt(0)]),
        ttnn.CBDescriptor(total_size=64 * TB, core_ranges=cores, format_descriptors=[fmt(1)]),
        ttnn.CBDescriptor(total_size=32 * TB, core_ranges=cores, format_descriptors=[fmt(2)]),
    ]
    r_rt, c_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for y in range(gy):
        for x in range(gx):
            first = c * gpc
            r_rt[x][y] = [gp.buffer_address(), first, gpc]
            c_rt[x][y] = [gpc * 32]
            w_rt[x][y] = [out_a.buffer_address(), out_b.buffer_address(), first, gpc]
            c += 1
    defines = [("GROUPS_PER_OP", str(groups_per_op)), ("CT_PER_OP", str(ct_per_op)),
               ("N_JT", str(n_jt)), ("N_IT", str(n_it)),
               ("GP_COL_TILES", str(gp_col_tiles)),
               ("P_CT_OFF", str(2 * ct_per_op))]
    if exchange:
        defines.append(("EXCHANGE", "1"))
    if nogate:
        defines.append(("NOGATE", "1"))
    K = ttnn.KernelDescriptor
    reader = K(kernel_source=READER, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
               compile_time_args=list(ttnn.TensorAccessorArgs(gp).get_compile_time_args()),
               defines=defines, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor())
    compute = K(kernel_source=COMPUTE, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=[], defines=defines, runtime_args=c_rt,
                config=ttnn.ComputeConfigDescriptor())
    writer = K(kernel_source=WRITER, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=list(ttnn.TensorAccessorArgs(out_a).get_compile_time_args()),
                defines=defines, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--grid", default="10x10")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--arms", default="v1a,v1b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gx, gy = (int(v) for v in a.grid.split("x"))
    dev = get_device()
    N, C = a.n, a.c
    L1 = ttnn.L1_MEMORY_CONFIG
    torch.manual_seed(0)

    gp_t = torch.randn(1, N, N, 4 * C) * 0.7
    gp = ttnn.from_torch(gp_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                         memory_config=L1)
    gp_bf = gp_t.to(torch.bfloat16)
    g_a, g_b, p_a, p_b = (gp_bf[..., i * C:(i + 1) * C] for i in range(4))
    gate_a = (p_a.float() * torch.sigmoid(g_a.float())).to(torch.bfloat16)
    gate_b = (p_b.float() * torch.sigmoid(g_b.float())).to(torch.bfloat16)

    # the ttnn chain being replaced, as the bit-exactness reference
    cg_a, cg_b, cp_a, cp_b = ttnn.chunk(gp, chunks=4, dim=-1)
    m_a = ttnn.multiply(cp_a, cg_a, memory_config=L1,
                        input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    m_b = ttnn.multiply(cp_b, cg_b, memory_config=L1,
                        input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ref_a_t, ref_b_t = ttnn.to_torch(m_a), ttnn.to_torch(m_b)
    for t in (cg_a, cg_b, cp_a, cp_b, m_a, m_b):
        ttnn.deallocate(t)
    print(f"\n=== fused input op, N={N} C={C}, grid {gx}x{gy} ===", flush=True)
    print("  ttnn gate vs torch gate: exact_a=%s exact_b=%s  maxdiff_a=%.3e" % (
        torch.equal(ref_a_t, gate_a), torch.equal(ref_b_t, gate_b),
        (ref_a_t.float() - gate_a.float()).abs().max()), flush=True)

    n_jt = n_it = N // 32
    ct_per_op = C // 32
    gp_col_tiles = 4 * C // 32
    rows = []

    def amort(fn, reps):
        for _ in range(2):
            for _ in range(reps):
                fn()
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3 / reps)
        return sorted(ts)[len(ts) // 2]

    def baseline(inner_swap):
        # mirrors TriangleMultiplication.__call__: in-place gate, g chunks freed at once
        ca, cbb, pa, pb = ttnn.chunk(gp, chunks=4, dim=-1)
        aa = ttnn.multiply_(pa, ca, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        bb = ttnn.multiply_(pb, cbb, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(ca)
        ttnn.deallocate(cbb)
        ap = ttnn.permute(aa, (0, 3, 1, 2), memory_config=L1)
        ttnn.deallocate(aa)
        bp = ttnn.permute(bb, (0, 3, 2, 1) if inner_swap else (0, 3, 1, 2), memory_config=L1)
        ttnn.deallocate(bb)
        ttnn.deallocate(ap)
        ttnn.deallocate(bp)

    moved_all = (N * N * 4 * C * 2 + 2 * N * N * C * 2) / 1e6
    for tag, swap in (("ttnn chain (production)", True),
                      ("ttnn chain, no inner swap", False)):
        ms = amort(lambda s=swap: baseline(s), a.reps)
        rows.append(dict(arm=tag, ms=round(ms, 4), eff_gbs=round(moved_all / ms, 1)))
        print("  [%-26s] %8.4f ms   (%.1f MB payload -> %6.1f GB/s)" % (
            tag, ms, moved_all, moved_all / ms), flush=True)

    for arm in a.arms.split(","):
        exch = arm.startswith("v1b")
        nogate = arm.endswith("-nogate")
        shape = [1, C, N, N] if exch else [1, N, C, N]
        oa = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, dev, L1)
        ob = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, dev, L1)
        try:
            pd = build(dev, gp, oa, ob, gx, gy, exch, n_jt, n_it, ct_per_op,
                       gp_col_tiles, nogate)
            ttnn.generic_op([gp, oa, ob], pd)
            ga, gb = ttnn.to_torch(oa), ttnn.to_torch(ob)
            if exch:
                ex_a = gate_a.permute(0, 3, 1, 2)
                ex_b = gate_b.permute(0, 3, 1, 2)
                tt_a = ref_a_t.permute(0, 3, 1, 2)
            else:
                ex_a = gate_a.permute(0, 1, 3, 2)
                ex_b = gate_b.permute(0, 1, 3, 2)
                tt_a = ref_a_t.permute(0, 1, 3, 2)
            if nogate:
                note = "nogate diagnostic, no correctness bar"
            else:
                note = ("exact_vs_torch a=%s b=%s | exact_vs_ttnn_chain=%s | maxdiff=%.3e"
                        % (torch.equal(ga, ex_a), torch.equal(gb, ex_b),
                           torch.equal(ga, tt_a.contiguous()),
                           (ga.float() - ex_a.float()).abs().max()))
            print(f"  [{arm}] {note}", flush=True)
            for _ in range(2):
                for _ in range(a.reps):
                    ttnn.generic_op([gp, oa, ob], pd)
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(5):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    ttnn.generic_op([gp, oa, ob], pd)
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3 / a.reps)
            ms = sorted(ts)[len(ts) // 2]
            if exch and not nogate:
                def fused_plus():
                    ttnn.generic_op([gp, oa, ob], pd)
                    t = ttnn.transpose(ob, -2, -1, memory_config=L1)
                    ttnn.deallocate(t)
                ms2 = amort(fused_plus, a.reps)
                rows.append(dict(arm=arm + " + inner-swap transpose", ms=round(ms2, 4)))
                print("  [%s + inner-swap transpose] %8.4f ms" % (arm, ms2), flush=True)
            moved = (N * N * 4 * C * 2 + 2 * N * N * C * 2) / 1e6
            rows.append(dict(arm=arm, ms=round(ms, 4), moved_mb=round(moved, 1),
                             eff_gbs=round(moved / ms, 1), note=note))
            print("  [%s] %8.4f ms   %.1f MB   %7.1f GB/s" % (arm, ms, moved, moved / ms),
                  flush=True)
        except Exception as e:
            print(f"  [{arm}] FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)
            rows.append(dict(arm=arm, error=f"{type(e).__name__}: {str(e)[:400]}"))
        ttnn.deallocate(oa)
        ttnn.deallocate(ob)

    if a.out:
        Path(a.out).write_text(json.dumps(dict(n=N, c=C, grid=a.grid, rows=rows), indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
