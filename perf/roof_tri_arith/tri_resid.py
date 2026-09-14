#!/usr/bin/env python3
"""Where the triangle in-projection's 3.4 ns per tile-MAC actually goes.

`tri_mech.py` established the invariant: six unlike triangle matmul classes all run at
3.40-4.06 ns per 32x32x32 tile-MAC on pc's p150a while the 4096 cube runs at 0.512 ns, and
NEITHER math fidelity (LoFi vs HiFi4, 4x the MAC passes) NOR fp32 destination accumulation
moves any of them by more than 4.1 %. The MAC array is therefore not what they are waiting on.

This narrows it to two candidates that the first harness could not separate:

  MEMORY   the operands and the result live in DRAM, and the compute cluster is waiting on the
           reader or the writer. Test: run the identical program with every operand resident in
           L1, sharded, so the DRAM path is removed entirely. If the rate moves, it was memory.

  ISSUE    the compute kernel's per-tile overhead -- the tile_regs acquire/commit/pack/release
           window, which amortises over out_subblock_h * out_subblock_w * in0_block_w tile-MACs
           and no more. At K = 4 tiles that window is short no matter what N is. Test: sweep the
           program config's subblock and in0_block_w explicitly instead of letting ttnn choose.

Both dimensions are swept in one session against one A/A pair, because the two hypotheses make
opposite predictions and a cross-session comparison could not tell them apart.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tri_mech import time_arms                                                # noqa: E402
from tt_bio.device_lease import CardSetLease                                   # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
T = 32

KT, NT = 4, 20            # the TriangleMultiplication in-projection: 128 -> 640


def build(dev, mt_small, mt_big):
    arch = dev.arch()
    kcls = (ttnn.types.WormholeComputeKernelConfig if arch == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)

    def kc(fid="HiFi4", acc=True):
        return kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                    fp32_dest_acc_en=acc, packer_l1_acc=True)

    cc = dev.compute_with_storage_grid_size()
    gx, gy = cc.x, cc.y
    cores = gx * gy
    assert mt_small % cores == 0, (mt_small, cores)
    per_core_M = mt_small // cores

    def t(shape, mc=DRAM):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    def hs(rows, cols):
        """Height-sharded L1 config: `rows` split evenly over the whole grid."""
        return ttnn.create_sharded_memory_config(
            (rows, cols), core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)

    m_s, m_b = mt_small * T, mt_big * T
    k, n = KT * T, NT * T
    in0_s_dram = t((m_s, k))
    in0_b_dram = t((m_b, k))
    w = t((k, n))
    in0_s_l1 = ttnn.to_memory_config(in0_s_dram, hs(m_s, k))
    keep = [in0_s_dram, in0_b_dram, w, in0_s_l1]

    flop = 2 * m_s * k * n
    flop_b = 2 * m_b * k * n
    A = {}

    def auto(x, out_mc, fid="HiFi4", fl=flop, reps=5):
        return (lambda: ttnn.linear(x, w, compute_kernel_config=kc(fid), memory_config=out_mc,
                                    dtype=ttnn.bfloat16), fl, reps)

    A["dram_auto_big"] = auto(in0_b_dram, DRAM, fl=flop_b, reps=3)
    A["dram_auto"] = auto(in0_s_dram, DRAM)
    A["dram_auto_LoFi"] = auto(in0_s_dram, DRAM, "LoFi")
    A["l1in_dramout"] = auto(in0_s_l1, DRAM)

    # The explicit program-config sweep. out_subblock_h x out_subblock_w tiles must fit DST
    # (8 with fp32 dest accumulation, 16 without); anything that does not is refused by the
    # op and lands in the harness's `refused` map rather than being silently dropped.
    def pcfg(bw, sh, sw):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=cc, in0_block_w=bw, out_subblock_h=sh,
            out_subblock_w=sw, per_core_M=per_core_M, per_core_N=NT, fuse_batch=True,
            fused_activation=None, mcast_in0=False)

    def explicit(x, out_mc, bw, sh, sw, acc=True, fid="HiFi4"):
        pc = pcfg(bw, sh, sw)
        return (lambda: ttnn.linear(x, w, compute_kernel_config=kc(fid, acc),
                                    memory_config=out_mc, dtype=ttnn.bfloat16,
                                    program_config=pc), flop, 5)

    # A sharded output additionally requires out_subblock_w == per_core_N or out_subblock_h == 1,
    # and DST holds 4 tiles with fp32 accumulation, so with per_core_N = 20 the only legal
    # subblocks are 1 x {1,2,4}. That is the whole legal sweep, not a selection from it.
    for bw in (1, 2, 4):
        for sh, sw in ((1, 1), (1, 2), (1, 4)):
            A["cfg_l1in_l1out_bw%d_s%dx%d" % (bw, sh, sw)] = explicit(
                in0_s_l1, hs(m_s, n), bw, sh, sw)
    A["cfg_dramin_l1out_bw4_s1x4"] = explicit(in0_s_dram, hs(m_s, n), 4, 1, 4)
    A["cfg_l1in_dramout_bw4_s1x4"] = explicit(in0_s_l1, DRAM, 4, 1, 4)
    A["cfg_l1in_l1out_bw4_s1x4_LoFi"] = explicit(in0_s_l1, hs(m_s, n), 4, 1, 4, fid="LoFi")
    A["cfg_l1in_l1out_bw4_s1x4_noacc"] = explicit(in0_s_l1, hs(m_s, n), 4, 1, 4, acc=False)
    for bw in (1, 2, 4):
        for sh, sw in ((1, 1), (2, 2), (1, 4), (4, 1), (2, 2)):
            A["cfg_dram_bw%d_s%dx%d" % (bw, sh, sw)] = explicit(in0_s_dram, DRAM, bw, sh, sw)

    A["dram_auto_AA"] = A["dram_auto"]
    A["cfg_l1in_l1out_bw4_s1x4_AA"] = A["cfg_l1in_l1out_bw4_s1x4"]
    return A, keep, (gx, gy), cores, per_core_M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "tri_resid.json")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--mt-small", type=int, default=1040)
    ap.add_argument("--mt-big", type=int, default=8192)
    a = ap.parse_args()

    lease = CardSetLease().acquire()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    try:
        arms, keep, grid, cores, pcm = build(dev, a.mt_small, a.mt_big)
        order = list(arms)
        best, err = time_arms(arms, order, a.blocks, dev)
        order = [n for n in order if n in best]
        tm = 2 * T ** 3
        rows = []
        for n in order:
            dt = best[n]
            fl = arms[n][1]
            rows.append({"arm": n, "ms": dt * 1e3, "TFLOPs": fl / dt / 1e12,
                         "ns_per_tile_mac": dt * 1e9 / (fl / tm)})
        out = {"host": platform.node(), "arch": str(dev.arch()), "grid": list(grid),
               "cores": cores, "per_core_M": pcm, "mt_small": a.mt_small,
               "mt_big": a.mt_big, "KT": KT, "NT": NT, "blocks": a.blocks,
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "refused": err, "rows": rows}
        a.out.write_text(json.dumps(out, indent=1))
        w = max(len(r["arm"]) for r in rows)
        for r in sorted(rows, key=lambda r: r["ns_per_tile_mac"]):
            print("%-*s  %9.4f ms  %7.2f TFLOP/s  %6.3f ns/tile-MAC"
                  % (w, r["arm"], r["ms"], r["TFLOPs"], r["ns_per_tile_mac"]), flush=True)
        for x in keep:
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
