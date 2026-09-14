#!/usr/bin/env python3
"""Which (program config, memory config, rank) combinations let the triangle product keep its
result in a height-sharded L1 buffer? Validation only, no timing.

`b1_l1out_4d.py` showed the recorded blocker is not the 4-D rank: with no program config the matmul
picks one for itself, derives the output shard from that config's per_core_N (8 tiles at s = 288)
and then rejects it against the tensor's 9-tile physical width. A height-sharded result needs
per_core_N == Nt. This enumerates the configs that can give it that.
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

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tt_bio.device_lease import CardSetLease                                  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                           # noqa: E402

T = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def widest_grid(units, gx_max, gy_max):
    best = None
    for y in range(1, gy_max + 1):
        for x in range(1, gx_max + 1):
            if units % (x * y) == 0 and (best is None or x * y > best[0] * best[1]):
                best = (x, y)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "b1_probe.json")
    ap.add_argument("-c", "--channels", type=int, default=128)
    ap.add_argument("-s", "--seq", type=int, default=288)
    a = ap.parse_args()
    C, S = a.channels, a.seq
    Mt = Nt = Kt = S // T

    lease = CardSetLease().acquire()
    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    res = {"host": platform.node(), "arch": str(dev.arch()), "C": C, "S": S,
           "Mt": Mt, "cases": {}}
    try:
        cc = dev.compute_with_storage_grid_size()
        gx, gy = cc.x, cc.y
        res["grid"] = [gx, gy]
        kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
                else ttnn.types.BlackholeComputeKernelConfig)
        ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                   fp32_dest_acc_en=True, packer_l1_acc=True)
        ta4 = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16),
                              layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        tb4 = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16),
                              layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        ta3, tb3 = ttnn.reshape(ta4, (C, S, S)), ttnn.reshape(tb4, (C, S, S))

        ibw = max(d for d in range(1, min(4, Kt) + 1) if Kt % d == 0)
        res["in0_block_w"] = ibw
        pcfgs = {"none": None}
        # batched reuse: one output block per core, so blocks = C * Mt / per_core_M must fit
        for pcm in sorted({d for d in (1, 3, 9, Mt) if Mt % d == 0}):
            blocks = C * (Mt // pcm)
            g = widest_grid(blocks, gx, gy) if blocks <= gx * gy else None
            if g:
                pcfgs["reuse_pcm%d" % pcm] = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(g[0], g[1]),
                    in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
                    per_core_M=pcm, per_core_N=Nt)
        # 2D multicast with the full width on one core column (what a height shard needs)
        for fb in (False, True):
            pcfgs["mcast2d_fb%d" % fb] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(1, min(gy, Mt)),
                in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
                per_core_M=-(-Mt // min(gy, Mt)), per_core_N=Nt,
                transpose_mcast=False, fused_activation=None, fuse_batch=fb)
        # 1D, batch folded into M
        gt = widest_grid(C * Mt, gx, gy)
        if gt:
            pcfgs["mcast1d_fuse"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gt[0], gt[1]),
                in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
                per_core_M=C * Mt // (gt[0] * gt[1]), per_core_N=Nt,
                fuse_batch=True, fused_activation=None, mcast_in0=False)
            res["mcast1d_grid"] = list(gt)

        mcs = {"dram": DRAM, "l1_int": L1,
               "l1_hs_auto": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                                               ttnn.BufferType.L1)}
        g = widest_grid(C * Mt, gx, gy)
        if g:
            mcs["l1_hs_spec"] = ttnn.create_sharded_memory_config(
                (C * S, S), core_grid=ttnn.CoreGrid(y=g[1], x=g[0]),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR)
            res["hs_spec_grid"] = list(g)

        for rank, (x, y) in (("4d", (ta4, tb4)), ("3d", (ta3, tb3))):
            for pn, pc in pcfgs.items():
                for mn, mc in mcs.items():
                    name = "%s|%s|%s" % (rank, pn, mn)
                    kw = {"program_config": pc} if pc is not None else {}
                    try:
                        o = ttnn.matmul(x, y, compute_kernel_config=ckc, memory_config=mc,
                                        dtype=ttnn.bfloat16, **kw)
                        ttnn.synchronize_device(dev)
                        sp = o.memory_config().shard_spec
                        res["cases"][name] = {"ok": True,
                                              "shard": list(sp.shape) if sp else None,
                                              "layout": str(o.memory_config().memory_layout)}
                        print("OK   %-28s shard=%s" % (name, list(sp.shape) if sp else None),
                              flush=True)
                        ttnn.deallocate(o)
                    except Exception as e:                                    # noqa: BLE001
                        m = [ln for ln in str(e).splitlines() if ln.strip()]
                        res["cases"][name] = {"ok": False, "error": str(e)[:700]}
                        print("FAIL %-28s %s" % (name, m[1] if len(m) > 1 else m[0][:120]),
                              flush=True)
    finally:
        a.out.write_text(json.dumps(res, indent=1))
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
