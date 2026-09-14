#!/usr/bin/env python3
"""Can the triangle product keep its result in a height-sharded L1 buffer, and is it faster?

`b1_probe.py` found three walls, none of them the 4-D rank the catalogue recorded:

  1. no program config    -> the matmul derives per_core_N = 8 against a 9-tile width and
                             refuses: "Shard width 256 must match physical width 288".
  2. per_core_N = Nt       -> a batched matmul emits one shard per (batch, M-block) output block,
                             so at C = 128 that is 1152 shards against 110 L1 banks.
  3. fuse_batch = True     -> refused outright, "get_batch_size(b_shape_padded) == 1": the triangle
                             product's second operand is batched too, so the batch cannot be
                             folded into M.

Wall 2 is a ceiling on the CHANNEL CHUNK, not on the op: with per_core_M = Mt and per_core_N = Nt
the shard count is exactly the chunk width, so any chunk at or below the core count fits. This
times that arm against the identical config writing to DRAM, plus the production config, on the
same tensors in one interleaved session.
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
sys.path.insert(0, str(ROOT / "perf" / "roof_tri_arith"))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tri_mech import time_arms                                                # noqa: E402
from tt_bio.device_lease import CardSetLease                                  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                           # noqa: E402

T = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def band_block_w(kt, cap=4):
    return max(d for d in range(1, min(cap, kt) + 1) if kt % d == 0)


def grid_for(blocks, gx, gy):
    """Smallest rectangle inside the grid holding `blocks` cores, widest first."""
    best = None
    for y in range(1, gy + 1):
        for x in range(1, gx + 1):
            if x * y >= blocks and (best is None or x * y < best[0] * best[1]):
                best = (x, y)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "b1_shard_ab.json")
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--shapes", default="64x288,64x512,32x512,110x288")
    a = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]

    lease = CardSetLease().acquire()
    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    res = {"host": platform.node(), "arch": str(dev.arch()), "shapes": a.shapes,
           "loadavg": open("/proc/loadavg").read().split()[:3], "geo": {}}
    try:
        cc = dev.compute_with_storage_grid_size()
        gx, gy = cc.x, cc.y
        res["grid"] = [gx, gy]
        kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
                else ttnn.types.BlackholeComputeKernelConfig)
        ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                   fp32_dest_acc_en=True, packer_l1_acc=True)

        arms, keep = {}, []
        for C, S in shapes:
            tag = "c%ds%d" % (C, S)
            Mt = Nt = Kt = S // T
            ta = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16),
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            tb = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16),
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            keep += [ta, tb]
            fl = 2 * C * S ** 3
            ibw = band_block_w(Kt)
            g = grid_for(C, gx, gy)
            res["geo"][tag] = {"Mt": Mt, "in0_block_w": ibw, "reuse_grid": list(g) if g else None,
                               "out_MB": C * S * S * 2 / 1e6}
            if g is None:
                res["geo"][tag]["skip"] = "C=%d exceeds the %d-core grid" % (C, gx * gy)
                continue
            reuse = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(g[0], g[1]),
                in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
                per_core_M=Mt, per_core_N=Nt)
            hs_auto = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                                        ttnn.BufferType.L1)
            prod_cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
                out_block_h=-(-Mt // gy), out_block_w=-(-Nt // gx),
                per_core_M=-(-Mt // gy), per_core_N=-(-Nt // gx),
                transpose_mcast=False, fused_activation=None, fuse_batch=False)

            def mm(pc, mc, ta=ta, tb=tb):
                kw = {"program_config": pc} if pc is not None else {}
                return (lambda: ttnn.matmul(ta, tb, compute_kernel_config=ckc, memory_config=mc,
                                            dtype=ttnn.bfloat16, **kw), fl, 2)

            arms[tag + "_prodcfg_dram"] = mm(prod_cfg, DRAM)
            arms[tag + "_prodcfg_dram_AA"] = arms[tag + "_prodcfg_dram"]
            arms[tag + "_prodcfg_l1int"] = mm(prod_cfg, L1)
            arms[tag + "_reuse_dram"] = mm(reuse, DRAM)
            arms[tag + "_reuse_dram_AA"] = arms[tag + "_reuse_dram"]
            arms[tag + "_reuse_l1hs"] = mm(reuse, hs_auto)
            arms[tag + "_reuse_l1hs_AA"] = arms[tag + "_reuse_l1hs"]
            arms[tag + "_default_dram"] = mm(None, DRAM)

        best, err = time_arms(arms, list(arms), a.blocks, dev)
        tm = 2 * T ** 3
        res["rows"] = [{"arm": n, "ms": dt * 1e3,
                        "TFLOPs": arms[n][1] / dt / 1e12,
                        "ns_per_tile_mac": dt * 1e9 / (arms[n][1] / tm)}
                       for n, dt in best.items()]
        res["refused"] = err
        w = max((len(r["arm"]) for r in res["rows"]), default=10)
        for r in res["rows"]:
            print("%-*s  %9.4f ms  %7.2f TFLOP/s  %7.3f ns/tile-MAC"
                  % (w, r["arm"], r["ms"], r["TFLOPs"], r["ns_per_tile_mac"]), flush=True)

        # bit-exactness of the sharded result against the DRAM result, same config
        res["exact"] = {}
        for C, S in shapes:
            tag = "c%ds%d" % (C, S)
            if tag + "_reuse_l1hs" not in best or tag + "_reuse_dram" not in best:
                continue
            try:
                d = ttnn.to_torch(arms[tag + "_reuse_dram"][0]())
                s = ttnn.to_torch(arms[tag + "_reuse_l1hs"][0]())
                res["exact"][tag] = {"torch_equal": bool(torch.equal(d, s)),
                                     "max_abs": float((d.float() - s.float()).abs().max())}
                print("%s  torch.equal=%s  max_abs=%g"
                      % (tag, res["exact"][tag]["torch_equal"],
                         res["exact"][tag]["max_abs"]), flush=True)
                del d, s
            except Exception as e:                                            # noqa: BLE001
                res["exact"][tag] = {"error": str(e)[:400]}
        for x in keep:
            ttnn.deallocate(x)
    finally:
        a.out.write_text(json.dumps(res, indent=1))
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
