#!/usr/bin/env python3
"""Catalogue row B1: does squeezing the batch-1 axis let the triangle product keep its result in L1?

`roof-tri-close` could not run the right ablation on the TriangleMultiplication product: asking ttnn
for a height-sharded L1 result on a 4-D `(1, 128, 288, 288)` matmul trips
`physical_width == physical_shard_width`. The batch dim is 1, so the question this script answers is
whether the 3-D form `(128, 288, 288)` is accepted where the 4-D form is refused.

Minimal on purpose: reproduce the refusal, try the squeeze, and price the reshape. No model code.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tt_bio.device_lease import CardSetLease                                  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                           # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
T = 32


def widest_grid(tiles: int, gx_max: int, gy_max: int):
    """Widest rectangle inside the grid whose core count divides `tiles`."""
    best = None
    for y in range(1, gy_max + 1):
        for x in range(1, gx_max + 1):
            c = x * y
            if tiles % c == 0 and (best is None or c > best[0] * best[1]):
                best = (x, y)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "b1_l1out_4d.json")
    ap.add_argument("-c", "--channels", type=int, default=128)
    ap.add_argument("-s", "--seq", type=int, default=288)
    a = ap.parse_args()

    C, S = a.channels, a.seq
    lease = CardSetLease().acquire()
    # A lone p300 chip is a CUSTOM topology and needs the 1x1 Blackhole descriptor.
    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    res = {"host": platform.node(), "arch": str(dev.arch()), "C": C, "S": S, "arms": {}}
    try:
        cc = dev.compute_with_storage_grid_size()
        gx, gy = cc.x, cc.y
        res["grid"] = [gx, gy]
        rows = C * S
        tiles = rows // T
        g = widest_grid(tiles, gx, gy)
        res["shard_grid"] = list(g) if g else None
        print("grid %dx%d = %d cores; %d rows = %d row-tiles; widest dividing rect %s"
              % (gx, gy, gx * gy, rows, tiles, g), flush=True)
        if g is None:
            res["arms"]["_fatal"] = "no rectangle divides %d row-tiles" % tiles
            a.out.write_text(json.dumps(res, indent=1))
            return 1
        sgx, sgy = g
        mc_shard = ttnn.create_sharded_memory_config(
            (rows, S), core_grid=ttnn.CoreGrid(y=sgy, x=sgx),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)
        res["shard_shape"] = [rows // (sgx * sgy), S]

        ta_t = torch.randn(1, C, S, S, dtype=torch.bfloat16)
        tb_t = torch.randn(1, C, S, S, dtype=torch.bfloat16)
        ta = ttnn.from_torch(ta_t, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        tb = ttnn.from_torch(tb_t, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)

        kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
                else ttnn.types.BlackholeComputeKernelConfig)
        ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                   fp32_dest_acc_en=True, packer_l1_acc=True)

        def attempt(name, fn):
            try:
                out = fn()
                ttnn.synchronize_device(dev)
                res["arms"][name] = {"ok": True, "shape": [int(d) for d in out.shape],
                                     "memory": str(out.memory_config())[:400]}
                print("OK   %-24s %s" % (name, [int(d) for d in out.shape]), flush=True)
                return out
            except Exception as e:                                            # noqa: BLE001
                first = str(e).strip().splitlines()
                res["arms"][name] = {"ok": False, "error": str(e)[:2000],
                                     "type": type(e).__name__}
                print("FAIL %-24s %s: %s" % (name, type(e).__name__,
                                             first[0] if first else ""), flush=True)
                return None

        # --- 1. baseline: 4-D, DRAM result ---------------------------------------------------
        o = attempt("mm4d_dramout", lambda: ttnn.matmul(ta, tb, compute_kernel_config=ckc,
                                                        memory_config=DRAM))
        if o is not None:
            ttnn.deallocate(o)

        # --- 2. the refusal the catalogue records --------------------------------------------
        o = attempt("mm4d_l1out_shard", lambda: ttnn.matmul(ta, tb, compute_kernel_config=ckc,
                                                            memory_config=mc_shard))
        if o is not None:
            ttnn.deallocate(o)

        # --- 3. the squeeze: is the reshape itself free? -------------------------------------
        addr4 = ta.buffer_address()
        t0 = time.perf_counter()
        ta3 = ttnn.reshape(ta, (C, S, S))
        ttnn.synchronize_device(dev)
        t_reshape = time.perf_counter() - t0
        tb3 = ttnn.reshape(tb, (C, S, S))
        ttnn.synchronize_device(dev)
        addr3 = ta3.buffer_address()
        res["reshape"] = {"ms": t_reshape * 1e3, "addr_4d": addr4, "addr_3d": addr3,
                          "same_buffer": addr4 == addr3,
                          "shape_3d": [int(d) for d in ta3.shape]}
        print("reshape 4d->3d  %.4f ms  buffer %s -> %s  same=%s"
              % (t_reshape * 1e3, hex(addr4), hex(addr3), addr4 == addr3), flush=True)

        # roundtrip bit-exactness: squeeze, unsqueeze, back to torch
        ta4b = ttnn.reshape(ta3, (1, C, S, S))
        ttnn.synchronize_device(dev)
        rt = ttnn.to_torch(ta4b)
        base = ttnn.to_torch(ta)
        res["reshape"]["roundtrip_torch_equal"] = bool(torch.equal(rt, base))
        print("reshape roundtrip torch.equal = %s"
              % res["reshape"]["roundtrip_torch_equal"], flush=True)
        del rt, base

        # --- 4. the question: 3-D, height-sharded L1 result -----------------------------------
        o3 = attempt("mm3d_l1out_shard", lambda: ttnn.matmul(ta3, tb3, compute_kernel_config=ckc,
                                                             memory_config=mc_shard))
        if o3 is not None:
            # can the result be unsquozen back to 4-D while sharded?
            try:
                u = ttnn.reshape(o3, (1, C, S, S))
                ttnn.synchronize_device(dev)
                res["arms"]["mm3d_l1out_shard"]["unsqueeze_ok"] = True
                res["arms"]["mm3d_l1out_shard"]["unsqueeze_memory"] = str(u.memory_config())[:400]
                print("OK   unsqueeze sharded result back to 4-D", flush=True)
            except Exception as e:                                            # noqa: BLE001
                res["arms"]["mm3d_l1out_shard"]["unsqueeze_ok"] = False
                res["arms"]["mm3d_l1out_shard"]["unsqueeze_error"] = str(e)[:2000]
                print("FAIL unsqueeze sharded result: %s" % str(e).splitlines()[0], flush=True)
            ttnn.deallocate(o3)

        # --- 5. control: 3-D, DRAM and interleaved-L1 results ---------------------------------
        for nm, mc in (("mm3d_dramout", DRAM), ("mm3d_l1out_int", L1)):
            o = attempt(nm, lambda mc=mc: ttnn.matmul(ta3, tb3, compute_kernel_config=ckc,
                                                      memory_config=mc))
            if o is not None:
                ttnn.deallocate(o)
    finally:
        a.out.write_text(json.dumps(res, indent=1))
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
