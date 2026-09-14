#!/usr/bin/env python3
"""The roof the triangle projections are actually against: the DRAM WRITE roof.

`tri_mech.py` killed the arithmetic explanation (LoFi vs HiFi4 moves these shapes 1.002-1.041x
where it moves the dense cube 1.203x, and fp32 dest accumulation moves them 0.7-1.8 % where it
moves the cube 1.488x). `tri_resid.py` then found the cause by ablation: the SAME matmul with the
SAME program config runs at 3.358 ns per tile-MAC with its result in DRAM and 1.177 ns with its
result in L1, while moving the INPUT to L1 is worth 4 %. It is the result write.

The campaign has been pricing these classes against the dense cube. That denominator is wrong for
a write-bound op. This measures the three DRAM roofs in one session -- write-only, read-only, and
the campaign's 2-read-1-write eltwise roof -- and prices each triangle class's result write against
the write roof it is actually against.

Three roofs, because they are not the same number and only one of them binds here.
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
S = 512
CZ = 128


def build(dev, roof_mb):
    arch = dev.arch()
    kcls = (ttnn.types.WormholeComputeKernelConfig if arch == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)

    def kc(fid="HiFi4"):
        return kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                    fp32_dest_acc_en=True, packer_l1_acc=True)

    def t(shape, mc=DRAM):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    rows = roof_mb * 1024 * 1024 // (2048 * 2)
    rows = (rows // 32) * 32
    nbytes = rows * 2048 * 2
    d_src = t((rows, 2048))
    l1_src = ttnn.to_memory_config(d_src, L1)
    big = t((8192, 8192)); bigb = t((8192, 8192))
    zflat = t((S * S, CZ))
    keep = [d_src, l1_src, big, bigb, zflat]

    A = {}
    # --- the three roofs, same session, same part -----------------------------------------
    # write-only: source in L1, destination DRAM. DRAM sees only the write.
    A["roof_write_l1_to_dram"] = (lambda: ttnn.to_memory_config(l1_src, DRAM), nbytes, 3)
    # read-only: source in DRAM, destination L1. DRAM sees only the read.
    A["roof_read_dram_to_l1"] = (lambda: ttnn.to_memory_config(d_src, L1), nbytes, 3)
    # 1 read + 1 write.
    A["roof_copy_dram_to_dram"] = (lambda: ttnn.clone(d_src, memory_config=DRAM), 2 * nbytes, 3)
    # the campaign's own roof: 2 reads + 1 write.
    A["roof_add8192"] = (lambda: ttnn.add(big, bigb, memory_config=DRAM),
                         3 * 8192 * 8192 * 2, 3)

    # --- the fold's four write-dominated triangle classes, at 512 aa --------------------------
    ws, byt = {}, {}
    for tag, n in (("trimul_in", 5 * CZ), ("triatt_in", 3 * 4 * 32 + CZ + 32),
                   ("pair_out", CZ), ("n256", 256), ("n384", 384)):
        w = t((CZ, n))
        ws[tag] = w
        keep.append(w)
        byt[tag] = S * S * n * 2

        def mk(w=w, n=n):
            return lambda: ttnn.linear(zflat, w, compute_kernel_config=kc(),
                                       memory_config=DRAM, dtype=ttnn.bfloat16)
        A["proj_%s" % tag] = (mk(), byt[tag], 2)

    A["roof_write_AA"] = A["roof_write_l1_to_dram"]
    A["proj_trimul_in_AA"] = A["proj_trimul_in"]
    return A, keep, byt, nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "tri_write.json")
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--roof-mb", type=int, default=256)
    a = ap.parse_args()

    lease = CardSetLease().acquire()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    try:
        arms, keep, byt, nbytes = build(dev, a.roof_mb)
        best, err = time_arms(arms, list(arms), a.blocks, dev)
        rows = []
        for n, dt in best.items():
            rows.append({"arm": n, "ms": dt * 1e3, "GBps": arms[n][1] / dt / 1e9})
        out = {"host": platform.node(), "arch": str(dev.arch()),
               "grid": [dev.compute_with_storage_grid_size().x,
                        dev.compute_with_storage_grid_size().y],
               "roof_bytes": nbytes, "write_bytes_512aa": byt, "blocks": a.blocks,
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "refused": err, "rows": rows}
        a.out.write_text(json.dumps(out, indent=1))
        w = max(len(r["arm"]) for r in rows)
        for r in rows:
            print("%-*s  %9.4f ms  %8.2f GB/s" % (w, r["arm"], r["ms"], r["GBps"]), flush=True)
        for x in keep:
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
