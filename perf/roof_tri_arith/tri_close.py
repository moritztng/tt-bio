#!/usr/bin/env python3
"""The two triangle classes the write mechanism does NOT cover, and a size ladder for the four it does.

`tri_write.py` closed the four projection classes: they are DRAM-write-bound, at 54.6-67.1 % of a
221.80 GB/s write roof. The other two classes in the brief's 68 % -- the TriangleMultiplication
triangle product (12.3 %) and the fused TriangleAttention SDPA (22.8 %) -- are equally fidelity-
insensitive but write only 34.7 and 18.7 GB/s, so the write roof cannot be what binds them.

Two things measured here:

  1. The same result-residency ablation applied to the product and to the SDPA, at a size whose
     output fits L1, against a projection run through the identical ablation as the positive
     control. If the product and the SDPA do not move when their result stops going to DRAM, the
     write mechanism is confirmed to cover exactly 33.0 of the 68 points and no more, and the
     other 35.1 are a second, separate, still-unnamed mechanism.

  2. The projection's write rate across the size ladder 320 / 512 / 768, because a mechanism
     measured at one size is a coincidence until it holds at three.
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
CZ = 128
HD, NH = 32, 4


def build(dev, roof_mb, abl_s):
    arch = dev.arch()
    kcls = (ttnn.types.WormholeComputeKernelConfig if arch == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)

    def kc(fid="HiFi4", acc=True):
        return kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                    fp32_dest_acc_en=acc, packer_l1_acc=True)

    cc = dev.compute_with_storage_grid_size()

    def t(shape, mc=DRAM):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    rows = roof_mb * 1024 * 1024 // (2048 * 2)
    rows = (rows // 32) * 32
    nbytes = rows * 2048 * 2
    d_src = t((rows, 2048))
    l1_src = ttnn.to_memory_config(d_src, L1)
    keep = [d_src, l1_src]

    A = {}
    A["roof_write_l1_to_dram"] = (lambda: ttnn.to_memory_config(l1_src, DRAM), nbytes, 3)
    A["roof_read_dram_to_l1"] = (lambda: ttnn.to_memory_config(d_src, L1), nbytes, 3)

    # --- 2. the size ladder, projection with DRAM result ------------------------------------
    w640 = t((CZ, 5 * CZ))
    keep.append(w640)
    for s in (320, 512, 768):
        z = t((s * s, CZ))
        keep.append(z)
        A["ladder_s%d_proj" % s] = (
            lambda z=z: ttnn.linear(z, w640, compute_kernel_config=kc(), memory_config=DRAM,
                                    dtype=ttnn.bfloat16), s * s * 5 * CZ * 2, 2)

    # --- 1. the residency ablation on all three kinds, one size, L1-interleaved result ------
    # S is chosen so that every arm's result fits aggregate L1, which is the only reason this
    # size differs from the fold's. The projection arm is the positive control: it is already
    # known to move 2.85x, so if it moves here and the other two do not, the null is the
    # measurement's, not the ablation's.
    s = abl_s
    zs = t((s * s, CZ))
    ta = t((1, CZ, s, s)); tb = t((1, CZ, s, s))
    q = t((s, NH, s, HD)); k = t((s, NH, s, HD)); v = t((s, NH, s, HD))
    bias = t((1, NH, s, s))
    keep += [zs, ta, tb, q, k, v, bias]
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=cc, exp_approx_mode=False,
                                q_chunk_size=min(256, s), k_chunk_size=min(256, s))

    for tag, mc in (("dramout", DRAM), ("l1out", L1)):
        A["abl_proj_%s" % tag] = (
            lambda mc=mc: ttnn.linear(zs, w640, compute_kernel_config=kc(), memory_config=mc,
                                      dtype=ttnn.bfloat16), 2 * s * s * CZ * 5 * CZ, 3)
        A["abl_product_%s" % tag] = (
            lambda mc=mc: ttnn.matmul(ta, tb, compute_kernel_config=kc(), memory_config=mc),
            2 * CZ * s * s * s, 3)
        A["abl_sdpa_%s" % tag] = (
            lambda mc=mc: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                program_config=pc, compute_kernel_config=kc()),
            2 * 2 * s * NH * s * s * HD, 2)

    # Does the product's accumulation depth, not its result, set it? in0_block_w controls how
    # much K the core accumulates before it drains DST, and packer_l1_acc controls whether the
    # partial sums round-trip through L1 in fp32.
    A["abl_product_nopl1acc"] = (
        lambda: ttnn.matmul(ta, tb, memory_config=DRAM, compute_kernel_config=kcls(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=False)), 2 * CZ * s * s * s, 3)

    A["roof_write_AA"] = A["roof_write_l1_to_dram"]
    A["abl_proj_dramout_AA"] = A["abl_proj_dramout"]
    A["abl_product_dramout_AA"] = A["abl_product_dramout"]
    A["abl_sdpa_dramout_AA"] = A["abl_sdpa_dramout"]
    return A, keep, nbytes, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "tri_close.json")
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--roof-mb", type=int, default=32)
    ap.add_argument("--abl-s", type=int, default=256)
    a = ap.parse_args()

    lease = CardSetLease().acquire()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    try:
        arms, keep, nbytes, s = build(dev, a.roof_mb, a.abl_s)
        best, err = time_arms(arms, list(arms), a.blocks, dev)
        tm = 2 * 32 ** 3
        rows = []
        for n, dt in best.items():
            unit = arms[n][1]
            rows.append({"arm": n, "ms": dt * 1e3,
                         "GBps" if n.startswith(("roof", "ladder")) else "TFLOPs":
                             unit / dt / (1e9 if n.startswith(("roof", "ladder")) else 1e12),
                         "ns_per_tile_mac": (dt * 1e9 / (unit / tm)
                                             if n.startswith("abl") else None)})
        out = {"host": platform.node(), "arch": str(dev.arch()),
               "grid": [dev.compute_with_storage_grid_size().x,
                        dev.compute_with_storage_grid_size().y],
               "roof_bytes": nbytes, "abl_s": s, "blocks": a.blocks,
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "refused": err, "rows": rows}
        a.out.write_text(json.dumps(out, indent=1))
        w = max(len(r["arm"]) for r in rows)
        for r in rows:
            v = r.get("GBps"); u = "GB/s written"
            if v is None:
                v = r.get("TFLOPs"); u = "TFLOP/s"
            extra = ("  %6.3f ns/tile-MAC" % r["ns_per_tile_mac"]) if r["ns_per_tile_mac"] else ""
            print("%-*s  %9.4f ms  %8.2f %s%s" % (w, r["arm"], r["ms"], v, u, extra), flush=True)
        for x in keep:
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
