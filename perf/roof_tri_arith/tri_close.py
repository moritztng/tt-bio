#!/usr/bin/env python3
"""The two triangle classes the write mechanism does NOT cover, and a size ladder for the four it does.

`tri_write.py` closed the four projection classes: they are DRAM-write-bound, at 54.6-67.1 % of a
221.80 GB/s write roof on pc's p150a. The other two classes in the brief's 68 % -- the
TriangleMultiplication triangle product (12.3 %) and the fused TriangleAttention SDPA (22.8 %) --
are equally fidelity-insensitive but write only 34.7 and 18.7 GB/s, so the write roof cannot be
what binds them.

Two things measured here:

  1. The same result-residency ablation applied to the product and to the SDPA, against a
     projection run through the identical ablation as the positive control. `tri_resid.py`'s 2.85x
     came from a HEIGHT-SHARDED L1 result under an explicit program config, not from an
     interleaved L1 result, so both L1 forms are carried on every op: interleaved L1 still crosses
     the NOC to 72 (or 130) scattered banks and is not the same ablation. Whichever form an op
     refuses lands in the harness's `refused` map instead of being silently dropped.

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
T = 32
KT, NT = 4, 20            # the TriangleMultiplication in-projection: 128 -> 640


def build(dev, roof_mb, pcm, s_p, s_a):
    arch = dev.arch()
    kcls = (ttnn.types.WormholeComputeKernelConfig if arch == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)

    def kc(fid="HiFi4", acc=True):
        return kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                    fp32_dest_acc_en=acc, packer_l1_acc=True)

    cc = dev.compute_with_storage_grid_size()
    gx, gy = cc.x, cc.y
    cores = gx * gy

    def t(shape, mc=DRAM):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    def hs(rows, cols):
        """Height-sharded L1: `rows` split evenly over the whole grid, each core keeps its own.

        Returns None when `rows` does not divide the grid, so the arm is skipped as unavailable
        rather than the whole session dying at build time."""
        if rows % (cores * T):
            return None
        return ttnn.create_sharded_memory_config(
            (rows, cols), core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)

    def put(name, mc, mk):
        if mc is None:
            skipped[name] = "height shard illegal on %d cores" % cores
        else:
            A[name] = mk(mc)

    keep = []
    skipped = {}

    # --- 0. the two DRAM roofs, this session, this part ---------------------------------------
    rows = roof_mb * 1024 * 1024 // (2048 * 2)
    rows = (rows // 32) * 32
    nbytes = rows * 2048 * 2
    d_src = t((rows, 2048))
    l1_src = ttnn.to_memory_config(d_src, L1)
    keep += [d_src, l1_src]
    A = {}
    A["roof_write_l1_to_dram"] = (lambda: ttnn.to_memory_config(l1_src, DRAM), nbytes, 3)
    A["roof_read_dram_to_l1"] = (lambda: ttnn.to_memory_config(d_src, L1), nbytes, 3)

    # --- 1. the positive control: tri_resid.py's arms, this part -------------------------------
    # m is cores * per_core_M * 32 so the sharded form is legal on whatever grid this part has.
    m = cores * pcm * T
    k, n = KT * T, NT * T
    ctl_dram = t((m, k))
    ctl_l1 = ttnn.to_memory_config(ctl_dram, hs(m, k))
    w = t((k, n))
    w640 = w
    keep += [ctl_dram, ctl_l1, w]
    fl_ctl = 2 * m * k * n
    pcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=cc, in0_block_w=KT, out_subblock_h=1, out_subblock_w=4,
        per_core_M=pcm, per_core_N=NT, fuse_batch=True, fused_activation=None, mcast_in0=False)

    def lin(x, out_mc, pg=None):
        kw = {"program_config": pg} if pg is not None else {}
        return (lambda: ttnn.linear(x, w, compute_kernel_config=kc(), memory_config=out_mc,
                                    dtype=ttnn.bfloat16, **kw), fl_ctl, 3)

    A["ctl_proj_dramout"] = lin(ctl_dram, DRAM)
    A["ctl_proj_l1out_int"] = lin(ctl_dram, L1)
    put("ctl_proj_l1out_shard", hs(m, n), lambda mc: lin(ctl_l1, mc, pcfg))
    A["ctl_proj_l1in_dramout"] = lin(ctl_l1, DRAM, pcfg)

    # --- 2. the two unexplained classes, through the identical ablation ------------------------
    ta = t((1, CZ, s_p, s_p)); tb = t((1, CZ, s_p, s_p))
    q = t((s_a, NH, s_a, HD)); k_ = t((s_a, NH, s_a, HD)); v = t((s_a, NH, s_a, HD))
    bias = t((1, NH, s_a, s_a))
    keep += [ta, tb, q, k_, v, bias]
    fl_prod = 2 * CZ * s_p ** 3
    fl_sdpa = 2 * 2 * s_a * NH * s_a * s_a * HD
    chunk = max(c for c in range(T, min(256, s_a) + 1, T) if s_a % c == 0)
    pc_sdpa = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=cc, exp_approx_mode=False,
                                     q_chunk_size=chunk, k_chunk_size=chunk)

    def prod(mc, grid=None):
        kw = {"core_grid": grid} if grid is not None else {}
        return (lambda: ttnn.matmul(ta, tb, compute_kernel_config=kc(), memory_config=mc, **kw),
                fl_prod, 2)

    def sdpa(mc, pg=None):
        pg = pg or pc_sdpa
        return (lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k_, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5, memory_config=mc,
            program_config=pg, compute_kernel_config=kc()), fl_sdpa, 2)

    def spc(ch, g=None):
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(g or cc),
                                      exp_approx_mode=False, q_chunk_size=ch, k_chunk_size=ch)

    A["abl_product_dramout"] = prod(DRAM)
    A["abl_product_l1out_int"] = prod(L1)
    put("abl_product_l1out_shard", hs(CZ * s_p, s_p), prod)
    A["abl_sdpa_dramout"] = sdpa(DRAM)
    A["abl_sdpa_l1out_int"] = sdpa(L1)
    put("abl_sdpa_l1out_shard", hs(s_a * NH * s_a, HD), sdpa)

    # Does the product's accumulation path, not its result, set it? packer_l1_acc controls
    # whether the partial sums round-trip through L1 in fp32.
    # --- 2b. the candidates residency does not cover -------------------------------------------
    # A per-core work-quantum floor and CB backpressure inside the fused kernel make opposite
    # predictions about the core-count ladder and the chunk ladder.
    for ch in [c for c in (32, 64, 96, 128, 256) if s_a % c == 0]:
        A["abl_sdpa_chunk%d" % ch] = sdpa(DRAM, spc(ch))
    for div in (2, 4):
        gyh = max(1, gy // div)
        A["abl_sdpa_cores%d" % (gx * gyh)] = sdpa(
            DRAM, spc(chunk, ttnn.CoreCoord(gx, gyh)))
        A["abl_product_cores%d" % (gx * gyh)] = prod(DRAM, ttnn.CoreGrid(y=gyh, x=gx))
    A["abl_product_lofi"] = (
        lambda: ttnn.matmul(ta, tb, memory_config=DRAM, compute_kernel_config=kc("LoFi")),
        fl_prod, 2)
    A["abl_sdpa_lofi"] = (
        lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k_, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5, memory_config=DRAM,
            program_config=pc_sdpa, compute_kernel_config=kc("LoFi")), fl_sdpa, 2)
    A["abl_sdpa_approx"] = (
        lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k_, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5, memory_config=DRAM,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=cc, exp_approx_mode=True,
                q_chunk_size=chunk, k_chunk_size=chunk),
            compute_kernel_config=kc()), fl_sdpa, 2)
    A["abl_sdpa_nomask"] = (
        lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k_, v, is_causal=False, scale=HD ** -0.5, memory_config=DRAM,
            program_config=pc_sdpa, compute_kernel_config=kc()), fl_sdpa, 2)

    A["abl_product_nopl1acc"] = (
        lambda: ttnn.matmul(ta, tb, memory_config=DRAM, compute_kernel_config=kcls(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=False)), fl_prod, 3)

    # --- 3. the size ladder, projection with DRAM result ---------------------------------------
    for s in (320, 512, 768):
        z = t((s * s, CZ))
        keep.append(z)
        A["ladder_s%d_proj" % s] = (
            lambda z=z: ttnn.linear(z, w640, compute_kernel_config=kc(), memory_config=DRAM,
                                    dtype=ttnn.bfloat16), s * s * 5 * CZ * 2, 2)

    A["roof_write_AA"] = A["roof_write_l1_to_dram"]
    A["ctl_proj_dramout_AA"] = A["ctl_proj_dramout"]
    if "ctl_proj_l1out_shard" in A:
        A["ctl_proj_l1out_shard_AA"] = A["ctl_proj_l1out_shard"]
    A["abl_product_dramout_AA"] = A["abl_product_dramout"]
    A["abl_sdpa_dramout_AA"] = A["abl_sdpa_dramout"]
    return A, keep, nbytes, (cores, m, s_p, s_a, chunk, skipped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "tri_close.json")
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--roof-mb", type=int, default=4)
    ap.add_argument("--pcm", type=int, default=4)
    ap.add_argument("--s-prod", type=int, default=288)
    ap.add_argument("--s-sdpa", type=int, default=288)
    a = ap.parse_args()

    lease = CardSetLease().acquire()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    try:
        arms, keep, nbytes, geo = build(dev, a.roof_mb, a.pcm, a.s_prod, a.s_sdpa)
        best, err = time_arms(arms, list(arms), a.blocks, dev)
        tm = 2 * T ** 3
        rows = []
        for n, dt in best.items():
            unit = arms[n][1]
            byteish = n.startswith(("roof", "ladder"))
            rows.append({"arm": n, "ms": dt * 1e3,
                         "GBps" if byteish else "TFLOPs":
                             unit / dt / (1e9 if byteish else 1e12),
                         "ns_per_tile_mac": None if byteish else dt * 1e9 / (unit / tm)})
        cores, m, s_p, s_a, chunk, skipped = geo
        err = dict(err, **skipped)
        out = {"host": platform.node(), "arch": str(dev.arch()),
               "grid": [dev.compute_with_storage_grid_size().x,
                        dev.compute_with_storage_grid_size().y],
               "cores": cores, "roof_bytes": nbytes, "ctl_m_rows": m, "per_core_M": a.pcm,
               "s_prod": s_p, "s_sdpa": s_a, "sdpa_chunk": chunk, "blocks": a.blocks,
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "refused": err, "rows": rows}
        a.out.write_text(json.dumps(out, indent=1))
        w = max(len(r["arm"]) for r in rows)
        for r in rows:
            v = r.get("GBps"); u = "GB/s"
            if v is None:
                v = r.get("TFLOPs"); u = "TFLOP/s"
            extra = ("  %7.3f ns/tile-MAC" % r["ns_per_tile_mac"]) if r["ns_per_tile_mac"] else ""
            print("%-*s  %9.4f ms  %8.2f %s%s" % (w, r["arm"], r["ms"], v, u, extra), flush=True)
        for x in keep:
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
