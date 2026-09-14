#!/usr/bin/env python3
"""The triangle product at the shapes and configs production actually issues, with a result the
matmul writes straight into a height-sharded L1 buffer.

Row B1 in `perf/ttx_deadends/CATALOGUE.md` records the blocker as a 4-D rank problem. It is not:
`b1_probe.py` gets the identical refusal on the squeezed 3-D form in all 16 combinations. The real
constraint is that a height-sharded result needs `per_core_N == Nt`, and a batched matmul then
emits one shard per (batch, M-block) output block, which must fit the 110 L1 banks. The channel
chunk is 32 at every production length above 352 aa, so the shard count does fit -- it just needs
an explicit program config, which the ablation never gave it.

Everything here comes from tt_bio's own helpers, so the incumbent arm is the call the fold makes:
`_triangle_mul_program_config`, `_triangle_mul_memory_config`, `_trimul_chunk_size`.
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
import tt_bio.tenstorrent as tt                                               # noqa: E402
from tt_bio.device_lease import CardSetLease                                  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                           # noqa: E402

T = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
HS = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)


def reuse_candidates(C, Mt, Kt, ncores):
    """(per_core_M, in0_block_w) in preference order: widest M block, then widest K block."""
    out = []
    for pcm in sorted((d for d in range(1, Mt + 1) if Mt % d == 0), reverse=True):
        if C * (Mt // pcm) > ncores:
            continue
        for ibw in sorted((d for d in range(1, Kt + 1) if Kt % d == 0), reverse=True):
            out.append((pcm, ibw))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "b1_prod_ab.json")
    ap.add_argument("--blocks", type=int, default=9)
    ap.add_argument("--lengths", default="128,256,288,384,512,640,768")
    a = ap.parse_args()
    lengths = [int(v) for v in a.lengths.split(",")]

    lease = CardSetLease().acquire()
    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    res = {"host": platform.node(), "arch": str(dev.arch()),
           "loadavg": open("/proc/loadavg").read().split()[:3], "shapes": {}}
    try:
        cc = dev.compute_with_storage_grid_size()
        gx, gy = cc.x, cc.y
        ncores = gx * gy
        res["grid"] = [gx, gy]
        ckc = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)

        arms, keep, picked = {}, [], {}
        for H in lengths:
            Mt = Nt = Kt = (H + 31) // 32
            C = tt._trimul_chunk_size(H, 128, 1)
            prod_mc = tt._triangle_mul_memory_config(H)
            prod_cfg = tt._triangle_mul_program_config(Mt)
            tag = "h%d" % H
            ta = ttnn.from_torch(torch.randn(1, C, H, H, dtype=torch.bfloat16),
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            tb = ttnn.from_torch(torch.randn(1, C, H, H, dtype=torch.bfloat16),
                                 layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            keep += [ta, tb]
            fl = 2 * C * H ** 3

            def mm(pc, mc, ta=ta, tb=tb, fl=fl):
                kw = {"program_config": pc} if pc is not None else {}
                return (lambda: ttnn.matmul(ta, tb, compute_kernel_config=ckc, memory_config=mc,
                                            dtype=ttnn.bfloat16, **kw), fl, 2)

            # the incumbent, exactly as the fold issues it
            arms[tag + "_prod"] = mm(prod_cfg, prod_mc)
            arms[tag + "_prod_AA"] = arms[tag + "_prod"]
            other = L1 if prod_mc.buffer_type == ttnn.BufferType.DRAM else DRAM
            arms[tag + "_prod_flip"] = mm(prod_cfg, other)

            # find the widest reuse config this shape can build, by trying it
            chosen = None
            for pcm, ibw in reuse_candidates(C, Mt, Kt, ncores):
                blocks = C * (Mt // pcm)
                g = None
                for y in range(1, gy + 1):
                    for x in range(1, gx + 1):
                        if x * y >= blocks and (g is None or x * y < g[0] * g[1]):
                            g = (x, y)
                cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(g[0], g[1]),
                    in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
                    per_core_M=pcm, per_core_N=Nt)
                try:
                    o = ttnn.matmul(ta, tb, compute_kernel_config=ckc, memory_config=HS,
                                    dtype=ttnn.bfloat16, program_config=cfg)
                    ttnn.synchronize_device(dev)
                    sp = o.memory_config().shard_spec
                    chosen = (cfg, pcm, ibw, blocks, list(g), list(sp.shape) if sp else None)
                    ttnn.deallocate(o)
                    break
                except Exception as e:                                        # noqa: BLE001
                    picked.setdefault(tag + "_tried", []).append(
                        "pcm=%d ibw=%d blocks=%d: %s" % (pcm, ibw, blocks,
                                                         str(e).splitlines()[2][:110]
                                                         if len(str(e).splitlines()) > 2
                                                         else str(e)[:110]))
            res["shapes"][tag] = {"H": H, "chunk_C": C, "Mt": Mt,
                                  "prod_out": str(prod_mc.buffer_type),
                                  "prod_pcm": prod_cfg.per_core_M,
                                  "prod_pcn": prod_cfg.per_core_N,
                                  "prod_ibw": prod_cfg.in0_block_w,
                                  "out_MB": C * H * H * 2 / 1e6}
            if chosen is None:
                res["shapes"][tag]["reuse"] = None
                print("h%-5d no legal height-sharded reuse config" % H, flush=True)
                continue
            cfg, pcm, ibw, blocks, g, shard = chosen
            res["shapes"][tag]["reuse"] = {"per_core_M": pcm, "in0_block_w": ibw,
                                           "shards": blocks, "grid": g, "shard_shape": shard}
            print("h%-5d C=%-4d Mt=%-3d reuse pcm=%d ibw=%d shards=%d grid=%s shard=%s"
                  % (H, C, Mt, pcm, ibw, blocks, g, shard), flush=True)
            arms[tag + "_reuse_dram"] = mm(cfg, DRAM)
            arms[tag + "_reuse_l1int"] = mm(cfg, L1)
            arms[tag + "_reuse_l1hs"] = mm(cfg, HS)
            arms[tag + "_reuse_l1hs_AA"] = arms[tag + "_reuse_l1hs"]

        best, err = time_arms(arms, list(arms), a.blocks, dev)
        tm = 2 * T ** 3
        res["rows"] = [{"arm": n, "ms": dt * 1e3, "TFLOPs": arms[n][1] / dt / 1e12,
                        "ns_per_tile_mac": dt * 1e9 / (arms[n][1] / tm)}
                       for n, dt in best.items()]
        res["refused"] = dict(err, **picked)
        w = max((len(r["arm"]) for r in res["rows"]), default=10)
        for r in res["rows"]:
            print("%-*s  %9.4f ms  %7.2f TFLOP/s  %7.3f ns/tile-MAC"
                  % (w, r["arm"], r["ms"], r["TFLOPs"], r["ns_per_tile_mac"]), flush=True)

        res["exact"] = {}
        for H in lengths:
            tag = "h%d" % H
            if tag + "_reuse_l1hs" not in best:
                continue
            try:
                p = ttnn.to_torch(arms[tag + "_prod"][0]())
                s = ttnn.to_torch(arms[tag + "_reuse_l1hs"][0]())
                res["exact"][tag] = {
                    "torch_equal": bool(torch.equal(p, s)),
                    "max_abs": float((p.float() - s.float()).abs().max()),
                    "rel": float((p.float() - s.float()).abs().max()
                                 / p.float().abs().max().clamp(min=1e-9))}
                print("%s vs production: torch.equal=%s max_abs=%g rel=%g"
                      % (tag, res["exact"][tag]["torch_equal"], res["exact"][tag]["max_abs"],
                         res["exact"][tag]["rel"]), flush=True)
                del p, s
            except Exception as e:                                            # noqa: BLE001
                res["exact"][tag] = {"error": str(e)[:400]}
                print("%s exactness check failed: %s" % (tag, str(e)[:200]), flush=True)
        for x in keep:
            ttnn.deallocate(x)
    finally:
        a.out.write_text(json.dumps(res, indent=1))
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
