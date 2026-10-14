#!/usr/bin/env python3
"""Trimul einsum plan A/B at 512 tokens: the shipped multicast plan (8x8 = 64 cores on an 11x10 grid at
Mt = Nt = 16) against whole channels per core over all cores (`MatmulMultiCoreReuseProgramConfig`), per
channel count. Output equality is against the shipped plan; ms are device time over `--reps` synced calls.

usage: TT_VISIBLE_DEVICES=N einsum_plan.py --out r.json [--reps 20]"""
import argparse, json, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--channels", default="128,256,384")
a = ap.parse_args()
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

dev = T.get_device()
gx, gy = T.COMPUTE_GRID_MAIN
St = a.seq // 32
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
shipped = T._triangle_mul_program_config(St, T._trimul_ibw_full(), T._trimul_subblock(), False)


def per_channel(ibw, sub):
    return ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw,
                                                  out_subblock_h=sub[0], out_subblock_w=sub[1],
                                                  per_core_M=St, per_core_N=St)


def timed(fn):
    fn(); ttnn.synchronize_device(dev)
    runs = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(a.reps): ttnn.deallocate(fn())
        ttnn.synchronize_device(dev); runs.append((time.perf_counter() - t) / a.reps * 1e3)
    return statistics.median(runs), min(runs)


res = dict(grid=[gx, gy], seq=a.seq, shipped=str(shipped), arms=[])
for C in [int(c) for c in a.channels.split(",")]:
    torch.manual_seed(C)
    ah, bh = torch.randn(1, C, a.seq, a.seq), torch.randn(1, C, a.seq, a.seq)
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        A = ttnn.from_torch(ah, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        B = ttnn.from_torch(bh, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mm = lambda pc, tb=False: ttnn.matmul(A, B, program_config=pc, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG, transpose_b=tb)
        ref = {tb: ttnn.to_torch(mm(shipped, tb)) for tb in (False, True)}
        arms = [("shipped", shipped)] + [(f"perch_ibw{w}_sub{s[0]}x{s[1]}", per_channel(w, s))
                                         for w in (1, 2, 4) for s in ((1, 1), (2, 2), (1, 4), (4, 1), (2, 4))]
        for name, pc in arms:
            for tb in (False, True):
                row = dict(C=C, dtype=str(dt), arm=name, transpose_b=tb)
                try:
                    med, mn = timed(lambda: mm(pc, tb))
                    out = ttnn.to_torch(mm(pc, tb))
                    row.update(ms=med, ms_min=mn, equal=bool(torch.equal(out, ref[tb])),
                               max_abs=float((out.float() - ref[tb].float()).abs().max()),
                               finite=bool(torch.isfinite(out).all()))
                except Exception as e:
                    row.update(err=str(e).splitlines()[0][:200])
                print(json.dumps(row), flush=True); res["arms"].append(row)
        ttnn.deallocate(A); ttnn.deallocate(B)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
