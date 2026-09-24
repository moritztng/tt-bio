#!/usr/bin/env python3
"""Replay the wrong elements a census captured (fold_census.py --save DIR) on one tile, one core.

Each capture holds the failing dot product's own operands (a row of A, a column of B, as the device
held them) and what the fold returned. Row 0 of a 32 x K tile gets them, everything else is zero,
and the product runs through a 1x1-grid multicast plan at every arm below. An arm reproduces the
fault when it misses the float64 product by more than 0.25 x sqrt(sum a^2 b^2) (the census bar).

    TT_VISIBLE_DEVICES=<card> python3 perf/mgx_wh_matmul/replay.py DIR [--out replay.json]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import torch
import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--max", type=int, default=64, help="hits replayed")
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = ttnn.open_device(device_id=0)
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
DT = {"DataType.FLOAT32": (torch.float32, ttnn.float32), "DataType.BFLOAT16": (torch.bfloat16, ttnn.bfloat16)}


def arms(kt):
    ws = [w for w in (1, 2, 4, 8, 16) if kt % w == 0]
    for w in ws:
        yield dict(fid="HiFi4", w=w, l1=True)
    yield dict(fid="HiFi4", w=1, l1=False)
    yield dict(fid="HiFi4", w=ws[-1], l1=False)
    yield dict(fid="HiFi3", w=ws[-1], l1=True)
    yield dict(fid="HiFi2", w=ws[-1], l1=True)


hits = []
for p in sorted(glob.glob(os.path.join(a.dir, "cap_*.pt"))):
    c = torch.load(p)
    key = json.loads(c["key"])
    for h in c["hits"]:
        hits.append((key, h))
rows = []
for key, h in hits[:a.max]:
    site, adt, bdt = key[0], key[12], key[13]
    av, bv = h["a"].double(), h["b"].double()
    K = av.numel()
    Kp = -(-K // 32) * 32
    ref = float(av @ bv)
    scale = float(((av * av) @ (bv * bv)).sqrt())
    ta_, tb_ = DT.get(adt, DT["DataType.BFLOAT16"]), DT.get(bdt, DT["DataType.BFLOAT16"])
    A = torch.zeros(1, 1, 32, Kp, dtype=ta_[0])
    B = torch.zeros(1, 1, 32, Kp, dtype=tb_[0])
    A[0, 0, 0, :K], B[0, 0, 0, :K] = h["a"].to(ta_[0]), h["b"].to(tb_[0])
    ta = ttnn.from_torch(A, dtype=ta_[1], layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(B, dtype=tb_[1], layout=ttnn.TILE_LAYOUT, device=dev)
    r = {"site": site, "M": key[2], "K": K, "N": key[4], "a_dtype": adt, "b_dtype": bdt,
         "fold_w": key[6], "ref": round(ref, 5), "fold_err": round(h["got"] - h["ref"], 5),
         "scale": round(scale, 5), "arms": []}
    for arm in arms(Kp // 32):
        ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, arm["fid"]), math_approx_mode=False,
                   fp32_dest_acc_en=True, packer_l1_acc=arm["l1"])
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(1, 1), in0_block_w=arm["w"], out_subblock_h=1, out_subblock_w=1,
            out_block_h=1, out_block_w=1, per_core_M=1, per_core_N=1, transpose_mcast=False,
            fused_activation=None, fuse_batch=False)
        o = ttnn.matmul(ta, tb, transpose_b=True, compute_kernel_config=ckc, program_config=pc, dtype=ttnn.float32)
        err = float(ttnn.to_torch(o)[0, 0, 0, 0]) - ref
        ttnn.deallocate(o)
        r["arms"].append(dict(arm, err=round(err, 5), fault=abs(err) > 0.25 * scale))
    ttnn.deallocate(ta)
    ttnn.deallocate(tb)
    rows.append(r)
    print(json.dumps(r), flush=True)
summ = {}
for r in rows:
    for x in r["arms"]:
        k = f'{x["fid"]} w{x["w"]} l1{int(x["l1"])}'
        s = summ.setdefault(k, [0, 0])
        s[0] += x["fault"]
        s[1] += 1
print("SUMMARY faults/replayed per arm:", json.dumps(summ), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "summary": summ, "rows": rows}, indent=1) + "\n")
ttnn.close_device(dev)
