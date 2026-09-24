#!/usr/bin/env python3
"""Find the add at which a captured wrong element goes wrong (replay.py's captures).

For each hit, the dot product is truncated after t K tiles (later terms zeroed) for t = 1..Kt and run
on one tile at in0_block_w 1, HiFi4, packer L1 acc on, fp32 out. The first t whose result misses the
float64 prefix sum is the failing add; the device's value before it, the tile partial it adds and
the fp32 bit patterns of both are printed.

    TT_VISIBLE_DEVICES=<card> python3 perf/mgx_wh_matmul/locate.py DIR [--w 1] [--l1 1]
"""
import argparse
import glob
import json
import os
import struct

import torch
import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--w", type=int, default=1)
ap.add_argument("--l1", type=int, default=1)
ap.add_argument("--fid", default="HiFi4")
ap.add_argument("--max", type=int, default=8)
a = ap.parse_args()
dev = ttnn.open_device(device_id=0)
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, a.fid), math_approx_mode=False, fp32_dest_acc_en=True,
           packer_l1_acc=bool(a.l1))


def bits(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return f"{u >> 31}|{(u >> 23) & 255:08b}|{(u >> 16) & 127:07b} {u & 65535:016b}"


def run(av, bv, dt):
    K = av.numel()
    A = torch.zeros(1, 1, 32, K, dtype=dt)
    B = torch.zeros(1, 1, 32, K, dtype=dt)
    A[0, 0, 0], B[0, 0, 0] = av.to(dt), bv.to(dt)
    ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(1, 1), in0_block_w=a.w, out_subblock_h=1, out_subblock_w=1,
        out_block_h=1, out_block_w=1, per_core_M=1, per_core_N=1, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)
    o = ttnn.matmul(ta, tb, transpose_b=True, compute_kernel_config=ckc, program_config=pc, dtype=ttnn.float32)
    v = float(ttnn.to_torch(o)[0, 0, 0, 0])
    for t in (o, ta, tb):
        ttnn.deallocate(t)
    return v


n = 0
for p in sorted(glob.glob(os.path.join(a.dir, "cap_*.pt"))):
    c = torch.load(p)
    key = json.loads(c["key"])
    dt = torch.float32 if key[12] == "DataType.FLOAT32" else torch.bfloat16
    for h in c["hits"]:
        if n >= a.max:
            break
        n += 1
        av, bv = h["a"].double(), h["b"].double()
        kt = av.numel() // 32
        scale = float(((av * av) @ (bv * bv)).sqrt())
        prev, prev_dev = 0.0, 0.0
        for t in range(a.w, kt + 1, a.w):
            m = torch.zeros_like(av)
            m[:32 * t] = 1
            got = run(av * m, bv, dt)
            ref = float((av * m) @ bv)
            if abs(got - ref) > 0.25 * scale:
                part = ref - prev
                print(json.dumps({"hit": f"{os.path.basename(p)}#{n}", "site": key[0], "fail_after_tiles": t,
                                  "device_before": prev_dev, "exact_before": prev, "addend_exact": part,
                                  "device_after": got, "exact_after": ref, "err": got - ref, "scale": scale}))
                print("   before ", bits(prev_dev))
                print("   addend ", bits(part))
                print("   after  ", bits(got), " exact", bits(ref))
                break
            prev, prev_dev = ref, got
        else:
            print(json.dumps({"hit": f"{os.path.basename(p)}#{n}", "no_fault_at": a.w}))
ttnn.close_device(dev)
