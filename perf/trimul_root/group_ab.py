#!/usr/bin/env python3
"""Role-major group width at 512 aa: module wall and bit-exactness against today's group 1."""
import json, statistics as st, sys, time
from pathlib import Path

ROOT = Path("/home/ttuser/.coworker/wt/trimul-bottleneck-rootcause")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "stage_split_298"))

import torch
import ttnn
from pf_layer import build_layer
import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import get_device

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
layer, c_z = build_layer(ckc)
torch.manual_seed(0)
z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)
res = {"n": N, "c_z": c_z, "arms": []}
ref = {}
for which, tm in (("start", layer.triangle_multiplication_start),
                  ("end", layer.triangle_multiplication_end)):
    for g in (1, 2, 4, 8):
        T._TRIMUL_INPROJ_GROUP = g
        out = tm(z, None)
        h = ttnn.to_torch(out)
        ttnn.deallocate(out)
        if g == 1:
            ref[which] = h
            eq, maxabs = True, 0.0
        else:
            eq = bool(torch.equal(h, ref[which]))
            maxabs = float((h.float() - ref[which].float()).abs().max())
        for _ in range(2):
            ttnn.deallocate(tm(z, None))
        ttnn.synchronize_device(dev)
        ser = []
        for _ in range(3):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = tm(z, None)
            ttnn.synchronize_device(dev)
            ser.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(r)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [tm(z, None) for _ in range(4)]
        ttnn.synchronize_device(dev)
        pipe = (time.perf_counter() - t0) * 1e3 / 4
        for o in outs:
            ttnn.deallocate(o)
        row = dict(which=which, group=g, serial_ms=round(st.median(ser), 4),
                   pipe_ms=round(pipe, 4), bit_exact_vs_g1=eq, max_abs=maxabs)
        res["arms"].append(row)
        print(json.dumps(row), flush=True)
T._TRIMUL_INPROJ_GROUP = 1
Path(ROOT / "perf/trimul_root/group_ab_512_qb2c0.json").write_text(json.dumps(res, indent=1))
print("RESULT_JSON " + json.dumps(res))
