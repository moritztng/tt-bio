#!/usr/bin/env python3
"""P3 phase 2b: does the tail-in-L1 arm actually delete DRAM bytes, and will the trimul matmul
take a height-sharded operand at all?

Graph capture, so the byte count carries no profiler perturbation. One device open.
"""
import json, os, sys, time, traceback
from collections import Counter
from pathlib import Path

import torch
import ttnn

import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import TriangleMultiplication, get_device

CKPT = "/home/ttuser/.boltz/boltz2_conf.ckpt"
N = 512
OUT = Path(sys.argv[1])

dev = get_device()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
P = "pairformer_module.layers.0."
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
pre = P + "tri_mul_out."
w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
tm = TriangleMultiplication(False, w, ckc)
c_z = int(tm._g_in_t.shape[0])
torch.manual_seed(0)
z = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.5, layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)
mask = ttnn.from_torch(torch.ones(1, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                       dtype=ttnn.bfloat16)

res = {"n": N, "c_z": c_z, "arms": []}

def capture(group, tail_l1, share):
    T._TRIMUL_INPROJ_GROUP = group
    T._TRIMUL_TAIL_L1 = tail_l1
    T._TRIMUL_TAIL_L1_SHARE = share
    ttnn.deallocate(tm(z, mask))            # warm this configuration's programs
    ttnn.synchronize_device(dev)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = tm(z, mask)
    g = ttnn.graph.end_graph_capture()
    ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    dram = l1 = 0
    kinds = Counter()
    ops = Counter()
    for nd in g:
        nt = nd.get("node_type")
        kinds[nt] += 1
        p = nd.get("params") or {}
        if nt == "buffer":
            size = int(p.get("size", 0))
            if str(p.get("type", "")).upper().endswith("DRAM"):
                dram += size
            else:
                l1 += size
        elif nt == "function_start":
            nm = p.get("name") or ""
            if nm.startswith("ttnn."):
                ops[nm] += 1
    row = {"group": group, "tail_l1": tail_l1, "share": share,
           "dram_alloc_MB": round(dram / 1e6, 2), "l1_alloc_MB": round(l1 / 1e6, 2),
           "ttnn_ops": sum(ops.values()), "nodes": len(g),
           "took_l1": T.TRIMUL_TAIL_L1_STATS["l1"], "refused": T.TRIMUL_TAIL_L1_STATS["dram"],
           "node_kinds": dict(kinds), "top_ops": dict(ops.most_common(8))}
    T.TRIMUL_TAIL_L1_STATS["l1"] = T.TRIMUL_TAIL_L1_STATS["dram"] = 0
    print("BYTES " + json.dumps({k: v for k, v in row.items() if k != "node_kinds"}), flush=True)
    print("  kinds " + json.dumps(dict(kinds)), flush=True)
    res["arms"].append(row)

for group, tail, share in ((4, False, 0.5), (2, False, 0.5), (2, True, 0.75),
                           (1, False, 0.5), (1, True, 0.5)):
    capture(group, tail, share)

T._TRIMUL_TAIL_L1 = False
T._TRIMUL_INPROJ_GROUP = 12

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print("WROTE " + str(OUT), flush=True)
