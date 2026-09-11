#!/usr/bin/env python3
"""P3 phase 2c: the controlled tail-L1 arm.

The first ladder's L1 arms were confounded: on an L1 destination `reblock_permute.eligible`
closes above N=352 and `_transform_chunk` stops decomposing the inner transpose, so the arm
swapped the hand-written channel move for the slow single permute. Here the reblock L1 window is
widened (TT_BIO_REBLOCK_L1_N_MAX) and the decomposition is forced, so the only difference between
the arms is where the tail lives. Also probes whether the triangle matmul takes height-sharded
operands at all, on a core count that divides the shard.
"""
import json, os, statistics as st, sys, time, traceback
from collections import Counter
from pathlib import Path

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RB
from tt_bio.tenstorrent import TriangleMultiplication, get_device

N = 512
OUT = Path(sys.argv[1])
dev = get_device()
print("REBLOCK_L1_N_MAX =", RB.L1_N_MAX, flush=True)
ck = torch.load("/home/ttuser/.boltz/boltz2_conf.ckpt", map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

def build(which):
    pre = "pairformer_module.layers.0." + ("tri_mul_in." if which == "end" else "tri_mul_out.")
    return TriangleMultiplication(which == "end",
                                  {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, ckc)

mods = {w: build(w) for w in ("start", "end")}
c_z = int(mods["start"]._g_in_t.shape[0])
torch.manual_seed(0)
z = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.5, layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)
mask = ttnn.from_torch(torch.ones(1, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                       dtype=ttnn.bfloat16)
res = {"n": N, "c_z": c_z, "reblock_l1_n_max": RB.L1_N_MAX, "arms": [], "shard": {}}
ref = {}

def arm(which, group, tail_l1, share=0.75, label=""):
    tm = mods[which]
    T._TRIMUL_INPROJ_GROUP, T._TRIMUL_TAIL_L1, T._TRIMUL_TAIL_L1_SHARE = group, tail_l1, share
    T.TRIMUL_TAIL_L1_STATS["l1"] = T.TRIMUL_TAIL_L1_STATS["dram"] = 0
    row = {"which": which, "group": group, "tail_l1": tail_l1, "label": label}
    try:
        out = tm(z, mask)
        h = ttnn.to_torch(out); ttnn.deallocate(out)
        row["took_l1"], row["refused"] = T.TRIMUL_TAIL_L1_STATS["l1"], T.TRIMUL_TAIL_L1_STATS["dram"]
        if which not in ref:
            ref[which], row["bit_exact"], row["max_abs"] = h, True, 0.0
        else:
            row["bit_exact"] = bool(torch.equal(h, ref[which]))
            row["max_abs"] = float((h.float() - ref[which].float()).abs().max())
        for _ in range(2):
            ttnn.deallocate(tm(z, mask))
        ttnn.synchronize_device(dev)
        ser = []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = tm(z, mask)
            ttnn.synchronize_device(dev)
            ser.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(r)
        row["ms"], row["series"] = round(st.median(ser), 4), [round(x, 3) for x in ser]
        T.TRIMUL_TAIL_L1_STATS["l1"] = T.TRIMUL_TAIL_L1_STATS["dram"] = 0
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        o = tm(z, mask)
        g = ttnn.graph.end_graph_capture()
        ttnn.deallocate(o); ttnn.synchronize_device(dev)
        dram = sum(int((nd.get("params") or {}).get("size", 0)) for nd in g
                   if nd.get("node_type") == "buffer"
                   and str(((nd.get("params") or {}).get("type", ""))).upper().endswith("DRAM"))
        ops = Counter((nd.get("params") or {}).get("name") for nd in g
                      if nd.get("node_type") == "function_start"
                      and ((nd.get("params") or {}).get("name") or "").startswith("ttnn."))
        row["dram_alloc_MB"] = round(dram / 1e6, 2)
        row["generic_op"] = ops.get("ttnn.generic_op", 0)
        row["permute"] = ops.get("ttnn.permute", 0)
        row["transpose"] = ops.get("ttnn.transpose", 0)
        row["ttnn_ops"] = sum(ops.values())
        row["ok"] = True
    except Exception as e:
        row["ok"], row["error"] = False, str(e)[:300]
        traceback.print_exc()
    print("ARM " + json.dumps(row), flush=True)
    res["arms"].append(row)

for which in ("start", "end"):
    arm(which, 12, False, label="shipped (group 4, tail DRAM)")
    arm(which, 2, False, label="group 2, tail DRAM")
    arm(which, 2, True, 0.75, label="group 2, tail L1, reblock forced")
    arm(which, 1, False, label="group 1, tail DRAM")
    arm(which, 1, True, 0.5, label="group 1, tail L1, reblock forced")
T._TRIMUL_TAIL_L1, T._TRIMUL_INPROJ_GROUP = False, 12

# --- does the triangle matmul take height-sharded operands? 64 cores divides every shard here. ---
for C in (32, 64):
    key, cores = f"C{C}", 64
    rows = C * N
    d = {"cores": cores, "shard_rows": rows // cores}
    try:
        mc = ttnn.create_sharded_memory_config(
            shape=(rows // cores, N),
            core_grid=ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(*T.COMPUTE_GRID_MAIN), True),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)
        a = ttnn.from_torch(torch.randn(1, C, N, N) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=mc)
        b = ttnn.from_torch(torch.randn(1, C, N, N) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=mc)
        d["operands_sharded"] = True
        pc = T._triangle_mul_program_config((N + 31) // 32)
        for tag, omc, prog in (("sharded_out", mc, pc), ("dram_out", ttnn.DRAM_MEMORY_CONFIG, pc),
                               ("dram_out_noprog", ttnn.DRAM_MEMORY_CONFIG, None)):
            try:
                kw = {"program_config": prog} if prog is not None else {}
                o = ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=omc,
                                dtype=ttnn.bfloat16, **kw)
                ttnn.synchronize_device(dev)
                d[tag] = True
                ttnn.deallocate(o)
            except Exception as e:
                d[tag] = False
                d[tag + "_error"] = str(e)[:250]
        ttnn.deallocate(a); ttnn.deallocate(b)
    except Exception as e:
        d["operands_sharded"] = False
        d["error"] = str(e)[:250]
    res["shard"][key] = d
    print("SHARD " + json.dumps({key: d}), flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print("WROTE " + str(OUT), flush=True)
