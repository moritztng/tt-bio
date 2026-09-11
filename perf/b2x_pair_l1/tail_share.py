#!/usr/bin/env python3
"""P3 phase 2d: the tail at the SHIPPED in-projection group, operands in L1, product in DRAM.

The controlled ladder showed residency is worth 1.06-1.07x at a fixed group but that narrowing
the group to make all three tail tensors fit costs 2.3 ms to buy 0.74 ms back. At the shipped
group only the two operands fit (2 x 67.11 MB against 160.79 MB of bank), so this sweeps the
bank share that admits them and asks whether the operand half alone pays.
"""
import json, statistics as st, sys, time, traceback
from collections import Counter
from pathlib import Path
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import TriangleMultiplication, get_device

N = 512
OUT = Path(sys.argv[1])
dev = get_device()
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
res = {"n": N, "c_z": c_z, "arms": []}
ref = {}

def arm(which, group, tail_l1, share, label):
    tm = mods[which]
    T._TRIMUL_INPROJ_GROUP, T._TRIMUL_TAIL_L1, T._TRIMUL_TAIL_L1_SHARE = group, tail_l1, share
    row = {"which": which, "group": group, "tail_l1": tail_l1, "share": share, "label": label}
    try:
        out = tm(z, mask)
        h = ttnn.to_torch(out); ttnn.deallocate(out)
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
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        o = tm(z, mask)
        g = ttnn.graph.end_graph_capture()
        ttnn.deallocate(o); ttnn.synchronize_device(dev)
        row["dram_alloc_MB"] = round(sum(
            int((nd.get("params") or {}).get("size", 0)) for nd in g
            if nd.get("node_type") == "buffer"
            and str(((nd.get("params") or {}).get("type", ""))).upper().endswith("DRAM")) / 1e6, 2)
        row["ok"] = True
    except Exception as e:
        row["ok"], row["error"] = False, str(e)[:250]
    print("ARM " + json.dumps(row), flush=True)
    res["arms"].append(row)

for which in ("start", "end"):
    arm(which, 12, False, 0.5, "shipped")
    for share in (0.5, 0.85, 0.95, 1.0):
        arm(which, 12, True, share, f"shipped group, tail L1 share {share}")
    arm(which, 12, False, 0.5, "shipped, repeat (A/A floor)")
T._TRIMUL_TAIL_L1, T._TRIMUL_INPROJ_GROUP = False, 12
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print("WROTE " + str(OUT), flush=True)
