#!/usr/bin/env python3
"""P3 phase 1+2: the p300c's real L1 roof, and a memory-config ladder on the trimul chunk tail.

One device open. Reports the capacity numbers the bet lives on, then runs the real Boltz-2
trunk triangle multiplication at 512 aa across (in-projection group) x (tail in DRAM / tail in
L1), recording whether the arm runs at all, what it costs, and whether it is bit-exact against
the shipped default.
"""
import json, os, statistics as st, sys, time, traceback
from pathlib import Path

import torch
import ttnn

import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import TriangleMultiplication, get_device

CKPT = "/home/ttuser/.boltz/boltz2_conf.ckpt"
N = int(os.environ.get("B2X_N", "512"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/b2xp/tail_ladder.json")

dev = get_device()
gx, gy = T.COMPUTE_GRID_MAIN
cap = {
    "grid": [gx, gy],
    "cores": gx * gy,
    "get_max_worker_l1_unreserved_size": int(ttnn.get_max_worker_l1_unreserved_size()),
    "l1_bank_bytes": int(T._l1_bank_bytes()),
}
cap["aggregate_l1_MB"] = round(cap["l1_bank_bytes"] * cap["cores"] / 1e6, 2)
cap["aggregate_unreserved_MB"] = round(
    cap["get_max_worker_l1_unreserved_size"] * cap["cores"] / 1e6, 2)
print("CAPACITY " + json.dumps(cap), flush=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
P = "pairformer_module.layers.0."
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

def build(which):
    pre = P + ("tri_mul_in." if which == "end" else "tri_mul_out.")
    w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    assert w, pre
    return TriangleMultiplication(which == "end", w, ckc)

mods = {w: build(w) for w in ("start", "end")}
c_z = int(mods["start"]._g_in_t.shape[0])
print(f"built trimul start/end c_z={c_z} hidden={mods['start']._hidden} N={N}", flush=True)

torch.manual_seed(0)
z = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.5, layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)
# 94.3 % of trunk trimul calls carry a mask (state/bioir-transfer-plan-p2.md).
mask = ttnn.from_torch(torch.ones(1, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                       dtype=ttnn.bfloat16)

res = {"n": N, "c_z": c_z, "capacity": cap, "arms": []}
ref = {}

def run_arm(which, group, tail_l1, share):
    tm = mods[which]
    T._TRIMUL_INPROJ_GROUP = group
    T._TRIMUL_TAIL_L1 = tail_l1
    T._TRIMUL_TAIL_L1_SHARE = share
    T.TRIMUL_TAIL_L1_STATS["l1"] = T.TRIMUL_TAIL_L1_STATS["dram"] = 0
    row = {"which": which, "group": group, "tail_l1": tail_l1, "share": share}
    try:
        out = tm(z, mask)
        h = ttnn.to_torch(out)
        ttnn.deallocate(out)
        row["tail_took_l1"] = T.TRIMUL_TAIL_L1_STATS["l1"]
        row["tail_refused"] = T.TRIMUL_TAIL_L1_STATS["dram"]
        key = which
        if key not in ref:
            ref[key] = h
            row["bit_exact_vs_shipped"] = True
            row["max_abs"] = 0.0
        else:
            row["bit_exact_vs_shipped"] = bool(torch.equal(h, ref[key]))
            row["max_abs"] = float((h.float() - ref[key].float()).abs().max())
        for _ in range(2):
            ttnn.deallocate(tm(z, mask))
        ttnn.synchronize_device(dev)
        ser = []
        for _ in range(3):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = tm(z, mask)
            ttnn.synchronize_device(dev)
            ser.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(r)
        row["ms"] = round(st.median(ser), 4)
        row["series"] = [round(x, 3) for x in ser]
        row["ok"] = True
    except Exception as e:
        row["ok"] = False
        row["error"] = str(e)[:400]
        row["etype"] = type(e).__name__
        traceback.print_exc()
    print("ARM " + json.dumps(row), flush=True)
    res["arms"].append(row)
    return row

# Shipped default first, so every later arm has a bit-exactness reference.
for which in ("start", "end"):
    run_arm(which, T._TRIMUL_INPROJ_GROUP, False, 0.5)
for which in ("start", "end"):
    for group in (4, 2, 1):
        for share in (0.5, 0.75):
            run_arm(which, group, True, share)
    # group ladder with the tail left in DRAM, to separate the group effect from the tail effect
    for group in (2, 1):
        run_arm(which, group, False, 0.5)

T._TRIMUL_INPROJ_GROUP = 12
T._TRIMUL_TAIL_L1 = False
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print("WROTE " + str(OUT), flush=True)
