#!/usr/bin/env python3
"""P3 phase 2e: what exactly refuses when the tail asks for L1 at the shipped group width.

The share sweep contaminated itself -- one arm threw, the channel loop's retry halved this
shape's fused in-projection budget, and every later arm silently ran at the narrower group. Here
the refusal is caught and printed instead of retried, and the budget cap is cleared between arms.
"""
import json, statistics as st, sys, time
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
pre = "pairformer_module.layers.0.tri_mul_out."
tm = TriangleMultiplication(False, {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, ckc)
c_z = int(tm._g_in_t.shape[0])
torch.manual_seed(0)
z = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.5, layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)
mask = ttnn.from_torch(torch.ones(1, N, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
res = {"n": N, "c_z": c_z, "arms": []}
ref = None

def reset():
    T._TRIMUL_INPROJ_FUSED_CAP.clear()
    T._TRIMUL_CHUNK_CLASH.clear() if hasattr(T, "_TRIMUL_CHUNK_CLASH") else None

def arm(label, tail_l1, share, catch):
    global ref
    reset()
    T._TRIMUL_TAIL_L1, T._TRIMUL_TAIL_L1_SHARE = tail_l1, share
    row = {"label": label, "tail_l1": tail_l1, "share": share}
    real = T._dram_oom
    if catch:                      # make the channel loop re-raise instead of narrowing
        T._dram_oom = lambda e: (row.setdefault("refusal", str(e)[:400]), False)[1]
    try:
        out = tm(z, mask)
        h = ttnn.to_torch(out); ttnn.deallocate(out)
        if ref is None:
            ref, row["bit_exact"] = h, True
        else:
            row["bit_exact"] = bool(torch.equal(h, ref))
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
        o = tm(z, mask); g = ttnn.graph.end_graph_capture()
        ttnn.deallocate(o); ttnn.synchronize_device(dev)
        row["dram_alloc_MB"] = round(sum(
            int((nd.get("params") or {}).get("size", 0)) for nd in g
            if nd.get("node_type") == "buffer"
            and str(((nd.get("params") or {}).get("type", ""))).upper().endswith("DRAM")) / 1e6, 2)
        row["ok"] = True
    except Exception as e:
        row["ok"], row["threw"] = False, str(e)[:400]
    finally:
        T._dram_oom = real
    print("ARM " + json.dumps(row), flush=True)
    res["arms"].append(row)

arm("shipped (group 4, tail DRAM)", False, 0.5, False)
arm("shipped group, tail operands L1 (share 0.95), refusal caught", True, 0.95, True)
arm("shipped repeat, A/A floor", False, 0.5, False)
T._TRIMUL_TAIL_L1 = False
reset()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print("WROTE " + str(OUT), flush=True)
