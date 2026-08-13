"""Which fill_precondition fails for K2 at the bg_R3 triangle-attention shape?"""
import sys, json
sys.path.insert(0, "/home/ttuser/.coworker/wt/boltzgen-optimize-on-fixture")
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import sdpa_generic as SG

dev = ttnn.open_device(device_id=0)
T.set_device(dev) if hasattr(T, "set_device") else None
B, H, S, D = 576, 4, 576, 32
mk = lambda shp: ttnn.from_torch(torch.zeros(shp, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                 device=dev, dtype=ttnn.bfloat16,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
q = mk((B, H, S, D)); k = mk((B, H, S, D)); v = mk((B, H, S, D))
mask = mk((1, H, S, S))
out = ttnn.allocate_tensor_on_device(ttnn.Shape([B, H, S, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                     dev, ttnn.DRAM_MEMORY_CONFIG)
grid = tuple(T.COMPUTE_GRID_MAIN)
print("COMPUTE_GRID_MAIN", grid, "cores", grid[0] * grid[1])
print("fits", T._tri_att_q_chunks(S, S), "k_chunk", T._sdpa_chunks_shipped(S, S)[1])
ckc = (ttnn.MathFidelity.HiFi2, True, False, False)
for qc in T._tri_att_q_chunks(S, S):
    kc = T._sdpa_chunks_shipped(S, S)[1]
    split = (grid[0] * grid[1] // H, H, 1)
    try:
        p = SG.plan(q, k, v, mask, out, qc, kc, grid, ckc, 1.0, split)
    except Exception as e:
        print(f"q_chunk={qc} k_chunk={kc} PLAN THREW {e}")
        continue
    checks = {
        "nh_per_core==1": p["nh_per_core"] == 1,
        "q_per_core==1": p["q_per_core"] == 1,
        "bcast_batch": p["bcast_batch"],
        "not use_padded_mask": not p["use_padded_mask"],
        "NKH==H": p["NKH"] == H,
        "NVH==H": p["NVH"] == H,
    }
    print(f"q_chunk={qc} k_chunk={kc} split={split} ->", json.dumps(checks),
          {kk: p[kk] for kk in ("q_num_chunks", "k_num_chunks", "Sq_chunk_t", "Sk_chunk_t",
                                "nh_per_core", "q_per_core", "batch_per_core", "Sq", "Sk",
                                "use_padded_mask", "NKH", "NVH", "bcast_batch")})
ttnn.close_device(dev)
