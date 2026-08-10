"""C1 follow-up: is the 2D untilize collapse fixed by reshaping to rank 3 FIRST?

Same 182 MB, same final layout, one extra tile-grid regrouping in front. Probe only.
"""
import sys, time, json, statistics as st
import torch, ttnn
sys.path.insert(0, "/home/ttuser/.coworker/wt/protenix-trunk--p2-msa-structural")
from tt_bio.tenstorrent import get_device
DRAM = ttnn.DRAM_MEMORY_CONFIG
dev = get_device()
I=J=298; C=D=32
def timed(fn, warm=2, pipe=3, reps=5):
    for _ in range(warm): fn()
    ttnn.synchronize_device(dev)
    o=[]
    for _ in range(reps):
        ttnn.synchronize_device(dev); t0=time.perf_counter()
        for _ in range(pipe): fn()
        ttnn.synchronize_device(dev); o.append((time.perf_counter()-t0)/pipe)
    return round(st.median(o)*1e6,1)
r={}
z = ttnn.from_torch(torch.randn(I*C, D*J)*0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
r["A_to_layout_RM_rank2"] = timed(lambda: ttnn.deallocate(ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)))
r["B_reshape_tile_to_rank3"] = timed(lambda: ttnn.deallocate(ttnn.reshape(z, (I, C, D*J))))
z3 = ttnn.reshape(z, (I, C, D*J))
r["C_to_layout_RM_rank3"] = timed(lambda: ttnn.deallocate(ttnn.to_layout(z3, ttnn.ROW_MAJOR_LAYOUT)))
zr = ttnn.to_layout(z3, ttnn.ROW_MAJOR_LAYOUT)
r["D_reshape_RM_to_I_CD_J"] = timed(lambda: ttnn.deallocate(ttnn.reshape(zr, (I, C*D, J))))
zs = ttnn.reshape(zr, (I, C*D, J))
r["E_to_layout_TILE"] = timed(lambda: ttnn.deallocate(ttnn.to_layout(zs, ttnn.TILE_LAYOUT)))
zt = ttnn.to_layout(zs, ttnn.TILE_LAYOUT)
r["F_permute_021"] = timed(lambda: ttnn.deallocate(ttnn.permute(zt, (0,2,1))))
zp = ttnn.permute(zt, (0,2,1))
r["final_shape"] = list(zp.shape)
r["new_chain_us"] = round(r["B_reshape_tile_to_rank3"]+r["C_to_layout_RM_rank3"]+r["D_reshape_RM_to_I_CD_J"]+r["E_to_layout_TILE"]+r["F_permute_021"],1)
r["new_chain_ms_per_fold_x40"] = round(r["new_chain_us"]*40/1000,1)
# parity: same values out of both chains, from the SAME source tensor
p_old = ttnn.to_torch(ttnn.permute(ttnn.to_layout(ttnn.reshape(ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT), (I, C*D, J)), ttnn.TILE_LAYOUT), (0,2,1)))
p_new = ttnn.to_torch(zp)
r["torch_equal_old_vs_new"] = bool(torch.equal(p_old, p_new))
r["max_abs_diff"] = float((p_old.float()-p_new.float()).abs().max())
print(json.dumps(r), flush=True)
open("/home/ttuser/.coworker/wt/protenix-trunk--p2-msa-structural/perf/p2_msa_structural/p5_rank_fix.json","w").write(json.dumps(r, indent=1))
