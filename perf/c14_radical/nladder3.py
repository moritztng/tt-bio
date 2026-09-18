#!/usr/bin/env python3
"""Round 3: the four decisive arms, INTERLEAVED rep by rep with the interior order reversed on
odd reps, plus an A/A twin arm. Rounds 1-2 ran arms in blocks and disagreed by up to 51 % on
merged_full across sessions, so a blocked ladder is not enough here."""
import json, statistics, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk, torch, ttnn  # noqa: E402
MHZ, REPS = 1350, 21
dev = ttnn.open_device(device_id=0)
held = clk.force(MHZ, clk.nodes_open_by_this_process())
for _ in range(200):
    if all(clk.aiclk(n) >= MHZ-5 for n in held): break
    time.sleep(0.05)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
g = dev.compute_with_storage_grid_size(); CG = ttnn.CoreGrid(y=g.y, x=g.x)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
def t(x, mc=L1): return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
B,M,K,H = 16,512,128,512
xn = t(torch.randn(1,B,M,K,dtype=torch.bfloat16))
w1,w2 = t(torch.randn(K,H,dtype=torch.bfloat16),DRAM), t(torch.randn(K,H,dtype=torch.bfloat16),DRAM)
w12 = t(torch.randn(K,2*H,dtype=torch.bfloat16),DRAM)
def lin(w,act=None): return ttnn.linear(xn,w,activation=act,compute_kernel_config=ckc,
    memory_config=L1,dtype=ttnn.bfloat16,core_grid=CG)
def two_linears():
    ttnn.deallocate(lin(w1)); ttnn.deallocate(lin(w2))
def two_linears_aa():
    ttnn.deallocate(lin(w1)); ttnn.deallocate(lin(w2))
def one_linear(): ttnn.deallocate(lin(w12))
def split_unfused():
    a=lin(w1); b=lin(w2); a=ttnn.silu(a,memory_config=L1,output_tensor=a)
    ttnn.deallocate(ttnn.multiply_(a,b)); ttnn.deallocate(b)
def merged_full():
    y=lin(w12)
    a=ttnn.slice(y,[0,0,0,0],[1,B,M,H],memory_config=L1)
    b=ttnn.slice(y,[0,0,0,H],[1,B,M,2*H],memory_config=L1); ttnn.deallocate(y)
    a=ttnn.silu(a,memory_config=L1,output_tensor=a)
    ttnn.deallocate(ttnn.multiply_(a,b)); ttnn.deallocate(b)
ARMS=[("two_linears",two_linears),("two_linears_AA",two_linears_aa),("one_linear",one_linear),
      ("split_unfused",split_unfused),("merged_full",merged_full)]
for _,fn in ARMS: fn()
ttnn.synchronize_device(dev)
s = clk.Sampler(held[0]); time.sleep(0.3)
acc={n:[] for n,_ in ARMS}
for r in range(REPS):
    order = ARMS if r%2==0 else list(reversed(ARMS))
    for n,fn in order:
        ttnn.synchronize_device(dev); t0=time.perf_counter(); fn()
        ttnn.synchronize_device(dev); acc[n].append((time.perf_counter()-t0)*1e3)
res={n:{"ms_min":min(v),"ms_med":statistics.median(v),"ms_p90":sorted(v)[int(.9*len(v))],
        "n":len(v)} for n,v in acc.items()}
res["AA_pct"]=abs(res["two_linears_AA"]["ms_min"]-res["two_linears"]["ms_min"])/res["two_linears"]["ms_min"]*100
res["merge_matmul_ratio"]=res["one_linear"]["ms_min"]/res["two_linears"]["ms_min"]
res["endtoend_ratio"]=res["merged_full"]["ms_min"]/res["split_unfused"]["ms_min"]
out={"arms":res,"grid":[g.y,g.x],"nodes":held,"clock":s.stop(),"reps":REPS,"interleaved":True}
ttnn.close_device(dev)
(HERE/"nladder3_qb1n0.json").write_text(json.dumps(out,indent=1))
print(json.dumps(out,indent=1))
