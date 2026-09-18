#!/usr/bin/env python3
"""Round 2: the merge against the POST-c13-land-first baseline (_UNFUSED_SILU on), plus the
slice cost isolated and an N=1024 A/A redo (round 1's N1024 twin blew its 3 % floor at 31.4 %)."""
import json, os, statistics, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk, torch, ttnn  # noqa: E402

MHZ, REPS = 1350, 15
dev = ttnn.open_device(device_id=0)
nodes = clk.nodes_open_by_this_process()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
g = dev.compute_with_storage_grid_size(); CG = ttnn.CoreGrid(y=g.y, x=g.x)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
def t(x, mc=L1): return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
def timeit(fn, reps=REPS):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev); t0 = time.perf_counter(); fn()
        ttnn.synchronize_device(dev); ts.append((time.perf_counter()-t0)*1e3)
    return min(ts), statistics.median(ts)
held = clk.force(MHZ, nodes)
for _ in range(200):
    if all(clk.aiclk(n) >= MHZ-5 for n in held): break
    time.sleep(0.05)
s = clk.Sampler(held[0]); time.sleep(0.3)
B,M,K,H = 16,512,128,512
xn = t(torch.randn(1,B,M,K,dtype=torch.bfloat16))
w1,w2 = t(torch.randn(K,H,dtype=torch.bfloat16),DRAM), t(torch.randn(K,H,dtype=torch.bfloat16),DRAM)
w12 = t(torch.randn(K,2*H,dtype=torch.bfloat16),DRAM)
def lin(w,act=None,x=None):
    return ttnn.linear(x if x is not None else xn, w, activation=act, compute_kernel_config=ckc,
                       memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG)
def split_unfused():
    x1 = lin(w1); x2 = lin(w2)
    x1 = ttnn.silu(x1, memory_config=L1, output_tensor=x1)
    ttnn.deallocate(ttnn.multiply_(x1,x2)); ttnn.deallocate(x2)
def split_fused():
    x1 = lin(w1,'silu'); x2 = lin(w2)
    ttnn.deallocate(ttnn.multiply_(x1,x2)); ttnn.deallocate(x2)
def merged_full():
    y = lin(w12)
    a = ttnn.slice(y,[0,0,0,0],[1,B,M,H],memory_config=L1)
    b = ttnn.slice(y,[0,0,0,H],[1,B,M,2*H],memory_config=L1)
    ttnn.deallocate(y)
    a = ttnn.silu(a, memory_config=L1, output_tensor=a)
    ttnn.deallocate(ttnn.multiply_(a,b)); ttnn.deallocate(b)
def two_linears():
    ttnn.deallocate(lin(w1)); ttnn.deallocate(lin(w2))
def one_linear():
    ttnn.deallocate(lin(w12))
def one_linear_plus_slices():
    y = lin(w12)
    a = ttnn.slice(y,[0,0,0,0],[1,B,M,H],memory_config=L1)
    b = ttnn.slice(y,[0,0,0,H],[1,B,M,2*H],memory_config=L1)
    for z in (y,a,b): ttnn.deallocate(z)
arms={}
for name,fn in (("split_unfused",split_unfused),("split_fused",split_fused),
                ("merged_full",merged_full),("two_linears",two_linears),
                ("one_linear",one_linear),("one_linear_plus_slices",one_linear_plus_slices)):
    mn,md = timeit(fn); mn2,_ = timeit(fn)
    arms[name]={"ms_min":mn,"ms_med":md,"aa_ms_min":mn2,"aa_pct":abs(mn2-mn)/mn*100}
# bit-exactness of the merge's matmul half: concat(W1,W2) then slice == separate matmuls
wc = torch.cat([torch.randn(K,H,dtype=torch.bfloat16), torch.randn(K,H,dtype=torch.bfloat16)], dim=-1)
wa, wb = wc[:, :H].contiguous(), wc[:, H:].contiguous()
ya = ttnn.to_torch(lin(t(wa,DRAM))); yb = ttnn.to_torch(lin(t(wb,DRAM)))
yc = ttnn.to_torch(lin(t(wc,DRAM)))
arms["bitexact_merge"]={"first_half":bool(torch.equal(yc[...,:H],ya)),
                         "second_half":bool(torch.equal(yc[...,H:],yb))}
out={"arms":arms,"grid":[g.y,g.x],"nodes":held,"clock":s.stop(),"reps":REPS}
ttnn.close_device(dev)
(HERE/"nladder2_qb1n0.json").write_text(json.dumps(out,indent=1))
print(json.dumps(out,indent=1))
