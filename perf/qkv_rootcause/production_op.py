import json, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN
import tt_bio.tenstorrent as T
N, C_Z, H, D = 128, 256, 8, 32
GF = 2*(N*N)*C_Z*(3*H*D)/1e9
def med(x): return sorted(x)[len(x)//2]
def timed(dev, fn, warm=8, pipe=12, reps=5):
    for _ in range(warm): fn()
    ttnn.synchronize_device(dev)
    o=[]
    for _ in range(reps):
        ttnn.synchronize_device(dev); t0=time.perf_counter()
        for _ in range(pipe): fn()
        ttnn.synchronize_device(dev); o.append((time.perf_counter()-t0)*1e3/pipe)
    return med(o)
dev=get_device(); dg=dev.compute_with_storage_grid_size()
ckc=ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
at=torch.randn(1,N*N,C_Z); wt=torch.randn(C_Z,3*H*D)
x=ttnn.from_torch(at,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
w=ttnn.from_torch(wt,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
res={"grid":str(T.COMPUTE_GRID_MAIN)}
cases={
 "minimal_matmul (PRODUCTION)": lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(input_tensor=x,weight_tensor=w,compute_kernel_config=ckc,dtype=ttnn.bfloat16)),
 "ttnn.linear core_grid (the microbench)": lambda: ttnn.deallocate(ttnn.linear(x,w,compute_kernel_config=ckc,core_grid=CORE_GRID_MAIN,memory_config=ttnn.DRAM_MEMORY_CONFIG)),
 "ttnn.linear tall-narrow cfg": lambda: ttnn.deallocate(ttnn.linear(x,w,compute_kernel_config=ckc,memory_config=ttnn.DRAM_MEMORY_CONFIG,**T._matmul_placement(x,w))),
}
for k,f in cases.items():
    ms=timed(dev,f); res[k]={"ms":round(ms,4),"tflops":round(GF/(ms/1e3)/1e3,2)}
    print(f"  {k:40s} {ms:8.4f} ms {GF/(ms/1e3)/1e3:7.2f} TFLOP/s",flush=True)
o1=ttnn.experimental.minimal_matmul(input_tensor=x,weight_tensor=w,compute_kernel_config=ckc,dtype=ttnn.bfloat16)
o2=ttnn.linear(x,w,compute_kernel_config=ckc,memory_config=ttnn.DRAM_MEMORY_CONFIG,**T._matmul_placement(x,w))
t1,t2=ttnn.to_torch(o1).float(),ttnn.to_torch(o2).float()
res["minimal_vs_tallnarrow"]={"bit_exact":bool(torch.equal(t1,t2)),"max_abs":round(float((t1-t2).abs().max()),6)}
print(" ",res["minimal_vs_tallnarrow"],flush=True)
json.dump(res,open(sys.argv[1],"w"),indent=2)
