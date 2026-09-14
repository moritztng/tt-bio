import os,sys,json,time,platform,torch,ttnn
sys.path.insert(0,"/home/agent/wt-triclose")
from tri_mech import time_arms
from tt_bio.device_lease import CardSetLease
DRAM=ttnn.DRAM_MEMORY_CONFIG; L1=ttnn.L1_MEMORY_CONFIG
lease=CardSetLease().acquire(); d=ttnn.open_device(device_id=0)
try:
    A={}; keep=[]
    for mb in (16,48):
        rows=(mb*1024*1024//(2048*2)//32)*32; nb=rows*2048*2
        ds=ttnn.from_torch(torch.randn(rows,2048,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=d,memory_config=DRAM)
        ls=ttnn.to_memory_config(ds,L1); keep+=[ds,ls]
        A["write_l1_to_dram_%dMB"%mb]=(lambda ls=ls: ttnn.to_memory_config(ls,DRAM),nb,4)
        A["read_dram_to_l1_%dMB"%mb]=(lambda ds=ds: ttnn.to_memory_config(ds,L1),nb,4)
        A["write_%dMB_AA"%mb]=A["write_l1_to_dram_%dMB"%mb]
        A["read_%dMB_AA"%mb]=A["read_dram_to_l1_%dMB"%mb]
    b=ttnn.from_torch(torch.randn(8192,8192,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=d,memory_config=DRAM)
    b2=ttnn.from_torch(torch.randn(8192,8192,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=d,memory_config=DRAM)
    keep+=[b,b2]
    A["eltwise_2r1w_add8192"]=(lambda: ttnn.add(b,b2,memory_config=DRAM),3*8192*8192*2,3)
    A["eltwise_AA"]=A["eltwise_2r1w_add8192"]
    kc=ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
    c=ttnn.from_torch(torch.randn(4096,4096,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=d,memory_config=DRAM)
    c2=ttnn.from_torch(torch.randn(4096,4096,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=d,memory_config=DRAM)
    keep+=[c,c2]
    A["cube4096_HiFi4"]=(lambda: ttnn.matmul(c,c2,compute_kernel_config=kc,memory_config=DRAM),2*4096**3,3)
    A["cube4096_AA"]=A["cube4096_HiFi4"]
    best,err=time_arms(A,list(A),9,d)
    out={"host":platform.node(),"arch":str(d.arch()),"grid":[d.compute_with_storage_grid_size().x,d.compute_with_storage_grid_size().y],
         "loadavg":open("/proc/loadavg").read().split()[:3],"refused":err,
         "rows":[{"arm":n,"ms":v*1e3,"GBps":A[n][1]/v/1e9,"TFLOPs":A[n][1]/v/1e12} for n,v in best.items()]}
    open("/home/agent/wt-triclose/roofs_whglx_wh.json","w").write(json.dumps(out,indent=1))
    for r in out["rows"]:
        print("%-28s %9.4f ms  %8.2f GB/s  %7.2f TFLOP/s"%(r["arm"],r["ms"],r["GBps"],r["TFLOPs"]),flush=True)
finally:
    ttnn.close_device(d); lease.release()
