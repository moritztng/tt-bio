"""`perf/roof_tri_arith/roofs.py`, unchanged except for its three hardcodes: the absolute
`/home/agent/wt-triclose` path, the Wormhole-only compute kernel config, and the output name.
Same arms, same harness, so the Blackhole roofs here are comparable to that row's Wormhole ones
arm for arm."""
import os,sys,json,time,platform,torch,ttnn
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"perf","roof_tri_arith"))
from tri_mech import time_arms
import tt_bio.tenstorrent as T
DRAM=ttnn.DRAM_MEMORY_CONFIG; L1=ttnn.L1_MEMORY_CONFIG
# A lone p300 chip is a CUSTOM topology to tt-metal and a bare `ttnn.open_device` TT_FATALs
# on it (tt_cluster.cpp:273). `tt_bio.tenstorrent.get_device` installs the 1x1 p150 mesh
# graph descriptor this box needs, so it is the only legal open here.
d=T.get_device()
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
    kc=(ttnn.types.BlackholeComputeKernelConfig if d.arch()!=ttnn.Arch.WORMHOLE_B0 else ttnn.types.WormholeComputeKernelConfig)(math_fidelity=ttnn.MathFidelity.HiFi4,math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
    c=ttnn.from_torch(torch.randn(4096,4096,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=d,memory_config=DRAM)
    c2=ttnn.from_torch(torch.randn(4096,4096,dtype=torch.bfloat16),layout=ttnn.TILE_LAYOUT,device=d,memory_config=DRAM)
    keep+=[c,c2]
    A["cube4096_HiFi4"]=(lambda: ttnn.matmul(c,c2,compute_kernel_config=kc,memory_config=DRAM),2*4096**3,3)
    A["cube4096_AA"]=A["cube4096_HiFi4"]
    best,err=time_arms(A,list(A),9,d)
    out={"host":platform.node(),"arch":str(d.arch()),"grid":[d.compute_with_storage_grid_size().x,d.compute_with_storage_grid_size().y],
         "loadavg":open("/proc/loadavg").read().split()[:3],"refused":err,
         "rows":[{"arm":n,"ms":v*1e3,"GBps":A[n][1]/v/1e9,"TFLOPs":A[n][1]/v/1e12} for n,v in best.items()]}
    open(os.path.join(HERE,"roofs_bh_qb2c3.json"),"w").write(json.dumps(out,indent=1))
    for r in out["rows"]:
        print("%-28s %9.4f ms  %8.2f GB/s  %7.2f TFLOP/s"%(r["arm"],r["ms"],r["GBps"],r["TFLOPs"]),flush=True)
finally:
    pass
