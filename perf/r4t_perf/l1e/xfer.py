import os, time, json, numpy as np
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
dev = T.get_device()
card=int(os.environ["TT_VISIBLE_DEVICES"])
p=f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
f=np.float32
shapes=json.loads(open(os.path.expanduser("~/.coworker/wt/r4t-perf/perf/r4t_perf/rfd4_denoiser_2d_shapes.txt")).read().splitlines()[2])
host=[np.random.default_rng(0).standard_normal((cnt,r,c)).astype(f)*f(1e-3) for (r,c),cnt in shapes]
print("elems",sum(a.size for a in host),flush=True)
def up(): return [ttnn.from_torch(torch.from_numpy(a),dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=dev) for a in host]
d=up(); ttnn.synchronize_device(dev)
for it in range(3):
    t0=time.time(); d=up(); ttnn.synchronize_device(dev); t1=time.time()
    back=[ttnn.to_torch(x) for x in d]; t2=time.time()
    print("fp32 up %.3f down %.3f aiclk %s"%(t1-t0,t2-t1,open(p).read().split()[0]),flush=True)
th=[a.copy() for a in host]; buf=[a.copy() for a in host]
for it in range(2):
    t0=time.time()
    for g,b,t in zip(host,buf,th):
        b += f(0.05)*(g-b); u=b-(b-g)*f(0.05); t[...] = t*f(1-2e-5)+f(-1e-3)*u
    print("host muon elementwise %.3f"%(time.time()-t0),flush=True)
for it in range(2):
    t0=time.time(); w=[ttnn.from_torch(torch.from_numpy(a).to(torch.bfloat16),dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev) for a in th]; ttnn.synchronize_device(dev)
    print("bf16 cast+write %.3f"%(time.time()-t0),flush=True)
