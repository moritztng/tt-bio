"""Fresh tmpfs file + fresh array (launcher today) vs overwrite of a resident file/array, 714 MB, one process."""
import numpy as np, os, time, json, sys
n=178_448_304; v=np.ones(n,np.float32); d="/dev/shm/r4t-fault"; os.makedirs(d,exist_ok=True)
r={}
def t(f,k=5):
    xs=[]
    for i in range(k):
        t0=time.perf_counter(); f(i); xs.append(time.perf_counter()-t0)
    return sorted(xs)[k//2]
def fresh(i):
    p=f"{d}/f{i}.npy"
    with open(p+".part","wb") as fh: np.save(fh,v)
    os.replace(p+".part",p); os.unlink(p)
np.save(f"{d}/slot.npy",v); m=np.load(f"{d}/slot.npy",mmap_mode="r+")
def reuse(i): m[:]=v
r["publish_fresh_s"]=t(fresh); r["publish_overwrite_s"]=t(reuse)
r["alloc_fresh_fill_s"]=t(lambda i: np.empty(n,np.float32).__setitem__(slice(None),v))
buf=np.empty(n,np.float32); buf[:]=v
r["fill_resident_s"]=t(lambda i: buf.__setitem__(slice(None),v))
r["loadavg"]=open("/proc/loadavg").read().split()[:3]
del m; os.unlink(f"{d}/slot.npy"); os.rmdir(d)
print(json.dumps(r))
